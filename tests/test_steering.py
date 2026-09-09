from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from fastapi.testclient import TestClient

from grm_runtime.common import file_sha, parse_score, progress_step, target_queries
from grm_runtime.config import load_heads, load_steering
from grm_runtime.grounding import AlignmentError, GroundingClient, GroundingError, validate_bbox
from grm_runtime.hf_backend import HFBackend, infer_spans
from grm_runtime.masking import ImageSpan, bbox_to_token_positions, make_attention_mask_hook, make_batched_attention_mask_hook, resolve_negative_positions
from monitor_runtime.core import MonitorState, MonitorConflict
from monitor_runtime.grm_backend import GRMMonitorBackend, _SubtaskState
from monitor_runtime.service import create_app, DeterministicMonitorBackend
from sam3_runtime.service import make_server

ROOT = Path(__file__).resolve().parents[1]


def setUpModule():
    (ROOT / 'results').mkdir(exist_ok=True)


class GeometryTests(unittest.TestCase):
    def test_each_camera_uses_its_own_box_and_size(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as directory:
            paths=[]
            for index in range(8):
                p=Path(directory)/f'{index}.png';Image.new('RGB',(200,100)).save(p);paths.append(str(p))
            from grm_runtime.prompt import IMAGE_LABELS
            spans=[ImageSpan(label,paths[i],i*16,(i+1)*16,(1,8,8)) for i,label in enumerate(IMAGE_LABELS)]
            model=HFBackend.__new__(HFBackend);model.merge=2
            model.config={'intervention_labels':['after_cam_high','after_cam_left_wrist']}
            sample={'task':'pick carrot','image':paths,'grounding':{
                'after_cam_high':{'file_sha256':file_sha(paths[5]),'bbox':[0,0,49,24]},
                'after_cam_left_wrist':{'file_sha256':file_sha(paths[6]),'bbox':[151,76,200,100]}}}
            selected,_,_,missing=model._regions(sample,spans)
            self.assertIsNone(missing);self.assertEqual(selected,[80,111])
            sample['grounding']['after_cam_left_wrist']['file_sha256']='wrong-camera'
            with self.assertRaises(ValueError):model._regions(sample,spans)
    def test_intersection_rectangular_small_and_invalid_grid(self):
        span=ImageSpan('after_cam_high','x',10,26,(1,8,8))
        self.assertEqual(bbox_to_token_positions(span,[0,0,100,50],(200,100),2),[10,11,14,15])
        self.assertEqual(bbox_to_token_positions(span,[24,12,26,14],(200,100),2),[10])
        with self.assertRaises(ValueError):
            bbox_to_token_positions(ImageSpan('x','x',0,15,(1,8,8)),[0,0,5,5],(100,100),2)

    def test_bad_boxes(self):
        for box in ([0,0,float('nan'),3],[10,0,1,1],[200,200,300,300],[1,1,1,2]):
            with self.assertRaises(AlignmentError): validate_bbox(box,(100,100))
        self.assertEqual(validate_bbox([-1,-2,110,105],(100,100)),[0,0,100,100])

    def test_eight_occurrences_not_paths(self):
        ids=[1]
        for _ in range(8): ids += [99]*4+[1]
        inputs={'input_ids':torch.tensor([ids]),'image_grid_thw':torch.tensor([[1,4,4]]*8)}
        spans=infer_spans(inputs,SimpleNamespace(image_token_id=99),['same.png']*8,2)
        self.assertEqual(spans[5].label,'after_cam_high')
        self.assertEqual(spans[5].start,26)
        self.assertNotEqual(spans[2].start,spans[5].start)
        with self.assertRaises(ValueError): infer_spans(inputs,SimpleNamespace(image_token_id=99),['x']*7,2)

    def test_negative_scope(self):
        spans=[ImageSpan('a','x',2,6,(1,4,4)),ImageSpan('b','y',8,12,(1,4,4))]
        negative,labels=resolve_negative_positions(spans,[9],'target_span')
        self.assertEqual(negative,[8,10,11]);self.assertEqual(labels,['b'])
        self.assertEqual(resolve_negative_positions(spans,[9],'other_spans')[0],[2,3,4,5])
        with self.assertRaises(ValueError): resolve_negative_positions(spans,[7],'target_span')


class MaskTests(unittest.TestCase):
    def test_batched_scopes_preserve_each_rows_heads_keys_and_padding(self):
        for scope in ('all', 'prefill', 'last_prompt', 'decode'):
            for query_length in (4, 1):
                with self.subTest(scope=scope, query_length=query_length):
                    mask = torch.zeros(3, 1, query_length, 8)
                    mask[0, :, :, :2] = -float('inf')
                    mask[:, :, :, 7] = -float('inf')
                    specs = [([0], [3], [4], {}), None, ([1], [5], [6], {})]
                    hook = make_batched_attention_mask_hook(specs, 2, 6, query_scope=scope)
                    actual = hook(None, (), {'attention_mask': mask})
                    expected = []
                    for i, spec in enumerate(specs):
                        row = mask[i:i+1]
                        one = make_attention_mask_hook(*spec[:3], 2, 6, {}, query_scope=scope) if spec else None
                        result = one(None, (), {'attention_mask': row}) if one else None
                        expected.append((result[1]['attention_mask'] if result else row).expand(1, 2, query_length, 8))
                    self.assertTrue(torch.equal(actual[1]['attention_mask'] if actual else mask.expand(3, 2, query_length, 8),
                                                torch.cat(expected)))
                    if actual:
                        self.assertTrue(torch.isneginf(actual[1]['attention_mask'][..., 7]).all())
                    with self.assertRaises(RuntimeError):
                        hook(None, (), {'attention_mask': mask[:2]})

    def test_selected_heads_causal_and_decode(self):
        diag={};hook=make_attention_mask_hook([1],[1],[2],3,6,diag)
        mask=torch.zeros(1,1,4,5);mask[...,4]=-float('inf')
        _,out=hook(None,(),{'attention_mask':mask})
        changed=out['attention_mask']
        self.assertTrue(torch.equal(changed[:,0],mask[:,0]))
        self.assertEqual(changed[0,1,0,1],6);self.assertEqual(changed[0,1,0,2],-6)
        self.assertTrue(torch.isneginf(changed[...,4]).all())
        _,out=hook(None,(),{'attention_mask':torch.zeros(1,1,1,9)})
        self.assertTrue((out['attention_mask'][...,5:]==0).all())
        self.assertEqual(diag['prefill_applied_calls'],1);self.assertEqual(diag['decode_applied_calls'],1)

    def test_query_scopes(self):
        for scope in ('prefill','last_prompt','decode'):
            diag={};hook=make_attention_mask_hook([0],[1],[2],2,2,diag,query_scope=scope)
            pre=hook(None,(),{'attention_mask':torch.zeros(1,1,4,6)})
            dec=hook(None,(),{'attention_mask':torch.zeros(1,1,1,7)})
            if scope=='decode': self.assertIsNone(pre);self.assertIsNotNone(dec)
            else: self.assertIsNotNone(pre);self.assertIsNone(dec)
            if scope=='last_prompt':
                self.assertTrue((pre[1]['attention_mask'][0,0,:-1]==0).all())

    def test_zero_and_bad_mask(self):
        mask=torch.randn(1,1,4,6)
        hook=make_attention_mask_hook([0],[1],[2],2,0)
        self.assertTrue(torch.equal(hook(None,(),{'attention_mask':mask})[1]['attention_mask'],mask.expand(1,2,4,6)))
        for value in (None, torch.ones(1,1,1,3,dtype=torch.bool)):
            with self.assertRaises(RuntimeError): hook(None,(),{'attention_mask':value})
        with self.assertRaises(ValueError): make_attention_mask_hook([0],[1],[1],2,1)

    def test_hooks_cleanup_after_partial_install_and_generation_error(self):
        model=HFBackend.__new__(HFBackend)
        layers=[SimpleNamespace(self_attn=torch.nn.Identity()) for _ in range(2)]
        model.model=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
        model.num_heads=2;model.config={'bias':2,'query_scope':'all'}
        from grm_runtime.masking import Head
        for heads in ([Head(0,0),Head(3,0)],[Head(0,0)]):
            with self.assertRaises((IndexError,RuntimeError)):
                with model.hooks(heads,[1],[2],{}): raise RuntimeError('generation failed')
            self.assertFalse(layers[0].self_attn._forward_pre_hooks)


class ProtocolTests(unittest.TestCase):
    def test_compatibility_facade_and_cli_override(self):
        from examples.inference import GRMInference
        with patch('grm_runtime.hf_backend.HFBackend') as backend:
            backend.return_value.inference_batch.return_value=[{'pred':'<score>0%</score>'}]
            model=GRMInference('local-model',engine='hf',steering_config='steering.yaml')
            self.assertEqual(model.inference_batch([{'id':'x'}])[0]['pred'],'<score>0%</score>')
            backend.assert_called_once()
        from monitor_runtime.service import _build_argparser
        parser=_build_argparser({'no_backward':True})
        self.assertTrue(parser.parse_args([]).no_backward)
        self.assertFalse(parser.parse_args(['--backward']).no_backward)
        self.assertEqual(parser.parse_args([]).hf_batch_size, 2)
        self.assertEqual(parser.parse_args(['--hf-batch-size', '1']).hf_batch_size, 1)
    def test_three_source_borda_matches_frozen_grm_order(self):
        from grm_runtime.ranking import consensus
        source=ROOT/'assets/steering'
        actual=consensus([source/f'{task}_source_ranking.json' for task in ('carrot','bottle','cube')],36,32,2)
        expected=json.loads((source/'grm_rawmean12_consensus.json').read_text())
        self.assertEqual([(r['layer'],r['head']) for r in actual['ranking']],
                         [(r['layer'],r['head']) for r in expected['ranking']])
        self.assertEqual(len(actual['ranking']),1088)
    def test_progress_and_strict_scores(self):
        self.assertEqual(parse_score('<score>-25%</score>'),-.25)
        for text in ('0','<score>101%</score>','thinking <score>0%</score>','<score>nan%</score>'):
            with self.assertRaises(ValueError):parse_score(text)
        self.assertAlmostEqual(progress_step('incremental',.5,.4,1)['progress'],.7)
        self.assertAlmostEqual(progress_step('incremental',-.5,.4,1)['progress'],.2)
        self.assertEqual(progress_step('backward',-.2)['progress'],.8)

    def test_task_targets(self):
        self.assertEqual(target_queries('pick the white cube and put it on yellow plate'),['white cube'])
        self.assertEqual(target_queries('  organize   table ',mappings={'organize table':['red can']}),['red can'])
        with self.assertRaises(ValueError):target_queries('整理桌子')
        self.assertEqual(target_queries('整理桌子',['红色罐子']),['红色罐子'])

    def test_profile_and_checkpoint_guard(self):
        cfg=load_steering(ROOT/'configs/steering.yaml')
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as directory:
            path=Path(directory);(path/'config.json').write_text('{}')
            cfg['profile']={**cfg['profile'],'model_path':str(path),'model_config_sha256':file_sha(path/'config.json')}
            high,low=load_heads(cfg,str(path),36,32)
            self.assertEqual(len(high),8);self.assertFalse(set(high)&set(low))
            self.assertEqual((high[0].layer,high[0].head),(19,16))
            with self.assertRaises(ValueError):load_heads(cfg,'another-checkpoint',36,32)
            with self.assertRaises(ValueError):load_heads(cfg,str(path),32,32)
            (path/'config.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):load_heads(cfg,str(path),36,32)

    def test_state_window_compatibility(self):
        state=MonitorState(success_stable_steps=3,success_max_drift=.05)
        for value in [.58,.59,.6]:state.update(value)
        self.assertEqual(state.status,'success')


class GroundingHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'results')
        self.path=Path(self.tmp.name)/'image.png';Image.new('RGB',(32,24),'red').save(self.path)
        class Detector:
            fingerprint='test-model'
            def detect(self,image,queries):return [{'bbox':[2,3,10,12],'score':.9,'query':queries[0]}]
        self.detector=Detector()
        self.server=make_server('127.0.0.1',0,self.detector)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.client=GroundingClient(f'http://127.0.0.1:{self.server.server_port}',cache_dir=Path(self.tmp.name)/'cache')
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()
    def test_png_contract_and_cache(self):
        row=self.client.detect(self.path,['red object'])
        self.assertEqual(row['selected']['bbox'],[2,3,10,12])
        self.assertEqual(row['image_sha256'],file_sha(self.path))
        self.assertEqual(row,self.client.detect(self.path,['red object']))
    def test_model_restart_invalidates_cached_detection(self):
        first=self.client.detect(self.path,['object'])
        self.detector.fingerprint='new-model'
        second=self.client.detect(self.path,['object'])
        self.assertEqual(second['model_fingerprint'],'new-model')
        self.assertNotEqual(first['request_id'],second['request_id'])
    def test_hash_mismatch_rejected(self):
        old=self.client._request
        def bad(route,payload=None):
            value=old(route,payload)
            if route.endswith('detect'):value['image_sha256']='wrong'
            return value
        self.client._request=bad
        with self.assertRaises(AlignmentError):self.client.detect(self.path,['object'])


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'results')
        goal=ROOT/'examples/blank_goal.png'
        self.model=SimpleNamespace(inference_batch=lambda samples:[{**s,'pred':'<score>+20%</score>','valid':True} for s in samples])
        self.backend=GRMMonitorBackend(runtime_url='http://unused',goal_image=str(goal),model=self.model,
            inference_engine='hf',output_root=self.tmp.name,active_modes=['forward','incremental'],interval=.1)
        self.state=_SubtaskState('m','e','pick carrot')
        self.backend.sessions['m']=self.state
        self.ref=self.make_images('reference');self.state.ref_start=self.ref;self.state.previous=self.ref
        self.state.last_observation='old'
        self.current=self.make_images('current')
        self.backend._snapshot_current=lambda *a,**kw:(self.current,{'identity':'new'})
    def make_images(self,name):
        folder=self.backend._session_dir(self.state)/name;folder.mkdir(parents=True,exist_ok=True)
        result={}
        for camera in ('cam_high','cam_left_wrist','cam_right_wrist'):
            p=folder/f'{camera}.png';Image.new('RGB',(32,32)).save(p);result[camera]=str(p)
        return result
    def tearDown(self):self.backend.close();self.tmp.cleanup()
    def test_transaction_and_duplicate_observation(self):
        reference_sha=file_sha(self.ref['cam_high'])
        self.backend._run_one_step(self.state)
        self.assertEqual(self.state.step,1)
        self.assertEqual(file_sha(self.ref['cam_high']),reference_sha)
        # Real snapshots use distinct capture directories, even when observation repeats.
        duplicate=self.make_images('duplicate')
        self.backend._snapshot_current=lambda *a,**kw:(duplicate,{'identity':'new'})
        self.assertIsNone(self.backend._run_one_step(self.state))
        self.assertEqual(self.state.step,1)
    def test_preview_is_exact_committed_input_and_survives_stop(self):
        record = self.backend._run_one_step(self.state)
        frame_id = record['preview']['frame_set_id']
        original = {camera: Path(path).read_bytes() for camera, path in record['frames'].items()}
        self.current = self.make_images('next')
        Image.new('RGB', (32,32), 'red').save(self.current['cam_high'])
        self.backend._snapshot_current = lambda *a, **kw: (self.current, {'identity':'next'})
        newer = self.backend._run_one_step(self.state)
        self.assertNotEqual(newer['preview']['frame_set_id'], frame_id)
        self.assertNotEqual(self.backend.frame_image(newer['preview']['frame_set_id'], 'cam_high'), original['cam_high'])
        with TestClient(create_app(self.backend)) as client:
            client.post('/monitors/stop', json={'monitor_id':'m'})
            for camera, data in original.items():
                response = client.get(f'/monitors/frames/{frame_id}/{camera}.png')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, data)
            self.assertEqual(client.get(f'/monitors/frames/{frame_id}/unknown.png').status_code, 404)
            self.assertEqual(client.get('/monitors/frames/not-registered/cam_high.png').status_code, 404)
    def test_invalid_mode_never_partially_advances(self):
        def output(samples):
            return [{**samples[0],'pred':'<score>+20%</score>','valid':True},
                    {**samples[1],'pred':'broken','valid':False}]
        self.model.inference_batch=output
        with self.assertRaises(RuntimeError):self.backend._run_one_step(self.state)
        self.assertEqual(self.state.step,0);self.assertEqual(self.state.tracker.counts['forward'],0)
        self.assertEqual(self.state.previous,self.ref)
        self.assertFalse(self.backend._preview_frames)
    def test_stop_during_generate_never_publishes(self):
        def output(samples):
            self.state.stop_event.set()
            return [{**s,'pred':'<score>+20%</score>','valid':True} for s in samples]
        self.model.inference_batch=output
        self.assertIsNone(self.backend._run_one_step(self.state));self.assertEqual(self.state.step,0)
        self.assertFalse(self.backend._preview_frames)
    def test_idempotent_start_conflict_and_status_metadata(self):
        old=self.backend.status({'monitor_id':'m'})
        again=self.backend.start({'monitor_id':'m','execution_id':'e','subtask':'pick carrot'})
        self.assertEqual(old.created_at,again.created_at);self.assertIsNone(self.state.thread)
        with self.assertRaises(MonitorConflict):self.backend.start({'monitor_id':'m','execution_id':'x','subtask':'pick carrot'})
        with self.assertRaises(ValueError):self.backend.status({'monitor_id':'m','execution_id':'x'})
    def test_http_contract(self):
        with TestClient(create_app(self.backend)) as client:
            self.assertEqual(client.post('/monitors/start',json={'monitor_id':'m','execution_id':'x','subtask':'pick carrot'}).status_code,409)
            self.assertEqual(client.post('/monitors/status',json={'monitor_id':'m'}).json()['data']['status'],'running')
            self.assertEqual(client.post('/monitors/status',json={'monitor_id':'missing'}).status_code,404)
    def test_deterministic_needs_no_model(self):
        with TestClient(create_app(DeterministicMonitorBackend(auto_success_after_polls=1))) as client:
            client.post('/monitors/start',json={'monitor_id':'d','execution_id':'e','subtask':'task'})
            self.assertEqual(client.post('/monitors/status',json={'monitor_id':'d'}).json()['data']['status'],'success')

    def test_deferred_inference_waits_for_activation(self):
        reference_ready, scored = threading.Event(), threading.Event()
        def snapshot(state, **kwargs):
            reference_ready.set()
            return self.ref, {'identity': 'reference'}
        def score(state):
            scored.set()
            state.stop_event.set()
        self.backend._snapshot_current = snapshot
        self.backend._run_one_step = score
        payload = {'monitor_id': 'deferred', 'execution_id': 'e2',
                   'subtask': 'pick carrot', 'defer_inference': True}
        with TestClient(create_app(self.backend)) as client:
            self.assertEqual(client.post('/monitors/start', json=payload).status_code, 200)
            self.assertTrue(reference_ready.wait(1))
            self.assertFalse(scored.wait(.15))
            status = client.post('/monitors/status', json={'monitor_id': 'deferred'}).json()['data']
            self.assertFalse(status['result']['warming_up'])
            self.assertFalse(status['result']['inference_enabled'])
            self.assertEqual(status['poll_count'], 0)
            self.assertEqual(client.post('/monitors/activate', json={
                'monitor_id': 'deferred', 'execution_id': 'wrong'}).status_code, 409)
            activated = client.post('/monitors/activate', json={'monitor_id': 'deferred', 'execution_id': 'e2'})
            self.assertEqual(activated.status_code, 200)
            self.assertTrue(activated.json()['data']['result']['inference_enabled'])
            self.assertTrue(scored.wait(1))

    def test_activation_requires_reference_and_rejects_stopped_session(self):
        self.state.ref_start = None
        with TestClient(create_app(self.backend)) as client:
            self.assertEqual(client.post('/monitors/activate', json={'monitor_id': 'm'}).status_code, 409)
            client.post('/monitors/stop', json={'monitor_id': 'm'})
            self.assertEqual(client.post('/monitors/activate', json={'monitor_id': 'm'}).status_code, 404)

    def test_terminal_failure_and_success_with_valid_new_observations(self):
        for threshold,minimum,expected in ((.1,.01,'success'),(.9,1.,'failed')):
            self.state.monitor=MonitorState(success_threshold=threshold,success_stable_steps=2,
                success_max_drift=1.,fail_stable_steps=2,fail_min_progress=minimum)
            for index in range(2):
                current=self.make_images(f'{expected}_{index}')
                self.backend._snapshot_current=lambda *a,**kw:(current,{'identity':f'{expected}_{index}'})
                self.backend._run_one_step(self.state)
            self.assertEqual(self.state.monitor.status,expected)

    def test_stop_before_inference_lock_releases_skips_model(self):
        called=[]
        self.model.inference_batch=lambda samples:called.append(samples)
        self.backend._infer_lock.acquire()
        worker=threading.Thread(target=self.backend._run_one_step,args=(self.state,));worker.start()
        self.state.stop_event.set();self.backend._infer_lock.release();worker.join(timeout=3)
        self.assertFalse(worker.is_alive());self.assertFalse(called)


class OfflineTests(unittest.TestCase):
    def test_missing_incremental_hop_breaks_chain_and_keeps_frame_axis(self):
        from examples.offline_steering import run
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            folder=Path(folder)
            for camera in ('cam_high','cam_left_wrist','cam_right_wrist'):
                destination=folder/camera;destination.mkdir()
                for index in range(5):Image.new('RGB',(32,32),(index,0,0)).save(destination/f'{index:06d}.png')
            class Model:
                grounder=None
                config={}
                manifest={'test_only':True}
                def inference_batch(self,samples):
                    return [{**s,'pred':'broken' if s['after_frame_id']==2 else '<score>+20%</score>',
                             'parsed_score':None if s['after_frame_id']==2 else .2,'valid':s['after_frame_id']!=2}
                            for s in samples]
            cfg={'model_path':'unused','steering_config':'unused','data_dir':str(folder),
                 'output_root':str(folder/'output'),'goal_image':str(ROOT/'examples/blank_goal.png'),
                 'task':'pick carrot','target_queries':['carrot'],'fps':5,'frame_interval':1,
                 'modes':['forward','incremental','backward'],'conditions':['baseline'],'visualize':False}
            root=run(cfg,Model())
            results=json.loads((root/'baseline/incremental/pred_vllm.json').read_text())
            self.assertEqual([r['progress'] for r in results],[.2,None,None,None])
            self.assertEqual([r['time_s'] for r in results],[.2,.4,.6,.8])
            self.assertFalse(json.loads((root/'summary.json').read_text())['complete'])


if __name__ == '__main__':unittest.main()
