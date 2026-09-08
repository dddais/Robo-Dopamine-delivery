"""Opt-in real SAM3/GRM validation with a controlled Robot Runtime HTTP replay.

Run from repository root with GPU 0 for GRM and a separately running SAM3 server.
This validates integration and reproducibility, not robot success accuracy.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

from grm_runtime.common import file_sha
from grm_runtime.hf_backend import HFBackend
from monitor_runtime.grm_backend import GRMMonitorBackend, build_online_samples
from monitor_runtime.service import create_app


def validate(root, output):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    from examples.offline_steering import read_config
    cfg=read_config(root/'configs/offline_steering.yaml')
    model=HFBackend(cfg['model_path'],steering_config=cfg['steering_config'])
    cameras=('cam_high','cam_left_wrist','cam_right_wrist')
    frames={}
    for camera in cameras:
        cap=cv2.VideoCapture(str(Path(cfg['data_dir'])/f'{camera}.mp4'))
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));sequence=[]
        for frame_id in (0,n//2,n-1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id);ok,image=cap.read();assert ok
            sequence.append(cv2.imencode('.jpg',image)[1].tobytes())
        cap.release();frames[camera]=sequence
    control={'index':0}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            index=control['index']
            if self.path.endswith('/metadata'):
                data=json.dumps({'frame_id':index,'timestamp':index,
                    'binary_endpoints':{c:f'/observations/latest/{c}.jpg' for c in cameras}}).encode()
                content='application/json'
            else:
                camera=self.path.rsplit('/',1)[-1].split('.')[0]
                data=frames[camera][index];content='image/jpeg'
            self.send_response(200);self.send_header('Content-Type',content)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Frame-Id',str(index));self.send_header('X-Timestamp',str(index))
            self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    backend=GRMMonitorBackend(model_path=cfg['model_path'],goal_image=cfg['goal_image'],
        runtime_url=f'http://127.0.0.1:{server.server_port}',inference_engine='hf',
        steering_config=cfg['steering_config'],model=model,active_modes=['forward','incremental'],
        output_root=output/'sessions',interval=.2,success_stable_steps=10,fail_stable_steps=10)
    polling=[]
    def wait_for(predicate, timeout=180):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if predicate():return
            time.sleep(.1)
        raise AssertionError('Timed out waiting for monitor: '+str({m:backend.status({'monitor_id':m}).to_dict() for m in backend.sessions}))
    try:
        with TestClient(create_app(backend)) as client:
            for mid in ('a','b'):
                body={'monitor_id':mid,'execution_id':'e','subtask':cfg['task'], 'target_queries':['red can']}
                first=client.post('/monitors/start',json=body);assert first.status_code==200,first.text
                same=client.post('/monitors/start',json=body);assert same.json()['data']['created_at']==first.json()['data']['created_at']
            wait_for(lambda:all(s.ref_start is not None for s in backend.sessions.values()))
            references={mid:{cam:file_sha(path) for cam,path in s.ref_start.items()} for mid,s in backend.sessions.items()}
            snapshots={}
            for step in (1,2):
                control['index']=step
                def done():
                    complete=True
                    for mid in ('a','b'):
                        start=time.monotonic();response=client.post('/monitors/status',json={'monitor_id':mid})
                        polling.append(time.monotonic()-start)
                        payload=response.json()['data']
                        if payload['poll_count']<step:complete=False
                        else:snapshots[(mid,step)]=payload
                    return complete
                wait_for(done)
            assert max(polling)<1.0, max(polling)
            for mid,state in backend.sessions.items():
                assert references[mid]=={cam:file_sha(path) for cam,path in state.ref_start.items()}
                for step in (1,2):
                    for result in snapshots[(mid,step)]['result']['modes'].values():
                        diag=result['steering'];assert diag['applied'] and not diag['degraded']
                        for layer in diag['per_layer'].values():
                            assert layer['prefill_applied_calls']==layer['prefill_calls']>0
                            assert layer['decode_applied_calls']==layer['decode_calls']>0
            a=backend.sessions['a'];b=backend.sessions['b']
            assert backend._session_dir(a)!=backend._session_dir(b)
            # Re-evaluate the exact first committed online frames offline with the same HF backend.
            first=snapshots[('a',1)]['result']
            samples=build_online_samples(a.subtask,0,a.ref_start,backend._ref_end_path,a.ref_start,first['frames'],backend.active_modes)
            for sample in samples:sample['target_queries']=['red can']
            rerun=model.inference_batch(samples)
            for result in rerun:
                assert result['pred']==first['modes'][result['eval_mode']]['pred']
            baseline=[{**samples[0],'condition':'baseline'}]
            before=model.inference_batch(baseline)[0]
            model.inference_batch([samples[0]])
            after=model.inference_batch(baseline)[0]
            assert before['pred']==after['pred']
            assert all(not layer.self_attn._forward_pre_hooks for layer in model.layers)
            for mid in ('a','b'):
                client.post('/monitors/stop',json={'monitor_id':mid})
                assert client.post('/monitors/status',json={'monitor_id':mid}).status_code==404
            import torch
            report={'passed':True,'real_grm':cfg['model_path'],'real_sam3':True,'runtime_transport':'controlled HTTP JPEG replay',
                    'monitors':2,'steps_per_monitor':2,'modes':backend.active_modes,
                    'online_offline_exact_match':True,'baseline_after_steering_match':True,
                    'max_status_latency_s':max(polling),'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'snapshots':{f'{mid}_{step}':value for (mid,step),value in snapshots.items()}}
            (output/'report.json').write_text(json.dumps(report,indent=2))
            print('REAL VALIDATION PASSED',output/'report.json',flush=True)
    finally:
        backend.close();server.shutdown();server.server_close();thread.join()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/validation/online')
    args=parser.parse_args()
    validate(Path(__file__).resolve().parents[1],args.output)
