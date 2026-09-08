"""Freeze a raw-mean three-source normalized-Borda profile without loading a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .common import file_sha, fingerprint


def consensus(sources, layers, heads, skip=2):
    if len(sources) != 3:
        raise ValueError("raw-mean-12 protocol requires three independent source rankings")
    if not 0 <= skip < layers:
        raise ValueError("Invalid early-layer exclusion")
    ranks, fingerprints = [], {}
    eligible = {(l,h) for l in range(skip,layers) for h in range(heads)}
    for path in sources:
        data=json.loads(Path(path).read_text())
        rows=data.get('rankings',{}).get('mean')
        if not rows or len(rows)!=layers*heads:
            raise ValueError(f'{path}: expected full mean ranking')
        pairs=[(int(r['layer']),int(r['head'])) for r in rows]
        if set(pairs)!={(l,h) for l in range(layers) for h in range(heads)}:
            raise ValueError(f'{path}: duplicate, invalid or missing heads')
        kept=[pair for pair in pairs if pair in eligible]
        ranks.append({pair:i/max(1,len(kept)-1) for i,pair in enumerate(kept)})
        fingerprints[str(Path(path).resolve())]=file_sha(path)
    if len(set(fingerprints.values()))!=3:
        raise ValueError('Three source artifacts must be distinct')
    rows=[{'layer':l,'head':h,'normalized_borda_rank':mean(r[(l,h)] for r in ranks),
           'source_normalized_ranks':[r[(l,h)] for r in ranks]} for l,h in sorted(eligible)]
    rows.sort(key=lambda r:(r['normalized_borda_rank'],r['layer'],r['head']))
    for i,row in enumerate(rows):row.update(rank=i+1,score=1-row['normalized_borda_rank'])
    result={'num_layers':layers,'num_heads':heads,'skip_early_layers':skip,
            'method':'mean_normalized_borda_rank','ranking':rows,'ranking_fingerprints':fingerprints}
    result['fingerprint']=fingerprint(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',nargs=3,required=True)
    parser.add_argument('--model-path',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--skip-early-layers',type=int,default=2)
    args=parser.parse_args()
    model=Path(args.model_path).resolve();config=json.loads((model/'config.json').read_text())
    text=config.get('text_config',config)
    ranking=consensus(args.sources,text['num_hidden_layers'],text['num_attention_heads'],args.skip_early_layers)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'profile.json').exists() or (out/'consensus_ranking.json').exists():
        raise FileExistsError('Choose a new output directory to preserve the frozen profile')
    (out/'consensus_ranking.json').write_text(json.dumps(ranking,indent=2))
    profile={'schema_version':1,'model_path':str(model),'model_config_sha256':file_sha(model/'config.json'),
             'num_layers':text['num_hidden_layers'],'num_heads':text['num_attention_heads'],
             'ranking_path':'consensus_ranking.json','skip_early_layers':args.skip_early_layers,
             'top_k':8,'bias':6.,'query_scope':'all','negative_scope':'target_span',
             'intervention_labels':['after_cam_high'],'status':'requires task/mode effect validation'}
    (out/'profile.json').write_text(json.dumps(profile,indent=2))


if __name__=='__main__':main()
