"""Interleaved device/wall timing after an anomalous initial timing sample."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cnn_baseline.model_cnn_gru import CNNGRU
from cnn_baseline.train_drift_comparison import setup_runtime,state_sha
from cnn_baseline.evaluate_rotation_consistency import aligned_branches
from scripts.run_ronin_npz_baseline import load_sequence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    out=args.run.resolve();target=out/'inference_benchmark_recheck.json'
    if target.exists():raise FileExistsError(target)
    setup_runtime();names=('yaw_augmentation','yaw_consistency');models={};states={}
    for name in names:
        checkpoint=torch.load(out/name/'checkpoint_best.pt',map_location='cpu',weights_only=True)
        model=CNNGRU().cuda().eval();model.load_state_dict(checkpoint['model_state_dict'])
        for p in model.parameters():p.requires_grad_(False)
        models[name]=model;states[name]=state_sha(model)
    f,_,_,_,_=load_sequence(ROOT/'RoNIN_Team_Delivery/dataset/seen_test/a000_11.npz')
    starts=torch.arange(256,device='cuda')*10;ft=torch.from_numpy(f).cuda()
    x=ft[starts[:,None]+torch.arange(200,device='cuda')].transpose(1,2).contiguous()
    tasks=[(n,m) for n in names for m in ('single','c4')]
    def forward(name,mode):
        return models[name](x) if mode=='single' else aligned_branches(models[name],x).double().mean(1)
    values={t:dict(wall_ms=[],device_ms=[]) for t in tasks}
    with torch.inference_mode():
        for _ in range(40):
            for name,mode in tasks:forward(name,mode)
        torch.cuda.synchronize()
        for i in range(60):
            order=tasks if i%2==0 else tasks[::-1]
            for name,mode in order:
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize();tic=time.perf_counter();begin.record()
                forward(name,mode)
                end.record();end.synchronize()
                values[name,mode]['wall_ms'].append((time.perf_counter()-tic)*1000)
                values[name,mode]['device_ms'].append(begin.elapsed_time(end))
    result={}
    for name in names:
        result[name]={}
        for mode in ('single','c4'):
            v=values[name,mode]
            result[name][mode]=dict(batch_size=256,samples=60,
                median_batch_ms=float(np.median(v['wall_ms'])),p10_batch_ms=float(np.percentile(v['wall_ms'],10)),p90_batch_ms=float(np.percentile(v['wall_ms'],90)),
                device_median_batch_ms=float(np.median(v['device_ms'])),device_p10_ms=float(np.percentile(v['device_ms'],10)),device_p90_ms=float(np.percentile(v['device_ms'],90)),
                raw_wall_ms=v['wall_ms'],raw_device_ms=v['device_ms'])
        result[name]['c4_to_single_median_ratio']=result[name]['c4']['median_batch_ms']/result[name]['single']['median_batch_ms']
        result[name]['device_c4_to_single_ratio']=result[name]['c4']['device_median_batch_ms']/result[name]['single']['device_median_batch_ms']
        assert state_sha(models[name])==states[name]
    result['protocol']=dict(reason='Initial A single timing was anomalous relative to A C4 and identical B architecture; retain initial record separately.',
        warmup='40 cycles of all four tasks, then 60 paired measurements per task; reverse task order on alternating cycles',
        inputs='same 256 windows of seen_test/a000_11, starts 0,10,...,2550; input only, no updates',
        scope='GPU forward plus GPU float64 C4 mean, synchronized wall and CUDA Events; excludes host transfer, disk and trajectory',
        original_benchmark_preserved=True,all_model_states_unchanged=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    shutil.copy2(__file__,out/'source_snapshot'/Path(__file__).name)
    print(json.dumps({n:{m:result[n][m]['median_batch_ms'] for m in ('single','c4')} for n in names}))


if __name__=='__main__':main()
