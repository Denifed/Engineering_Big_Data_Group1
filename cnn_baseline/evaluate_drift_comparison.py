"""Evaluate paired drift-loss runs with full saved-prediction verification."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_baseline.model_cnn_gru import CNNGRU
from cnn_baseline.dataset import validate_checkpoint
from cnn_baseline.run_fixed_comparison import dump, sha
from cnn_baseline.compare_with_resnet import predict_windows, trajectory_metrics, write_csv
from cnn_baseline.train_drift_comparison import CONDITIONS, PROTOCOL, setup_runtime, bias_metrics
from scripts.run_ronin_npz_baseline import load_sequence, reconstruct
import matplotlib.pyplot as plt

DISPLAY = {"mse_control": "MSE control", "bias_constraint": "MSE + cumulative bias penalty"}
COLORS = {"mse_control": "#357bb6", "bias_constraint": "#c56a22"}
MACRO = ("ATE_official_m", "RTE_60s_official_m", "endpoint_error_m", "mean_velocity_bias_norm_mps")
COMPARE = ("RMSE", "MAE", *MACRO)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_sequence(path, split, stem, t, starts, gt, truth, predictions, trajectories):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    ax.plot(gt[:,0], gt[:,1], color="black", lw=1.1, label="Ground truth")
    for name, position in trajectories.items():
        ax.plot(position[:,0], position[:,1], color=COLORS[name], lw=1, label=DISPLAY[name])
    ax.scatter(*gt[0], c="green", s=20, label="Start")
    ax.set(title=f"{split}/{stem}", xlabel="X [m]", ylabel="Y [m]")
    ax.axis("equal"); ax.legend(fontsize=7)
    times = t[starts]-t[0]
    for c, ax in ((0, axes[0,1]), (1, axes[0,2])):
        ax.plot(times, truth[:,c], color="black", alpha=.65, lw=.6, label="Window-mean target")
        for name, pred in predictions.items():
            ax.plot(times, pred[:,c], color=COLORS[name], lw=.65, alpha=.8, label=DISPLAY[name])
        ax.set(xlabel="Time [s]", ylabel=("Vx" if c==0 else "Vy")+" [m/s]"); ax.legend(fontsize=7)
    delta = t[starts+10]-t[starts]
    for name, pred in predictions.items():
        axes[1,0].plot(t-t[0], np.linalg.norm(trajectories[name]-gt, axis=1), color=COLORS[name], label=DISPLAY[name])
        residual = np.cumsum((pred-truth)*delta[:,None], axis=0)
        axes[1,1].plot(times, np.linalg.norm(residual,axis=1), color=COLORS[name], label=DISPLAY[name])
        axes[1,2].plot(times, np.linalg.norm(residual/np.cumsum(delta)[:,None],axis=1), color=COLORS[name], label=DISPLAY[name])
    axes[1,0].set(xlabel="Time [s]", ylabel="Euclidean position error [m]")
    axes[1,1].set(xlabel="Time [s]", ylabel="Integrated window-mean residual [m]")
    axes[1,2].set(xlabel="Time [s]", ylabel="Time-weighted prefix bias norm [m/s]")
    for ax in axes.flat:
        ax.grid(alpha=.25)
    for ax in axes[1]:
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def summarize(split, name, records, sq, absolute, count):
    return dict(split=split, condition=name, sequences=len(records), windows=count,
        MSE=sq/(2*count), RMSE=math.sqrt(sq/(2*count)), MAE=absolute/(2*count),
        **{key: float(np.mean([r[key] for r in records])) for key in MACRO})


def plot_summaries(out, summaries):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    titles = ("Velocity RMSE [m/s]", "Official ATE [m]", "Official 60s RTE [m]",
              "Velocity MAE [m/s]", "Endpoint error [m]", "Mean sequence bias norm [m/s]")
    keys = ("RMSE", "ATE_official_m", "RTE_60s_official_m", "MAE", "endpoint_error_m", "mean_velocity_bias_norm_mps")
    for ax, key, title in zip(axes.flat, keys, titles):
        for i, name in enumerate(CONDITIONS):
            values=[next(r[key] for r in summaries if r['split']==split and r['condition']==name) for split in ('seen_test','unseen_test')]
            bars=ax.bar(np.arange(2)+(i-.5)*.35,values,.35,label=DISPLAY[name],color=COLORS[name])
            ax.bar_label(bars,fmt="%.4f" if key in ('RMSE','MAE','mean_velocity_bias_norm_mps') else "%.3f",fontsize=8)
        ax.set_xticks(np.arange(2),('seen_test','unseen_test'));ax.set_title(title)
        ax.grid(axis='y',alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'comparison/test_summary.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name in CONDITIONS:
        records=read_csv(out/name/'training_log.csv')
        for ax,key,label in zip(axes,('train_mse','val_mse','val_ATE_official_m'),
                               ('Online training velocity MSE','Validation velocity MSE','Validation ATE [m]')):
            ax.plot([int(r['epoch']) for r in records],[float(r[key]) for r in records],color=COLORS[name],label=DISPLAY[name])
            ax.set(xlabel='Epoch',ylabel=label);ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'comparison/training_comparison.png',dpi=160);plt.close(fig)


def write_report(out, training, summaries, rows, historical, smoke):
    changes=[]
    for split in ('seen_test','unseen_test'):
        a=next(r for r in summaries if r['condition']=='mse_control' and r['split']==split)
        b=next(r for r in summaries if r['condition']=='bias_constraint' and r['split']==split)
        changes.append(dict(split=split,**{key:100*(a[key]-b[key])/a[key] for key in COMPARE}))
    dump(out/'comparison/improvement_percent.json',dict(definition='100*(control_error-improved_error)/control_error; positive means lower error',rows=changes))
    table='\n'.join(f"| {r['split']} | {DISPLAY[r['condition']]} | {r['RMSE']:.6f} | {r['MAE']:.6f} | {r['ATE_official_m']:.3f} | {r['RTE_60s_official_m']:.3f} | {r['endpoint_error_m']:.3f} | {r['mean_velocity_bias_norm_mps']:.6f} |" for r in summaries)
    change_table='\n'.join(f"| {r['split']} | {r['RMSE']:+.2f}% | {r['ATE_official_m']:+.2f}% | {r['RTE_60s_official_m']:+.2f}% | {r['endpoint_error_m']:+.2f}% | {r['mean_velocity_bias_norm_mps']:+.2f}% |" for r in changes)
    train_table='\n'.join(f"| {DISPLAY[name]} | {r['epochs_completed']} | {r['best_epoch']} | {r['best_validation_ATE']:.6f} | {r['best_validation_metrics']['MSE']:.6f} |" for name,r in training.items())
    history_table='\n'.join(f"| {r['split']} | {r['RMSE']:.6f} | {r['ATE_official_m']:.3f} | {r['RTE_60s_official_m']:.3f} | {r['mean_velocity_bias_norm_mps']:.6f} |" for r in historical)
    manifest=read_json(out/'dataset_manifest.json')['train']
    note='本目录仅为两轮小规模流程检查，不能作为正式实验或性能结论。' if smoke else '两组各完整训练 30 轮，以完整验证集最低序列平均 ATE 选择权重。'
    report=f"""# CNN-GRU 连续窗口累计误差约束：配套对照实验

{note}

## 问题与改进

在之前的实验中发现速度误差与长期轨迹漂移未必同时改善。本次保持 CNN-GRU 网络和 78,370 个参数不变，研究连续窗口上增加累计速度残差惩罚是否改善漂移。没有增加推理输入、传感器、注意力或在线真值校正。

误差 e=prediction-target，target 仍是原有重叠窗口平均 XY 速度。对每段前 h=16/32/64 个窗口，计算 b_h=sum(e_i*dt_i)/sum(dt_i)，新损失为 velocity_MSE + lambda*mean(b_h^2)。dt_i=t[start+10]-t[start]，约 0.05 秒，绝不使用约 1 秒的窗口长度。该项是时间相关残差/持续偏差正则，不是精确真值位移监督，不保证轨迹误差下降。

## 配套对照与可比性

- MSE control：lambda=0；改进：lambda=1，预先固定，无测试集调参。
- 两组共用完全相同初始化、网络、64 窗口片段、4 片段/batch、每轮片段顺序、优化步数和验证 ATE 选模规则。
- 片段内部不打乱，片段间打乱；不跨序列和异常过滤缺口，不填充。完整训练可用窗口 {manifest['eligible_windows']:,}，完整片段覆盖 {manifest['used_windows']:,}，排除不完整片段尾部 {manifest['dropped_run_tail_windows']:,} 个窗口。
- 200 帧 IMU 窗口、10 帧步长，Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8，weight_decay=0，seed=42，FP32/确定性算法；无增强、标准化、调度、早停或裁剪。数据文件、划分与既有实验一致。
- 旧三路线实验使用随机单窗口和验证 MSE 选模，不能作为本次损失消融的唯一对照。下方仅将旧 CNN-GRU 放为历史背景，不能把采样/选模差异归因于新损失。
- 验证、测试不筛选，不截取到 64 的倍数，测试覆盖所有有效窗口。模型仍独立处理每个 1 秒窗口；约束通过训练梯度联系相邻预测，不增加推理时的长序列状态。

| 条件 | 完成轮数 | 最佳轮次 | 最佳验证 ATE (m) | 对应验证速度 MSE |
|---|---:|---:|---:|---:|
{train_table}

## 测试结果

| 测试集 | 条件 | RMSE (m/s) | MAE (m/s) | 官方 ATE (m) | 官方 60s RTE (m) | 终点误差 (m) | 平均序列偏差模 (m/s) |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

正数表示相较配套对照的误差下降，负数表示变差：

| 测试集 | RMSE 降低 | ATE 降低 | RTE 降低 | 终点误差降低 | 持续偏差降低 |
|---|---:|---:|---:|---:|---:|
{change_table}

## 历史 CNN-GRU（不同采样与选模，仅作背景）

| 测试集 | RMSE (m/s) | ATE (m) | RTE (m) | 平均序列偏差模 (m/s) |
|---|---:|---:|---:|---:|
{history_table}

## 指标与复核

速度指标对全部测试窗口和 XY 分量平均。ATE/RTE 沿用同一官方坐标误差定义，按序列等权平均，RTE=60 秒。终点误差为每序列末点欧氏误差再等权平均。持续偏差先在每序列计算 sum((prediction-target)*dt)/sum(dt) 的二维向量模，再按序列等权平均。轨迹采用与旧实验相同的 reconstruct()，以首位置初始化，不旋转拟合。累计速度残差诊断图不是直接位置误差图。

全部预测 NPZ 保存后重新读取、复算速度/轨迹/偏差指标，且两条件时间、索引、标签、真值轨迹一致。source_snapshot/、数据与源码指纹、初始化摘要、片段摘要及逐轮顺序摘要用于复核。单个 seed=42 的探索性实验，既有测试集曾用于提出假设，不能据此宣称新测试集上的统计显著性或算法首创。相关思想参考 RoNIN 论文第 4.3 节：https://arxiv.org/html/1905.12853v1 。

index.html 为总览；comparison/ 为配套对照图表；两条件目录分别保留 best/last 权重、日志、验证逐序列结果、seen_test/、unseen_test/ 及预测 NPZ。failed/smoke 目录不属于正式结果。
"""
    (out/'README_实验结果.md').write_text(report,encoding='utf-8')
    table_html=''.join(f"<tr><td>{r['split']}</td><td>{DISPLAY[r['condition']]}</td><td>{r['RMSE']:.5f}</td><td>{r['ATE_official_m']:.3f}</td><td>{r['RTE_60s_official_m']:.3f}</td><td>{r['endpoint_error_m']:.3f}</td><td>{r['mean_velocity_bias_norm_mps']:.5f}</td></tr>" for r in summaries)
    data=[dict(split=r['split'],sequence=r['sequence']) for r in rows if r['condition']=='mse_control']
    page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CNN-GRU 累计误差约束实验</title>
<style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f6f8;color:#182735;max-width:1450px;margin:30px auto;padding:0 20px}section{background:white;border-radius:10px;padding:22px;margin:18px 0}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,td:nth-child(2){text-align:left}select,button{font:inherit;padding:7px;margin:4px}a{color:#17639a}</style>
<h1>CNN-GRU：MSE 与累计速度误差约束</h1><p>__NOTE__</p><p>相同网络、初始化、连续片段与顺序；按验证 ATE 选模；唯一条件差异 lambda=0/1。单种子结果，未用测试集调参。</p>
<section><h2>测试结果</h2><table><tr><th>测试集</th><th>条件</th><th>速度 RMSE</th><th>官方 ATE</th><th>官方 RTE</th><th>终点误差</th><th>持续偏差模</th></tr>__TABLE__</table><p>速度与偏差单位 m/s，轨迹单位 m。<a href="README_实验结果.md">实验说明与完整指标</a> · <a href="comparison/test_metrics.csv">指标 CSV</a> · <a href="comparison/improvement_percent.json">误差变化比例</a> · <a href="result_verification.json">复核记录</a></p></section>
<section><h2>训练与测试汇总</h2><img src="comparison/training_comparison.png"><img src="comparison/test_summary.png"></section>
<section><h2>轨迹、速度与持续偏差</h2><select id="split"><option>seen_test</option><option>unseen_test</option></select><select id="seq"></select><button id="prev">上一条</button><button id="next">下一条</button><img id="chart"><p><a id="control">对照预测 NPZ</a> · <a id="improved">约束组预测 NPZ</a></p></section>
<script>const data=__DATA__,s=document.getElementById('split'),q=document.getElementById('seq');function render(){document.getElementById('chart').src='comparison/'+s.value+'/'+q.value+'.png';document.getElementById('control').href='mse_control/'+s.value+'/'+q.value+'_predictions.npz';document.getElementById('improved').href='bias_constraint/'+s.value+'/'+q.value+'_predictions.npz';}function update(){q.replaceChildren(...data.filter(r=>r.split===s.value).map(r=>{let o=document.createElement('option');o.value=o.textContent=r.sequence;return o;}));render();}s.onchange=update;q.onchange=render;for(const [id,d] of [['prev',-1],['next',1]])document.getElementById(id).onclick=()=>{q.selectedIndex=(q.selectedIndex+d+q.length)%q.length;render();};update();</script></html>"""
    (out/'index.html').write_text(page.replace('__NOTE__',note).replace('__TABLE__',table_html).replace('__DATA__',json.dumps(data)),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.run.resolve();setup_runtime()
    record=read_json(out/'protocol.json');protocol=record['protocol'];digest=record['protocol_sha256']
    expected=dict(PROTOCOL,smoke_only=args.smoke)
    if args.smoke:expected.update(epochs=2,smoke_train_segment_limit=16)
    assert protocol==expected
    assert hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()==digest
    summary=read_json(out/'summary.json')
    assert summary['status']==('smoke_training_completed' if args.smoke else 'training_completed')
    training=summary['training'];checks=read_json(out/'acceptance_checks.json')
    for key in ('same_initialization','same_segments_and_orders','same_optimization_steps','same_validation_ATE_selection','both_checkpoints_reloaded','saved_validation_ATE_recomputed'):
        assert checks[key] is True,key
    orders=read_json(out/'data_order_manifest.json');manifest=read_json(out/'dataset_manifest.json')
    models={}
    for name,weight in CONDITIONS.items():
        r=training[name];assert r['epochs_completed']==protocol['epochs'] and r['lambda_bias']==weight
        log=read_csv(out/name/'training_log.csv')
        assert len(log)==protocol['epochs'] and int(min(log,key=lambda x:float(x['val_ATE_official_m']))['epoch'])==r['best_epoch']
        for epoch,row in enumerate(log,1):
            assert int(row['epoch'])==epoch and row['order_sha256']==orders[str(epoch)]
        saved=torch.load(out/name/'checkpoint_best.pt',map_location='cuda',weights_only=True)
        validate_checkpoint(saved,'cnn_gru',200,10)
        assert saved['condition']==name and saved['lambda_bias']==weight and saved['epoch']==r['best_epoch']
        assert saved['protocol_sha256']==digest
        models[name]=CNNGRU().cuda().eval();models[name].load_state_dict(saved['model_state_dict'],strict=True)
    assert training['mse_control']['initialization_sha256']==training['bias_constraint']['initialization_sha256']
    for p in (out/'comparison',out/'mse_control/seen_test',out/'bias_constraint/seen_test',out/'evaluation_summary.json'):
        if p.exists():raise FileExistsError(f'Refusing overwrite: {p}')
    for relative,value in read_json(out/'source_fingerprints.json').items():
        assert sha(ROOT/relative)==value,f'Training source or split changed: {relative}'
    data_root=ROOT/'RoNIN_Team_Delivery/dataset';fingerprints=read_csv(out/'data_fingerprints.csv')
    for row in fingerprints:
        p=data_root/row['path'];assert p.stat().st_size==int(row['bytes']) and sha(p)==row['sha256']
    (out/'comparison').mkdir()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(out/'evaluation.log',encoding='utf-8')])
    shutil.copy2(__file__,out/'source_snapshot'/Path(__file__).name)
    provenance=dict(script_sha256=sha(Path(__file__)),protocol_sha256=digest,all_training_sources_unchanged=True,
        all_data_files_match_reference=True,checkpoint_sha256={name:sha(out/name/'checkpoint_best.pt') for name in CONDITIONS},
        reconstruction='unchanged scripts/run_ronin_npz_baseline.py:reconstruct',historical_predictions_sha256={})
    history_root=ROOT/'results/fixed_resnet18_vs_cnn_gru_20261025/cnn_gru'
    history_old={(r['split'],r['sequence']):r for r in read_csv(history_root/'sequence_metrics.csv')}
    rows=[];summaries=[];historical=[];history_rows=[];verification={}
    for split in ('seen_test','unseen_test'):
        names=[r['path'] for r in fingerprints if r['split']==split]
        if args.smoke:names=names[:1]
        for folder in (out/'comparison'/split,*[out/name/split for name in CONDITIONS]):folder.mkdir(parents=True)
        totals={name:[0.,0.,0] for name in (*CONDITIONS,'historical')}
        for no,filename in enumerate(names,1):
            f,p,t,_,y=load_sequence(data_root/filename)
            starts=np.arange(0,len(y),10,dtype=np.int64);truth=y[starts].astype(np.float64);gt=p[:,:2];stem=Path(filename).stem
            delta=t[starts+10]-t[starts];predictions={};trajectories={}
            for name,model in models.items():
                pred=predict_windows(model,f,starts,torch.device('cuda'),512);position=reconstruct(t,p,starts,pred)
                predictions[name]=pred;trajectories[name]=position
                row=dict(split=split,sequence=stem,condition=name,**trajectory_metrics(pred,truth,position,gt),**bias_metrics(pred,truth,t,starts))
                rows.append(row);e=pred-truth
                totals[name][0]+=float(np.sum(e**2));totals[name][1]+=float(np.sum(abs(e)));totals[name][2]+=len(truth)
                residual=np.cumsum(e*delta[:,None],axis=0)
                np.savez_compressed(out/name/split/f'{stem}_predictions.npz',starts=starts,
                    prediction_timestamps=t[starts],window_end_timestamps=t[starts+200],predicted_velocity=pred,
                    target_velocity=truth,timestamps=t,predicted_positions=position,ground_truth_positions=gt,
                    prediction_spacing=delta,cumulative_velocity_residual_m=residual,
                    prefix_velocity_bias_norm_mps=np.linalg.norm(residual/np.cumsum(delta)[:,None],axis=1))
                plot_sequence(out/name/split/f'{stem}.png',split,stem,t,starts,gt,truth,{name:pred},{name:position})
                logging.info('TEST %s %s %d/%d %s ATE=%.3f RTE=%.3f bias=%.5f',name,split,no,len(names),stem,row['ATE_official_m'],row['RTE_60s_official_m'],row['mean_velocity_bias_norm_mps'])
            plot_sequence(out/'comparison'/split/f'{stem}.png',split,stem,t,starts,gt,truth,predictions,trajectories)
            old_path=history_root/split/f'{stem}_predictions.npz';provenance['historical_predictions_sha256'][str(old_path)]=sha(old_path)
            with np.load(old_path,allow_pickle=False) as old:
                for key,expected_value in dict(starts=starts,target_velocity=truth,timestamps=t,ground_truth_positions=gt).items():
                    np.testing.assert_array_equal(old[key],expected_value)
                hp=old['predicted_velocity'];ht=old['predicted_positions']
                hm=trajectory_metrics(hp,truth,ht,gt)
                for key,value in hm.items():np.testing.assert_allclose(value,float(history_old[split,stem][key]),rtol=1e-12,atol=1e-12)
                history_rows.append(dict(split=split,sequence=stem,condition='historical',**hm,**bias_metrics(hp,truth,t,starts)))
                e=hp-truth;totals['historical'][0]+=float(np.sum(e**2));totals['historical'][1]+=float(np.sum(abs(e)));totals['historical'][2]+=len(e)
            dump(out/'progress.json',dict(stage='evaluation',split=split,sequence=no,sequences=len(names)))
        for name,(sq,absolute,count) in totals.items():
            selected=[r for r in (history_rows if name=='historical' else rows) if r['split']==split and r['condition']==name]
            r=summarize(split,name,selected,sq,absolute,count)
            (historical if name=='historical' else summaries).append(r)
        # Independently re-open all persisted predictions, comparing both conditions.
        for name in CONDITIONS:
            sq=absolute=0.;count=0;recomputed=[]
            for row in [r for r in rows if r['split']==split and r['condition']==name]:
                path=out/name/split/f"{row['sequence']}_predictions.npz"
                with np.load(path,allow_pickle=False) as saved,np.load(out/'mse_control'/split/path.name,allow_pickle=False) as ref:
                    for key in ('starts','timestamps','target_velocity','ground_truth_positions','prediction_spacing'):
                        np.testing.assert_array_equal(saved[key],ref[key])
                    assert all(np.isfinite(saved[k]).all() for k in saved.files)
                    pred=saved['predicted_velocity'];target=saved['target_velocity'];delta=saved['prediction_spacing']
                    actual=dict(**trajectory_metrics(pred,target,saved['predicted_positions'],saved['ground_truth_positions']),
                                **bias_metrics(pred,target,saved['timestamps'],saved['starts']))
                    for key,value in actual.items():np.testing.assert_allclose(value,row[key],rtol=1e-12,atol=1e-12)
                    e=pred-target;residual=np.cumsum(e*delta[:,None],axis=0)
                    np.testing.assert_array_equal(residual,saved['cumulative_velocity_residual_m'])
                    np.testing.assert_array_equal(np.linalg.norm(residual/np.cumsum(delta)[:,None],axis=1),saved['prefix_velocity_bias_norm_mps'])
                    sq+=float(np.sum(e**2));absolute+=float(np.sum(abs(e)));count+=len(e);recomputed.append(actual)
            actual=summarize(split,name,recomputed,sq,absolute,count)
            expected_summary=next(r for r in summaries if r['split']==split and r['condition']==name)
            for key in ('MSE',*COMPARE):np.testing.assert_allclose(actual[key],expected_summary[key],rtol=1e-12,atol=1e-12)
            verification[f'{name}/{split}']=dict(sequences_checked=len(names),windows=count,saved_metrics_match=True,
                labels_times_indices_gt_identical_between_conditions=True,saved_residual_diagnostics_match=True)
    write_csv(out/'comparison/test_metrics.csv',summaries);write_csv(out/'comparison/sequence_metrics.csv',rows)
    write_csv(out/'comparison/historical_context_metrics.csv',historical)
    for name in CONDITIONS:
        write_csv(out/name/'test_metrics.csv',[r for r in summaries if r['condition']==name])
        write_csv(out/name/'sequence_metrics.csv',[r for r in rows if r['condition']==name])
        training[name].update(status='completed',test_metrics=[r for r in summaries if r['condition']==name])
        dump(out/name/'summary.json',training[name])
    plot_summaries(out,summaries);write_report(out,training,summaries,rows,historical,args.smoke)
    dump(out/'evaluation_config.json',provenance);dump(out/'result_verification.json',verification)
    status='smoke_evaluation_completed' if args.smoke else 'completed'
    dump(out/'evaluation_summary.json',dict(status=status,test_metrics=summaries,historical_context=historical))
    dump(out/'summary.json',dict(status=status,protocol_sha256=digest,training=training,test_metrics=summaries,historical_context=historical))
    checks.update(evaluation_completed=True,saved_test_metrics_recomputed=True,test_sequences=sum(r['sequences'] for r in summaries if r['condition']=='mse_control'))
    dump(out/'acceptance_checks.json',checks);dump(out/'progress.json',dict(stage=status))
    logging.info('COMPLETED %s',json.dumps(summaries))


if __name__=='__main__':main()
