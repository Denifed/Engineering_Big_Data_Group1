"""Generate the paired yaw-training report, historical comparison and gallery."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
LABELS={'old_single':'旧 CNN-GRU 单次','old_c4':'旧 CNN-GRU C4',
        'augmentation_single':'A：旋转增强 单次','augmentation_c4':'A：旋转增强 C4',
        'consistency_single':'B：旋转一致性 单次','consistency_c4':'B：旋转一致性 C4',
        'pure_cnn':'旧纯 CNN','attention':'旧 CNN-GRU-Attention','resnet18':'RoNIN ResNet18'}
KEYS=('RMSE','ATE_official_m','RTE_60s_official_m','endpoint_error_m','mean_velocity_bias_norm_mps')


def read_json(p):return json.loads(p.read_text(encoding='utf-8'))
def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();out=args.run.resolve()
    audit=read_json(out/'final_audit.json');assert audit['status']=='passed'
    evaluation=read_json(out/'evaluation_summary.json');summary=read_json(out/'summary.json');training=summary['training']
    rows=evaluation['test_metrics'];seq=read_csv(out/'comparison/sequence_metrics.csv');paired=read_json(out/'comparison/paired_sequence_analysis.json')
    effects=read_csv(out/'comparison/improvement_percent.csv')
    timing_path=out/'inference_benchmark_recheck.json'
    timing_document=read_json(timing_path if timing_path.exists() else out/'inference_benchmark.json')
    timings={n:timing_document[n] for n in training}
    comparisons=evaluation['comparisons']
    lookup={(r['split'],r['variant']):r for r in rows}
    descriptive=[]
    for improved,control in (('augmentation_c4','old_c4'),('consistency_c4','augmentation_c4')):
        for split in ('seen_test','unseen_test'):
            a,b=lookup[split,control],lookup[split,improved]
            entry=dict(split=split,improved=improved,control=control,
                       **{k+'_error_reduction_percent':100*(a[k]-b[k])/a[k] for k in KEYS})
            descriptive.append(entry);effects.append(entry)
    with (out/'comparison/additional_descriptive_comparisons.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(descriptive[0]));writer.writeheader();writer.writerows(descriptive)
    a_seen,b_seen=lookup['seen_test','augmentation_single'],lookup['seen_test','consistency_single']
    a_unseen,b_unseen=lookup['unseen_test','augmentation_single'],lookup['unseen_test','consistency_single']
    seen_change=100*(b_seen['ATE_official_m']-a_seen['ATE_official_m'])/a_seen['ATE_official_m']
    unseen_reduction=100*(a_unseen['ATE_official_m']-b_unseen['ATE_official_m'])/a_unseen['ATE_official_m']
    direct_pairs={r['split']:r for r in paired if r['improved']=='consistency_single' and r['control']=='augmentation_single'}
    seen_pair=direct_pairs['seen_test'];unseen_pair=direct_pairs['unseen_test']
    disagreement={(r['split'],r['condition']):float(r['rotation_disagreement_RMSE_mps']) for r in read_csv(out/'comparison/rotation_disagreement.csv')}
    disagreement_table='\n'.join(f"| {s} | {disagreement[s,'yaw_augmentation']:.6f} | {disagreement[s,'yaw_consistency']:.6f} | {100*(disagreement[s,'yaw_augmentation']-disagreement[s,'yaw_consistency'])/disagreement[s,'yaw_augmentation']:.2f}% |" for s in ('seen_test','unseen_test'))
    loss_success=comparisons['consistency_single_vs_augmentation_single']['overall_improvement']
    old_success=comparisons['consistency_single_vs_old_single']['overall_improvement']
    augmentation_verdict=('旋转增强训练方案本身已满足相对旧CNN-GRU单次路线的整体改善条件' if comparisons['augmentation_single_vs_old_single']['overall_improvement'] else '旋转增强训练方案未满足相对旧CNN-GRU单次路线的整体改善条件')
    a_c4_better=all(lookup[s,'augmentation_c4'][k]<lookup[s,'consistency_c4'][k] for s in ('seen_test','unseen_test') for k in (*KEYS,'MAE'))
    ensemble_verdict=('本次A+C4的两套测试集平均指标均略低于B+C4' if a_c4_better else 'A+C4与B+C4的差异需结合下方全部指标逐项判断')
    timing_description=('对四个任务充分预热40个循环，再交错测量每个任务60次，并在相邻循环反转测量顺序；同时保存同步墙钟时间和CUDA Events设备时间' if timing_path.exists() else '各任务预热10次后测量30次，同步墙钟计时')
    verdict=(f"相对旋转增强对照，一致性损失{'满足' if loss_success else '未满足'}预设整体改善条件；"
             f"相对旧 CNN-GRU 单次路线，一致性训练模型{'满足' if old_success else '未满足'}该条件。应依据以下完整指标区分增强收益、一致性损失额外收益和 C4 推理收益。")
    if audit['smoke_only']:verdict='本目录为两轮小规模流程检查，不属于正式实验；以下数字仅用于检查保存和分析流程。'+verdict
    history_table=[]
    for split in ('seen_test','unseen_test'):
        for name in ('pure_cnn','old_single','attention','resnet18','old_c4','augmentation_single','consistency_single','augmentation_c4','consistency_c4'):
            r=lookup[split,name]
            history_table.append(f"| {split} | {LABELS[name]} | {r['RMSE']:.6f} | {r['MAE']:.6f} | {r['ATE_official_m']:.4f} | {r['RTE_60s_official_m']:.4f} | {r['endpoint_error_m']:.4f} | {r['mean_velocity_bias_norm_mps']:.6f} |")
    training_table='\n'.join(f"| {name} | {r['epochs_completed']} | {r['best_epoch']} | {r['best_val_mse']:.7f} | {r['best_validation_metrics']['ATE_official_m']:.4f} | {r['best_ATE_epoch_diagnostic']} | {r['training_wall_seconds']/60:.2f} |" for name,r in training.items())
    effect_table=[]
    for r in effects:
        effect_table.append(f"| {r['split']} | {LABELS[r['improved']]} / {LABELS[r['control']]} | "+' | '.join(f"{float(r[k+'_error_reduction_percent']):+.2f}%" for k in KEYS)+' |')
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for name,color in (('yaw_augmentation','#2a8c66'),('yaw_consistency','#d86b22')):
        log=read_csv(out/name/'training_log.csv');epochs=[int(r['epoch']) for r in log]
        for ax,key in zip(axes.flat,('train_supervised_mse','train_consistency_mse','train_total_loss','val_mse','val_ATE_official_m','val_bias_norm_mps')):
            ax.plot(epochs,[float(r[key]) for r in log],color=color,label=name);ax.set(xlabel='Epoch',title=key);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'comparison/training_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5));example_rows=[]
    for ax,split in zip(axes,('seen_test','unseen_test')):
        a={r['sequence']:r for r in seq if r['split']==split and r['variant']=='augmentation_single'}
        b={r['sequence']:r for r in seq if r['split']==split and r['variant']=='consistency_single'}
        av=np.array([float(a[n]['ATE_official_m']) for n in a]);bv=np.array([float(b[n]['ATE_official_m']) for n in a]);limit=max(av.max(),bv.max())*1.05
        ax.scatter(av,bv,c=np.where(bv<av,'#238c6b','#c65f3c'),s=30);ax.plot([0,limit],[0,limit],'--',c='gray')
        ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Augmentation-only ATE [m]',ylabel='Consistency ATE [m]',title=f'{split}: improved {int((bv<av).sum())}/{len(av)}')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
        delta=sorted([(float(a[n]['ATE_official_m'])-float(b[n]['ATE_official_m']),n) for n in a])
        for label,(change,n) in (('largest deterioration',delta[0]),('largest reduction',delta[-1])):
            example_rows.append(dict(split=split,sequence=n,type=label,ATE_augmentation=float(a[n]['ATE_official_m']),ATE_consistency=float(b[n]['ATE_official_m']),ATE_reduction_m=change))
    fig.tight_layout();fig.savefig(out/'comparison/consistency_vs_augmentation_scatter.png',dpi=160);plt.close(fig)
    dump_path=out/'comparison/representative_examples.json';dump_path.write_text(json.dumps(example_rows,indent=2,ensure_ascii=False),encoding='utf-8')
    distribution=[]
    for r in paired:
        if r['improved']=='consistency_single' and r['control'] in ('augmentation_single','old_single','old_c4'):
            for key in ('ATE_official_m','endpoint_error_m','mean_velocity_bias_norm_mps'):
                m=r['metrics'][key]
                distribution.append(f"| {r['split']} | {LABELS[r['control']]} | {key} | {m['improved']}/{r['sequences']} | {m['median_control']:.4f} → {m['median_improved']:.4f} | {m['p90_control']:.4f} → {m['p90_improved']:.4f} |")
    example_table='\n'.join(f"| {r['split']} | {r['sequence']} | {r['type']} | {r['ATE_augmentation']:.4f} → {r['ATE_consistency']:.4f} |" for r in example_rows)
    criterion_table='\n'.join(f"| {k} | {'满足' if v['overall_improvement'] else '未满足'} |" for k,v in comparisons.items())
    benchmark_table='\n'.join(f"| {n} | {r['single']['median_batch_ms']:.4f} | {r['c4']['median_batch_ms']:.4f} | {r['c4_to_single_median_ratio']:.2f} |" for n,r in timings.items())
    report=f"""# CNN-GRU 旋转一致性训练与旧路线对比

## 正式结论

{verdict}

本次固定lambda=0.1时，B相对A的unseen平均ATE降低{unseen_reduction:.2f}%，但seen平均ATE升高{seen_change:.2f}%。seen有{seen_pair['metrics']['ATE_official_m']['improved']}/{seen_pair['sequences']}条轨迹ATE改善；unseen有{unseen_pair['metrics']['ATE_official_m']['improved']}/{unseen_pair['sequences']}条改善，但unseen ATE的P90从{unseen_pair['metrics']['ATE_official_m']['p90_control']:.4f}变为{unseen_pair['metrics']['ATE_official_m']['p90_improved']:.4f}m，需同时考虑尾部表现。

{augmentation_verdict}，A相对旧模型的变化不能归于一致性损失。{ensemble_verdict}，因此需要分别判断训练损失和推理集成的作用。最优均值属于已测条件的描述性比较，不是独立新测试集验证后的模型选择结论。

所有测试指标均由保存的完整预测重新计算。主 checkpoint 仅按未增强的完整验证集 MSE 选择；保存的最佳 ATE 权重仅供诊断，本轮没有用它们做测试或在两种选模方式间挑结果。

## 固定方案与可比性

- A：yaw_augmentation，lambda=0；B：yaw_consistency，lambda=0.1。两组均从 seed42 的同一 CNN-GRU 初始化训练，结构仍为 78,370 个参数；各 {training['yaw_augmentation']['epochs_completed']} 轮。
- 原输入 x 和随机绕 Z 轴旋转的 Rx 一同输入网络，标签为 y 与 Ry。同步旋转 gyro/acc XY，保持 Z、时间、窗口标签定义不变。
- `Lsup=0.5*(MSE(f(x),y)+MSE(f(Rx),Ry))`；`Leq=MSE(f(Rx),R*f(x))`；B 使用 `Lsup+0.1*Leq`，两个预测分支均接收梯度，A 仅使用 Lsup。
- 原始窗口 batch=256，拼接旋转副本后网络 batch=512（末批按实际大小）；每窗口独立随机角度 U[0,2pi)。两组共用相同窗口顺序、角度、初始化、优化步数及 BN 批次组织，唯一实验条件差异是一致性损失权重。
- 原始样本顺序逐轮与旧固定 CNN-GRU 实验哈希一致；相同训练/验证/测试数据和异常过滤，200 帧输入、10 帧步长、窗口平均全局 XY 速度标签。
- Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0；FP32/确定性算法，无调度、早停、标准化、额外噪声或 BN 重校准。
- 与旧未增强模型的比较衡量整体训练方案，包含双分支增强和网络批次大小变化；A/B 才是隔离一致性损失的直接消融。30 轮不等于与旧路线相同计算预算，新的每个原始样本还多处理一个旋转副本。
- 单次和 C4 都使用各组相同的主 checkpoint；C4 固定 0/90/180/270 度逆旋转后等权平均，不调融合权重，不进行测试时训练。

| 条件 | 训练轮数 | 主 checkpoint 轮次 | 最佳验证 MSE | 对应验证 ATE(m) | 最佳 ATE 诊断轮次 | 训练耗时(min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{training_table}

耗时是实际墙钟时间。A组第3轮曾因并行辅助检查造成显存拥堵，随后已停止辅助进程并恢复正常训练；这部分等待包含在A组耗时中，因此A/B的耗时差不能直接解释为损失函数的计算效率差异。后续GPU任务均串行执行。

## 与旧路线的全部测试指标

RMSE/MAE/速度偏置单位 m/s，ATE/RTE/终点误差单位 m。速度指标按窗口和坐标分量平均，轨迹指标对序列等权平均。ATE/RTE 沿用官方坐标误差定义，RTE=60秒；轨迹重建完全复用原实现，以真值首位置初始化坐标原点，不进行轨迹旋转拟合。

| 测试集 | 路线 | RMSE | MAE | 官方ATE | 官方60sRTE | 终点误差 | 平均速度残差偏置 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(history_table)}

旧路线结果读取并复算其已有全量预测，旧纯CNN、GRU、Attention和ResNet18均来自此前固定30轮协议，旧C4来自已完成的固定四方向集成实验。旧模型未在本轮重新训练。所有参考文件的标签、窗口索引、时间戳和真值轨迹逐元素核对。

## 关键对照：正数表示误差下降，负数表示退化

| 测试集 | 改进组 / 对照组 | RMSE降低 | ATE降低 | RTE降低 | 终点误差降低 | 偏置降低 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(effect_table)}

不能将 A 相比旧模型的收益全部归于 Leq。若 B 不优于 A，应将主要收益归于增强训练方案；若 B 优于 A 才能支持一致性项的额外作用。C4 的收益需与单次推理分别说明。

整体改善的预设条件：seen/unseen 平均 ATE 都降低，且 unseen 终点误差、速度偏置与 RMSE 不劣于对应对照。

| 比较 | 是否满足整体改善条件 |
| --- | --- |
{criterion_table}

## 逐序列与尾部

| 测试集 | B单次的对照 | 指标 | 改善序列数 | 中位数：对照 → B | P90：对照 → B |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(distribution)}

下表为 A/B 之间按绝对 ATE 变化选出的最大改善、最大退化案例，全部轨迹均在浏览页保存，不以个例代替总体指标：

| 测试集 | 序列 | 类型 | ATE：A → B |
| --- | --- | --- | --- |
{example_table}

`comparison/rotation_disagreement.csv` 报告四方向逆旋转后与原预测的不一致程度。该值降低仅表明一致性增强，需要与准确率指标一起解释；低不一致甚至输出零也不自动意味着定位准确。

| 测试集 | A四方向不一致RMSE(m/s) | B四方向不一致RMSE(m/s) | 不一致程度下降 |
| --- | ---: | ---: | ---: |
{disagreement_table}

本次一致性项确实降低了上述不一致程度，但seen的平均速度残差偏置仍从{a_seen['mean_velocity_bias_norm_mps']:.6f}升至{b_seen['mean_velocity_bias_norm_mps']:.6f}m/s，终点误差从{a_seen['endpoint_error_m']:.4f}升至{b_seen['endpoint_error_m']:.4f}m。旋转关系更一致不等于残差均值更接近零，这能解释为何要同时监测物理一致性、速度误差、累计误差，而不能仅凭Leq下降认定定位更准。

## 推理成本

batch=256，{timing_description}，下表为墙钟中位数。C4聚合计时在GPU进行，实际保存预测的均值在CPU以float64计算，不含输出回传、磁盘、轨迹和画图，不代表端到端时延：

| 模型 | 单次 ms/batch | C4 ms/batch | C4/单次 |
| --- | ---: | ---: | ---: |
{benchmark_table}

最初顺序计时中，A组出现C4耗时小于单次预测、且相同结构的A/B单次耗时相差过大的异常，不能作为稳定开销估计。原始记录完整保留在inference_benchmark.json；正式目录中若存在inference_benchmark_recheck.json，则以上表格使用其充分预热、交错测量的复核记录，包含全部原始采样和CUDA设备时间，不修改准确率结果或权重。

新模型的单次预测仍只需要一次 CNN-GRU 前向，未增加推理参数。训练成本因配对旋转样本增加，C4推理仍需四次前向。

## 研究边界

旋转等变性已有相关惯性里程计研究，参考 RIO：https://arxiv.org/abs/2111.11676 。本项目使用双分支速度监督与速度预测的旋转一致性MSE约束，是该物理思想的项目应用，不构成完整RIO复现或首创算法。

仅一个训练seed42，测试集此前已用于多次项目分析，属于探索性实验，不能宣称独立盲测或统计显著性。模型仍独立处理约1秒IMU窗口，未引入绝对位置观测或长时状态；误差改善不意味着长期漂移消失或有界。超参数0.1和30轮预先固定，未依据测试修改。

## 交付与复核

- `PLAN_执行计划.md`、`protocol.json`、`environment.json`、数据/源码/旧文件指纹、`source_snapshot/`。
- 两组训练目录：config、逐轮CSV、曲线、主最佳MSE/最佳ATE诊断/last权重、主最佳及last验证逐序列指标、summary。
- 四个新推理条件目录：每条测试序列完整预测NPZ；C4目录还保存四方向分支输出。
- `comparison/`：9条路线的全量指标、相对误差变化、逐序列统计、旋转不一致程度、67张轨迹/速度/误差图、训练曲线与汇总图。
- `additional_descriptive_comparisons.csv`位于comparison目录，补列A+C4相对旧C4、B+C4相对A+C4的已测结果描述性比较，不构成额外调参或新训练。
- `final_audit.json`、`result_verification.json`、`analysis_audit.json`：权重重载、保存预测复算、轨迹重建、历史标签对齐、原始文件不变等核验记录。
- `index.html`：查看所有测试轨迹及失败案例。

训练：`E:\\conda_envs\\torch_env\\python.exe -B cnn_baseline/train_rotation_consistency.py --out results/新的目录`；中断后同目录加 `--resume`。
评估：`E:\\conda_envs\\torch_env\\python.exe -B cnn_baseline/evaluate_yaw_training.py --run results/训练目录`。
分析：`E:\\conda_envs\\torch_env\\python.exe -B cnn_baseline/analyze_yaw_training.py --run results/训练目录`。
"""
    (out/'README_实验结果.md').write_text(report,encoding='utf-8')
    options=[dict(split=r['split'],sequence=r['sequence'],ATE_a=float(r['ATE_official_m'])) for r in seq if r['variant']=='augmentation_single']
    for r in options:r['ATE_b']=float(next(s['ATE_official_m'] for s in seq if s['split']==r['split'] and s['sequence']==r['sequence'] and s['variant']=='consistency_single'))
    table=''.join(f"<tr><td>{r['split']}</td><td>{LABELS[r['variant']]}</td><td>{r['RMSE']:.5f}</td><td>{r['ATE_official_m']:.3f}</td><td>{r['RTE_60s_official_m']:.3f}</td><td>{r['endpoint_error_m']:.3f}</td><td>{r['mean_velocity_bias_norm_mps']:.5f}</td></tr>" for r in rows)
    page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CNN-GRU旋转一致性训练</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1500px;margin:28px auto;padding:0 20px;color:#193044;background:#f4f6f8}section{background:white;padding:22px;margin:20px 0;border-radius:10px}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,td:nth-child(2){text-align:left}select,button{font:inherit;padding:7px;margin:5px}a{color:#17689e}</style>
<h1>CNN-GRU：旋转增强与旋转一致性训练</h1><p>__VERDICT__</p><p>同初始化、窗口顺序、随机旋转、BN批次和优化步数；A为旋转增强，B增加lambda=0.1一致性项。统一按验证MSE选模。</p>
<section><h2>新旧路线指标</h2><table><tr><th>测试集</th><th>路线</th><th>RMSE</th><th>ATE</th><th>RTE</th><th>终点误差</th><th>速度偏置</th></tr>__TABLE__</table><p><a href="README_实验结果.md">完整报告</a> · <a href="PLAN_执行计划.md">固定计划</a> · <a href="comparison/test_metrics.csv">指标CSV</a> · <a href="final_audit.json">核验</a></p></section>
<section><h2>训练与全量对比</h2><img src="comparison/training_comparison.png"><img src="comparison/single_pass_routes.png"><img src="comparison/main_test_summary.png"><img src="comparison/consistency_vs_augmentation_scatter.png"></section>
<section><h2>全部测试序列</h2><select id="split"><option>seen_test</option><option>unseen_test</option></select><select id="seq"></select><button id="prev">上一条</button><button id="next">下一条</button><p id="caption"></p><img id="chart"><p><a id="a">A单次预测NPZ</a> · <a id="b">B单次预测NPZ</a></p></section>
<script>const data=__DATA__,s=document.getElementById('split'),q=document.getElementById('seq');function render(){const r=data.find(r=>r.split===s.value&&r.sequence===q.value);document.getElementById('chart').src='comparison/'+s.value+'/'+q.value+'.png';document.getElementById('caption').textContent='单次预测ATE：A '+r.ATE_a.toFixed(3)+' m → B '+r.ATE_b.toFixed(3)+' m';for(const [id,folder] of [['a','augmentation_single'],['b','consistency_single']])document.getElementById(id).href=folder+'/'+s.value+'/'+q.value+'_predictions.npz';}function update(){q.replaceChildren(...data.filter(r=>r.split===s.value).map(r=>{const o=document.createElement('option');o.value=o.textContent=r.sequence;return o;}));render();}s.onchange=update;q.onchange=render;for(const [id,d] of [['prev',-1],['next',1]])document.getElementById(id).onclick=()=>{q.selectedIndex=(q.selectedIndex+d+q.length)%q.length;render();};update();</script></html>"""
    (out/'index.html').write_text(page.replace('__VERDICT__',verdict).replace('__TABLE__',table).replace('__DATA__',json.dumps(options)),encoding='utf-8')
    shutil.copy2(__file__,out/'source_snapshot'/Path(__file__).name)
    from PIL import Image
    for path in (out/'comparison').rglob('*.png'):
        with Image.open(path) as im:im.verify()
    for link in re.findall(r'(?:href|src)="([^"]+)"',(out/'index.html').read_text(encoding='utf-8')):assert (out/link).is_file(),link
    for r in options:
        assert (out/'comparison'/r['split']/f"{r['sequence']}.png").is_file()
        for name in ('augmentation_single','consistency_single'):assert (out/name/r['split']/f"{r['sequence']}_predictions.npz").is_file()
    extra=dict(status='passed',report_sha256=sha(out/'README_实验结果.md'),analysis_script_sha256=sha(Path(__file__)),
               gallery_links_verified=True,all_test_sequences_in_gallery=len(options),plot_files=len(list((out/'comparison').rglob('*.png'))))
    (out/'analysis_audit.json').write_text(json.dumps(extra,indent=2,ensure_ascii=False),encoding='utf-8')
    summary.update(status='smoke_completed' if audit['smoke_only'] else 'completed',
                   evaluation_summary='evaluation_summary.json',analysis_report='README_实验结果.md',
                   consistency_loss_overall_improvement=loss_success,consistency_vs_old_overall_improvement=old_success)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'progress.json').write_text(json.dumps(dict(stage=summary['status']),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='analysis_completed',verdict=verdict,comparisons=comparisons,
                         single_pass_test_metrics=[r for r in rows if r['variant'] in ('old_single','old_c4','augmentation_single','consistency_single')]),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
