"""Evaluate the fixed yaw-training pair and compare all previous model routes."""
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
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cnn_baseline.dataset import validate_checkpoint
from cnn_baseline.model_cnn_gru import CNNGRU
from cnn_baseline.run_fixed_comparison import dump,sha
from cnn_baseline.train_rotation_consistency import CONDITIONS,PROTOCOL
from cnn_baseline.train_drift_comparison import setup_runtime,state_sha,bias_metrics,validate
from cnn_baseline.sequence_dataset import validation_sequences
from cnn_baseline.dataset import RoNINNPZDataset
from cnn_baseline.run_fixed_comparison import FixedWindows
from cnn_baseline.compare_with_resnet import trajectory_metrics,write_csv
from cnn_baseline.evaluate_rotation_consistency import infer,benchmark,summarize
from scripts.run_ronin_npz_baseline import load_sequence,reconstruct

OLD=ROOT/'results/fixed_resnet18_vs_cnn_gru_20261025'
C4=ROOT/'results/cnn_gru_rotation_consistency_20261026'
NEW_VARIANTS=('augmentation_single','augmentation_c4','consistency_single','consistency_c4')
MAIN=('old_single','old_c4',*NEW_VARIANTS)
REFERENCES={
    'old_single':OLD/'cnn_gru',
    'old_c4':C4/'original_c4',
    'pure_cnn':ROOT/'results/cnn_three_routes_fixed_20261026/cnn',
    'attention':ROOT/'results/cnn_gru_attention_fixed_20261026/cnn_gru_attention',
    'resnet18':OLD/'resnet18',
}
LABELS={'old_single':'Old CNN-GRU single','old_c4':'Old CNN-GRU C4',
        'augmentation_single':'Yaw augmentation single','augmentation_c4':'Yaw augmentation C4',
        'consistency_single':'Yaw consistency single','consistency_c4':'Yaw consistency C4',
        'pure_cnn':'Pure CNN','attention':'CNN-GRU-Attention','resnet18':'RoNIN ResNet18'}
COLORS=dict(zip(MAIN,('#777777','#a487ba','#2a8c66','#8ec8a8','#d86b22','#f0b56d')))
MACRO=('ATE_official_m','RTE_60s_official_m','endpoint_error_m','mean_velocity_bias_norm_mps')
KEYS=('RMSE','MAE',*MACRO)


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def metrics(pred,target,position,gt,t,starts):
    return dict(**trajectory_metrics(pred,target,position,gt),**bias_metrics(pred,target,t,starts))


def plot_sequence(path,split,stem,t,starts,target,gt,predictions,trajectories):
    fig,axes=plt.subplots(2,3,figsize=(17,9));ts=t[starts]-t[0]
    axes[0,0].plot(gt[:,0],gt[:,1],c='black',lw=1.2,label='Ground truth')
    for name in MAIN:
        pos=trajectories[name]
        axes[0,0].plot(pos[:,0],pos[:,1],c=COLORS[name],lw=.9,label=LABELS[name])
        axes[1,0].plot(t-t[0],np.linalg.norm(pos-gt,axis=1),c=COLORS[name],lw=.8)
        residual=np.cumsum((predictions[name]-target)*(t[starts+10]-t[starts])[:,None],axis=0)
        axes[1,1].plot(ts,np.linalg.norm(residual,axis=1),c=COLORS[name],lw=.8)
        axes[1,2].plot(ts,np.linalg.norm(residual/np.cumsum(t[starts+10]-t[starts])[:,None],axis=1),c=COLORS[name],lw=.8)
    axes[0,0].set(title=f'{split}/{stem}',xlabel='X [m]',ylabel='Y [m]');axes[0,0].axis('equal');axes[0,0].legend(fontsize=6)
    for j in range(2):
        ax=axes[0,j+1];ax.plot(ts,target[:,j],c='black',lw=.6,label='Target')
        for name in ('old_single','augmentation_single','consistency_single'):
            ax.plot(ts,predictions[name][:,j],c=COLORS[name],lw=.6,label=LABELS[name])
        ax.set(xlabel='Time [s]',ylabel=('Vx' if j==0 else 'Vy')+' [m/s]');ax.legend(fontsize=6)
    axes[1,0].set(xlabel='Time [s]',ylabel='Euclidean position error [m]')
    axes[1,1].set(xlabel='Time [s]',ylabel='Integrated window-mean residual [m]')
    axes[1,2].set(xlabel='Time [s]',ylabel='Time-weighted prefix bias [m/s]')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def plot_summary(path,records,variants):
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for ax,key,title in zip(axes.flat,('RMSE','ATE_official_m','RTE_60s_official_m','MAE','endpoint_error_m','mean_velocity_bias_norm_mps'),
                           ('Velocity RMSE [m/s]','Official ATE [m]','60s RTE [m]','Velocity MAE [m/s]','Endpoint error [m]','Mean velocity bias [m/s]')):
        width=.78/len(variants)
        for i,name in enumerate(variants):
            values=[next(r[key] for r in records if r['split']==s and r['variant']==name) for s in ('seen_test','unseen_test')]
            bars=ax.bar(np.arange(2)+(i-(len(variants)-1)/2)*width,values,width*.96,label=LABELS[name],color=COLORS.get(name))
            ax.bar_label(bars,fmt='%.3f',rotation=65,fontsize=6,padding=2)
        ax.set_xticks(range(2),('seen_test','unseen_test'));ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.3)
    axes[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.run.resolve();setup_runtime()
    summary=read_json(out/'summary.json');assert summary['status']=='training_completed'
    protocol_record=read_json(out/'protocol.json');protocol=protocol_record['protocol']
    expected=dict(PROTOCOL,smoke_only=args.smoke)
    if args.smoke:expected.update(epochs=2,smoke_anchor_limit=2048)
    assert protocol==expected and summary['smoke_only']==args.smoke
    assert hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()==summary['protocol_sha256']
    assert read_json(out/'training_acceptance.json')['status']=='passed'
    if (out/'evaluation_summary.json').exists() or (out/'comparison').exists():raise FileExistsError('Evaluation outputs already exist')
    for rel,digest in read_json(out/'source_fingerprints.json').items():assert sha(ROOT/rel)==digest,rel
    models={};frozen_states={};checkpoint_hashes={}
    for condition in CONDITIONS:
        saved=torch.load(out/condition/'checkpoint_best.pt',map_location='cpu',weights_only=True)
        validate_checkpoint(saved,'cnn_gru',200,10)
        log=read_csv(out/condition/'training_log.csv')
        assert saved['epoch']==int(min(log,key=lambda r:float(r['val_mse']))['epoch'])
        assert len(log)==protocol['epochs'] and saved['protocol_sha256']==summary['protocol_sha256']
        model=CNNGRU().cuda().eval();model.load_state_dict(saved['model_state_dict'],strict=True)
        for p in model.parameters():p.requires_grad_(False)
        models[condition]=model;frozen_states[condition]=state_sha(model);checkpoint_hashes[condition]=sha(out/condition/'checkpoint_best.pt')
    # Recompute the full validation metrics on the primary checkpoints before tests.
    split_root=out/'smoke_splits' if args.smoke else ROOT/'RoNIN_Team_Delivery/splits'
    ds=RoNINNPZDataset(ROOT/'RoNIN_Team_Delivery/dataset',split_root/'val.txt',mode='val')
    metadata=validation_sequences(ds);pool=FixedWindows(ds,torch.device('cuda'))
    for name,model in models.items():
        got,_=validate(model,pool,metadata)
        for key in ('MSE',*KEYS):np.testing.assert_allclose(got[key],summary['training'][name]['best_validation_metrics'][key],rtol=1e-12,atol=1e-12)
    del ds,pool,metadata;torch.cuda.empty_cache()
    (out/'comparison').mkdir();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[
        logging.StreamHandler(sys.stdout),logging.FileHandler(out/'evaluation.log',encoding='utf-8')])
    provenance=dict(evaluation_script_sha256=sha(Path(__file__)),checkpoint_sha256=checkpoint_hashes,
                    primary_selection='validation MSE only; no best-ATE diagnostic weights tested',
                    reference_prediction_sha256={})
    shutil.copy2(__file__,out/'source_snapshot'/Path(__file__).name)
    references_before={str(p.relative_to(ROOT)):sha(p) for folder in REFERENCES.values() for p in folder.rglob('*') if p.is_file()}
    dump(out/'evaluation_reference_hashes.json',references_before)
    fingerprints=read_csv(out/'data_fingerprints.csv')
    for r in fingerprints:
        p=ROOT/'RoNIN_Team_Delivery/dataset'/r['path'];assert sha(p)==r['sha256']
    rows=[];summaries=[];consistency=[];timings={};verify=[]
    for split in ('seen_test','unseen_test'):
        files=[r['path'] for r in fingerprints if r['split']==split]
        if args.smoke:files=files[:1]
        (out/'comparison'/split).mkdir()
        for name in NEW_VARIANTS:(out/name/split).mkdir(parents=True)
        totals={name:[0.,0.,0] for name in (*NEW_VARIANTS,*REFERENCES)}
        disagreement={name:[0.,0] for name in models}
        for no,filename in enumerate(files,1):
            f,p,t,_,y=load_sequence(ROOT/'RoNIN_Team_Delivery/dataset'/filename)
            starts=np.arange(0,len(y),10,dtype=np.int64);target=y[starts].astype(np.float64);gt=p[:,:2];stem=Path(filename).stem
            predictions={};trajectories={}
            for condition,model in models.items():
                prefix='augmentation' if condition=='yaw_augmentation' else 'consistency'
                if split=='seen_test' and no==1:timings[condition]=benchmark(model,f,starts)
                branches=infer(model,f,starts,512);aligned=branches.astype(np.float64)
                diff=aligned[:,1:]-aligned[:,:1];disagreement[condition][0]+=float(np.sum(diff**2));disagreement[condition][1]+=diff.size
                np.savez_compressed(out/f'{prefix}_c4'/split/f'{stem}_aligned_branches.npz',starts=starts,aligned_velocity=branches,angles_degrees=np.array([0,90,180,270]))
                for mode,pred in (('single',aligned[:,0]),('c4',aligned.mean(axis=1))):
                    name=f'{prefix}_{mode}';positions=reconstruct(t,p,starts,pred);predictions[name]=pred;trajectories[name]=positions
                    row=dict(split=split,sequence=stem,variant=name,**metrics(pred,target,positions,gt,t,starts));rows.append(row)
                    error=pred-target;dt=t[starts+10]-t[starts]
                    totals[name][0]+=float(np.sum(error**2));totals[name][1]+=float(np.sum(abs(error)));totals[name][2]+=len(error)
                    np.savez_compressed(out/name/split/f'{stem}_predictions.npz',starts=starts,timestamps=t,prediction_timestamps=t[starts],window_end_timestamps=t[starts+200],
                        predicted_velocity=pred,target_velocity=target,predicted_positions=positions,ground_truth_positions=gt,
                        prediction_spacing=dt,cumulative_velocity_residual_m=np.cumsum(error*dt[:,None],axis=0))
                assert state_sha(model)==frozen_states[condition]
            for name,folder in REFERENCES.items():
                ref_path=folder/split/f'{stem}_predictions.npz';provenance['reference_prediction_sha256'][str(ref_path.relative_to(ROOT))]=sha(ref_path)
                with np.load(ref_path,allow_pickle=False) as ref:
                    for key,value in dict(starts=starts,timestamps=t,target_velocity=target,ground_truth_positions=gt).items():np.testing.assert_array_equal(ref[key],value)
                    pred=ref['predicted_velocity'];pos=ref['predicted_positions']
                    np.testing.assert_array_equal(reconstruct(t,p,starts,pred),pos)
                    rows.append(dict(split=split,sequence=stem,variant=name,**metrics(pred,target,pos,gt,t,starts)))
                    if name in MAIN:predictions[name]=pred.copy();trajectories[name]=pos.copy()
                    error=pred-target;totals[name][0]+=float(np.sum(error**2));totals[name][1]+=float(np.sum(abs(error)));totals[name][2]+=len(error)
            plot_sequence(out/'comparison'/split/f'{stem}.png',split,stem,t,starts,target,gt,predictions,trajectories)
            dump(out/'progress.json',dict(stage='test_evaluation',split=split,sequences_done=no,total=len(files)))
            logging.info('SEQUENCE_DONE %s %d/%d %s augmentationATE=%.4f consistencyATE=%.4f',split,no,len(files),stem,
                next(r['ATE_official_m'] for r in rows[-9:] if r['variant']=='augmentation_single'),
                next(r['ATE_official_m'] for r in rows[-9:] if r['variant']=='consistency_single'))
        for name,(sq,absolute,count) in totals.items():
            selected=[r for r in rows if r['split']==split and r['variant']==name]
            summaries.append(summarize(split,name,selected,sq,absolute,count))
        for condition,(sq,count) in disagreement.items():
            consistency.append(dict(split=split,condition=condition,rotation_disagreement_RMSE_mps=math.sqrt(sq/count)))
    for split in ('seen_test','unseen_test'):
        for variant in NEW_VARIANTS:
            selected=[r for r in rows if r['split']==split and r['variant']==variant];sq=absolute=0.;count=0;records=[]
            for r in selected:
                path=out/variant/split/f"{r['sequence']}_predictions.npz"
                with np.load(path,allow_pickle=False) as z:
                    assert all(np.isfinite(z[k]).all() for k in z.files)
                    pred=z['predicted_velocity'];target=z['target_velocity'];e=pred-target
                    got=metrics(pred,target,z['predicted_positions'],z['ground_truth_positions'],z['timestamps'],z['starts'])
                    for key in ('MSE',*KEYS):np.testing.assert_allclose(got[key],r[key],rtol=1e-12,atol=1e-12)
                    np.testing.assert_array_equal(reconstruct(z['timestamps'],z['ground_truth_positions'],z['starts'],pred),z['predicted_positions'])
                    np.testing.assert_array_equal(np.cumsum(e*z['prediction_spacing'][:,None],axis=0),z['cumulative_velocity_residual_m'])
                    prefix=variant.rsplit('_',1)[0]
                    with np.load(out/f'{prefix}_c4'/split/f"{r['sequence']}_aligned_branches.npz",allow_pickle=False) as b:
                        aligned=b['aligned_velocity'].astype(np.float64);expected=aligned[:,0] if variant.endswith('single') else aligned.mean(axis=1)
                        np.testing.assert_array_equal(pred,expected)
                    with np.load(out/'augmentation_single'/split/path.name,allow_pickle=False) as other:
                        for key in ('starts','target_velocity','timestamps','ground_truth_positions','prediction_spacing'):np.testing.assert_array_equal(z[key],other[key])
                    sq+=float(np.sum(e**2));absolute+=float(np.sum(abs(e)));count+=len(e);records.append(got)
            actual=summarize(split,variant,records,sq,absolute,count);expected=next(r for r in summaries if r['split']==split and r['variant']==variant)
            for key in ('MSE',*KEYS):np.testing.assert_allclose(actual[key],expected[key],rtol=1e-12,atol=1e-12)
            verify.append(dict(split=split,variant=variant,sequences=len(selected),windows=count,metrics_recomputed=True,trajectories_reconstructed=True,branch_averages_recomputed=True))
    write_csv(out/'comparison/test_metrics.csv',summaries);write_csv(out/'comparison/sequence_metrics.csv',rows)
    write_csv(out/'comparison/rotation_disagreement.csv',consistency)
    dump(out/'inference_benchmark.json',timings)
    comparisons=[('consistency_single','augmentation_single'),('augmentation_single','old_single'),('consistency_single','old_single'),
                 ('consistency_single','old_c4'),('augmentation_c4','augmentation_single'),('consistency_c4','consistency_single'),('consistency_c4','old_c4')]
    effects=[];paired=[];criteria={}
    for improved,control in comparisons:
        criteria[f'{improved}_vs_{control}']={}
        for split in ('seen_test','unseen_test'):
            a=next(r for r in summaries if r['split']==split and r['variant']==control);b=next(r for r in summaries if r['split']==split and r['variant']==improved)
            effects.append(dict(split=split,improved=improved,control=control,**{k+'_error_reduction_percent':100*(a[k]-b[k])/a[k] for k in KEYS}))
            ra={r['sequence']:r for r in rows if r['split']==split and r['variant']==control};rb={r['sequence']:r for r in rows if r['split']==split and r['variant']==improved}
            record=dict(split=split,improved=improved,control=control,sequences=len(ra),metrics={})
            for key in MACRO:
                av=np.array([ra[n][key] for n in ra]);bv=np.array([rb[n][key] for n in ra])
                record['metrics'][key]=dict(improved=int((bv<av).sum()),worsened=int((bv>av).sum()),median_control=float(np.median(av)),median_improved=float(np.median(bv)),p90_control=float(np.percentile(av,90)),p90_improved=float(np.percentile(bv,90)))
            paired.append(record)
            criteria[f'{improved}_vs_{control}'][split]=dict(ATE_lower=b['ATE_official_m']<a['ATE_official_m'],endpoint_not_worse=b['endpoint_error_m']<=a['endpoint_error_m'],bias_not_worse=b['mean_velocity_bias_norm_mps']<=a['mean_velocity_bias_norm_mps'],RMSE_not_worse=b['RMSE']<=a['RMSE'])
        c=criteria[f'{improved}_vs_{control}'];c['overall_improvement']=c['seen_test']['ATE_lower'] and c['unseen_test']['ATE_lower'] and all(c['unseen_test'][k] for k in ('endpoint_not_worse','bias_not_worse','RMSE_not_worse'))
    write_csv(out/'comparison/improvement_percent.csv',effects);dump(out/'comparison/paired_sequence_analysis.json',paired)
    plot_summary(out/'comparison/main_test_summary.png',summaries,MAIN)
    plot_summary(out/'comparison/single_pass_routes.png',summaries,('old_single','augmentation_single','consistency_single'))
    dump(out/'result_verification.json',verify);dump(out/'evaluation_config.json',provenance)
    dump(out/'evaluation_summary.json',dict(status='smoke_completed' if args.smoke else 'completed',test_metrics=summaries,comparisons=criteria))
    for rel,digest in references_before.items():assert sha(ROOT/rel)==digest,rel
    for rel,digest in read_json(out/'original_artifact_hashes.json').items():assert sha(ROOT/rel)==digest,rel
    for r in fingerprints:assert sha(ROOT/'RoNIN_Team_Delivery/dataset'/r['path'])==r['sha256']
    for name,model in models.items():assert state_sha(model)==frozen_states[name]
    from PIL import Image
    for path in (out/'comparison').rglob('*.png'):
        with Image.open(path) as image:image.verify()
    audit=dict(status='passed',smoke_only=args.smoke,full_validation_checkpoints_recomputed=True,
        same_initialization_orders_rotations_steps=True,primary_MSE_selection_verified=True,
        all_new_persisted_predictions_recomputed=True,reference_labels_indices_times_gt_identical=True,
        all_old_predictions_reconstructed=True,original_artifacts_unchanged=True,all_data_files_unchanged=True,
        no_test_BN_updates=True,all_model_states_unchanged_during_test=True,
        new_prediction_files=len(list(out.glob('*_single/*/*_predictions.npz')))+len(list(out.glob('*_c4/*/*_predictions.npz'))),
        reference_sequences_per_route=2 if args.smoke else 67)
    dump(out/'final_audit.json',audit);dump(out/'progress.json',dict(stage='evaluation_completed'))
    logging.info('EVALUATION_COMPLETED %s',json.dumps(criteria))


if __name__=='__main__':main()
