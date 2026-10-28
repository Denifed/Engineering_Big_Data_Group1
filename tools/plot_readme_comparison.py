"""Plot the README's old CNN-GRU / old CNN-GRU+C4 / A+C4 comparison.

Reads saved aggregate test metrics only. Does not train or load checkpoints.
"""
from pathlib import Path
import argparse
import csv
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'reports/cnn_gru_yaw_equivariance_20261027/comparison'
SPLITS=('seen_test','unseen_test')
ROUTES=(('old_single','Old CNN-GRU','#d88425'),
        ('old_c4','Old CNN-GRU + C4','#3279b7'),
        ('augmentation_c4','A + C4','#4c9857'))
METRICS=(('RMSE','Velocity RMSE [m/s]'),
         ('ATE_official_m','Official ATE [m]'),
         ('RTE_60s_official_m','Official 60s RTE [m]'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,default=FOLDER/'test_metrics.csv')
    parser.add_argument('--out',type=Path,default=FOLDER/'gru_c4_augmentation_comparison.png')
    args=parser.parse_args()
    with args.metrics.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    selected={}
    for split in SPLITS:
        for variant,_,_ in ROUTES:
            matches=[r for r in rows if r['split']==split and r['variant']==variant]
            if len(matches)!=1:
                raise ValueError(f'Expected one saved row for {split}/{variant}')
            row=matches[0]
            if int(row['sequences'])!= {'seen_test':35,'unseen_test':32}[split]:
                raise ValueError('Expected full test splits')
            selected[split,variant]=row
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    x=np.arange(len(SPLITS));width=.25
    for ax,(metric,title) in zip(axes,METRICS):
        maximum=0
        for i,(variant,label,color) in enumerate(ROUTES):
            values=[float(selected[split,variant][metric]) for split in SPLITS]
            if not all(math.isfinite(v) and v>=0 for v in values):
                raise ValueError(f'Invalid saved values: {variant}/{metric}')
            maximum=max(maximum,*values)
            bars=ax.bar(x+(i-1)*width,values,width,label=label,color=color)
            ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
        ax.set_xticks(x,SPLITS);ax.set_title(title,fontsize=12)
        ax.set_ylim(0,maximum*1.30)
        ax.tick_params(labelsize=10)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.25)
        ax.legend(loc='upper left',fontsize=9,framealpha=.95)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=180,facecolor='white')
    plt.close(fig)
    print(f'Saved {args.out}')
    for split in SPLITS:
        for variant,_,_ in ROUTES:
            r=selected[split,variant]
            print(split,variant,*(f'{metric}={r[metric]}' for metric,_ in METRICS))

if __name__=='__main__':main()
