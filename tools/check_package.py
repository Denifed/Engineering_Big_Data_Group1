"""Read-only checks. No training, downloads, file writes, or CUDA inference."""
from pathlib import Path
import argparse,ast,csv,hashlib,json,struct,sys,zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def npz_headers(path):
    result={}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('.npy'):continue
            with z.open(name) as f:
                if f.read(6)!=b'\x93NUMPY':raise ValueError(f'Invalid NPY magic: {path}:{name}')
                major,minor=f.read(2)
                size=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0]
                hdr=ast.literal_eval(f.read(size).decode('utf-8' if major>=3 else 'latin1'))
                result[name[:-4]]={'shape':tuple(hdr['shape']),'dtype':hdr['descr']}
    return result

def read_splits():
    result={}
    for split,count in [('train',72),('val',13),('seen_test',35),('unseen_test',32)]:
        path=ROOT/'RoNIN_Team_Delivery/splits'/f'{split}.txt'
        entries=[s.strip().replace('\\','/') for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip() and not s.lstrip().startswith('#')]
        if len(entries)!=count or len(set(entries))!=count:raise ValueError(f'Incorrect split: {split}')
        if any(Path(s).is_absolute() or '..' in Path(s).parts for s in entries):raise ValueError(f'Unsafe split path: {split}')
        result[split]=entries
    flattened=[name for names in result.values() for name in names]
    if len(flattened)!=len(set(flattened)):raise ValueError('Overlapping sequence files between splits')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',action='store_true',help='Check all 152 NPZ headers and a small real-data label sample')
    p.add_argument('--models',action='store_true',help='Check three model shapes on CPU only')
    p.add_argument('--hash',action='store_true',help='Verify every archived copy against copied_files.csv (about 7 GiB)')
    args=p.parse_args()
    required=['README.md','.gitignore','.gitattributes','requirements.txt','cnn_baseline/dataset.py',
              'cnn_baseline/model_cnn1d.py','cnn_baseline/model_cnn_gru.py','cnn_baseline/model_cnn_gru_attention.py',
              'scripts/run_ronin_npz_baseline.py','ronin-master/source/model_resnet1d.py','ronin-master/source/metric.py',
              'ronin-master/LICENSE','reports/cnn_three_routes_fixed_20261026/comparison/test_metrics.csv',
              'manifests/data_manifest.csv','manifests/copied_files.csv']
    missing=[f for f in required if not (ROOT/f).is_file()]
    if missing:raise FileNotFoundError(missing)
    sources=[*(ROOT/'cnn_baseline').glob('*.py'),*(ROOT/'scripts').glob('*.py')]
    for f in sources:ast.parse(f.read_text(encoding='utf-8-sig'),filename=str(f))
    splits=read_splits()
    report={'basic':'passed','python_files_parsed':len(sources),'split_counts':{s:len(n) for s,n in splits.items()}}
    if args.data:
        metadata=list(csv.DictReader((ROOT/'manifests/data_manifest.csv').open(encoding='utf-8')))
        if len(metadata)!=152:raise ValueError('Data manifest must contain 152 entries')
        data=ROOT/'RoNIN_Team_Delivery/dataset'
        for row in metadata:
            path=data/row['path']
            if not path.is_file():raise FileNotFoundError(f'Restore the dataset: {path}')
            if path.stat().st_size!=int(row['bytes']):raise ValueError(f'Data byte count mismatch: {path}')
            headers=npz_headers(path);n=int(row['frames'])
            expected={'features':(n,6),'positions':(n,3),'targets':(n,2),'timestamps':(n,),'orientations':(n,4)}
            if n<=200 or any(headers.get(k,{}).get('shape')!=v for k,v in expected.items()):raise ValueError(f'NPZ shape mismatch: {path}')
        import numpy as np
        sample=min(metadata,key=lambda row:int(row['bytes']))
        with np.load(data/sample['path'],allow_pickle=False) as z:
            t=z['timestamps'];pos=z['positions'];x=z['features'][:200].T
            if not np.all(np.isfinite(t)) or not np.all(np.diff(t)>0):raise ValueError('Sample timestamps are not finite/increasing')
            y=(pos[200,:2]-pos[0,:2])/(t[200]-t[0])
            if x.shape!=(6,200) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):raise ValueError('Real sample label/input check failed')
        report['data']={'headers_checked':152,'real_sample':sample['path'],'sample_input_shape':list(x.shape),'sample_target_shape':list(y.shape),'full_data_sha256_checked':args.hash}
    if args.models:
        sys.path.insert(0,str(ROOT))
        import torch
        from cnn_baseline.model_cnn1d import CNN1D
        from cnn_baseline.model_cnn_gru import CNNGRU
        from cnn_baseline.model_cnn_gru_attention import CNNGRUAttention
        torch.set_num_threads(2);out={}
        with torch.inference_mode():
            for cls in [CNN1D,CNNGRU,CNNGRUAttention]:
                model=cls().cpu().eval();y=model(torch.zeros(2,6,200))
                if tuple(y.shape)!=(2,2) or not torch.isfinite(y).all():raise ValueError(f'Invalid model output: {cls.__name__}')
                out[cls.__name__]={'input':[2,6,200],'output':list(y.shape),'parameters':sum(p.numel() for p in model.parameters())}
        report['CPU_model_interfaces']=out
    if args.hash:
        rows=list(csv.DictReader((ROOT/'manifests/copied_files.csv').open(encoding='utf-8')))
        for row in rows:
            if sha256(ROOT/row['path'])!=row['sha256']:raise ValueError(f'Archive hash mismatch: {row["path"]}')
        report['archive_hashes_checked']=len(rows)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
