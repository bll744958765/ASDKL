"""EPA sparse-observation study with coordinate-grouped outer splits."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from datasets import load_dataset
from data_loader import preprocess_spatial_split
from baselines.baseline_models import build_baselines

SEEDS=[1042,1052,1062,1072,1082]
FRACTIONS=[10,20,30,40,50,60,70]


def grouped_split(coords, train_fraction, val_fraction, seed):
    """Fixed 20% test and 10% validation coordinate groups; nested training subsets."""
    rounded=np.round(np.asarray(coords,float),8)
    _,gid=np.unique(rounded,axis=0,return_inverse=True)
    groups=np.random.default_rng(seed).permutation(np.unique(gid))
    def take(pool,target):
        chosen=[]; count=0
        for g in pool:
            chosen.append(int(g)); count += int(np.sum(gid==g))
            if count>=target: break
        return set(chosen)
    test_groups=take(groups,int(round(0.20*len(coords))))
    remaining=[g for g in groups if g not in test_groups]
    val_groups=take(remaining,int(round(val_fraction*len(coords))))
    train_pool=[g for g in remaining if g not in val_groups]
    train_groups=take(train_pool,int(round(train_fraction*len(coords))))
    tr=np.flatnonzero(np.isin(gid,list(train_groups)))
    va=np.flatnonzero(np.isin(gid,list(val_groups)))
    te=np.flatnonzero(np.isin(gid,list(test_groups)))
    assert not (set(gid[tr])&set(gid[va]) or set(gid[tr])&set(gid[te]) or set(gid[va])&set(gid[te]))
    return tr,va,te


def run_asdkl(a,S):
    rows=[]; code=Path(__file__).resolve().parents[1] / "methods" / "ASDKL"
    for pct in a.fractions:
        split_dir=a.output_dir/f"fraction_{pct}"/"splits"; split_dir.mkdir(parents=True,exist_ok=True)
        for seed in a.seeds:
            tr,va,te=grouped_split(S,pct/100,0.10,seed)
            split_file=split_dir/f"seed_{seed}.npz"
            np.savez_compressed(split_file,train_idx=tr,val_idx=va,test_idx=te)
            out=a.output_dir/f"fraction_{pct}"/"ASDKL"/f"seed_{seed}"; result=out/"pm25_run.json"
            if not result.exists():
                cmd=[sys.executable,str(code/"run.py"),"--dataset","pm25","--seed",str(seed),"--device",a.device,"--data-root",str(a.data_root),"--output-dir",str(out),"--split-file",str(split_file),"--epochs",str(a.epochs),"--neighbors","64","--freq-dim","64","--hidden-dim","64","--fusion","gated","--trend-width","128","--trend-depth","6","--trend-residual","1","--lambda-mse","0","--lambda-spectral","0.1","--lr","0.003","--batch-size","64","--restore-best","1","--representation-kernel","1","--base-kernel","rbf","--residual-ensemble","0","--local-expert","0","--covariate-expert","0","--gp-inference","exact","--resume","1"]
                subprocess.run(cmd,check=True)
            d=json.loads(result.read_text(encoding="utf-8")); d["training_fraction_pct"]=pct; rows.append(d)
            pd.DataFrame(rows).to_csv(a.output_dir/"asdkl_fraction_results.csv",index=False)


def run_baselines(a,S0,X0,y0,pct):
    rows=[]; split_dir=a.output_dir/f"fraction_{pct}"/"splits"
    out=a.output_dir/f"fraction_{pct}"/"baselines"; out.mkdir(parents=True,exist_ok=True)
    for seed in a.seeds:
        split=np.load(split_dir/f"seed_{seed}.npz")
        (S,X,y),(itr,iva,ite),scalers=preprocess_spatial_split(S0,X0,y0,split["train_idx"],split["val_idx"],split["test_idx"])
        Z=np.c_[S,X]
        for name,model in build_baselines(seed).items():
            if name not in {"SimpleKriging","RegressionKriging","RFKriging","RandomForest","ExtraTrees","HistGBR","Coordinate-MLP","DeepKriging","4N-adapted","KCN-adapted","PE-GNN-adapted","NNGP-Matern","SparseGP-Nystrom"}: continue
            start=time.perf_counter(); model.fit(Z[itr],y[itr]); pred=model.predict(Z[ite])
            pred_raw=scalers[2].inverse_transform(np.asarray(pred)[:,None]).ravel(); target=np.asarray(y0)[ite]
            rows.append(dict(dataset="pm25",training_fraction_pct=pct,seed=seed,method=("DeepKriging-adapted" if name=="DeepKriging" else name),rmse=mean_squared_error(target,pred_raw)**.5,mae=mean_absolute_error(target,pred_raw),r2=r2_score(target,pred_raw),seconds=time.perf_counter()-start))
            pd.DataFrame(rows).to_csv(out/"baseline_results.csv",index=False)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda"); p.add_argument("--epochs",type=int,default=300); p.add_argument("--output-dir",type=Path,default=Path(__file__).parents[1]/"results"/"epa_sparse_fraction"); p.add_argument("--fractions",nargs="+",type=int,default=FRACTIONS); p.add_argument("--seeds",nargs="+",type=int,default=SEEDS); p.add_argument("--baseline-fraction",type=int); p.add_argument("--data-root",type=Path,default=Path(__file__).resolve().parents[1]/"data"); a=p.parse_args()
    S,X,y=load_dataset("pm25",a.data_root,a.seeds[0],None)
    run_asdkl(a,S)
    if a.baseline_fraction is not None: run_baselines(a,S,X,y,a.baseline_fraction)

if __name__=="__main__": main()
