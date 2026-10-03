"""Run all matched baselines on every requested dataset and seed."""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from baselines.baseline_models import build_baselines
from datasets import load_dataset

def make_spatial_splits(S,X,y,seed=42,test_size=.2,val_size=.1):
 idx=np.arange(len(y)); itr,ite=train_test_split(idx,test_size=test_size,random_state=seed); itr,iva=train_test_split(itr,test_size=val_size/(1-test_size),random_state=seed)
 X=np.asarray(X,dtype=float).copy()
 train_medians=np.nanmedian(X[itr],axis=0)
 train_medians=np.where(np.isfinite(train_medians),train_medians,0.0)
 missing=np.where(np.isnan(X))
 X[missing]=train_medians[missing[1]]
 cs,xs,ys=StandardScaler().fit(S[itr]),StandardScaler().fit(X[itr]),StandardScaler().fit(np.asarray(y)[itr,None])
 return (cs.transform(S),xs.transform(X),ys.transform(np.asarray(y)[:,None]).ravel()),(itr,iva,ite),(cs,xs,ys)

MANUSCRIPT_BASELINES = {"SimpleKriging","RegressionKriging","RFKriging","NNGP-Matern","SparseGP-Nystrom","RandomForest","ExtraTrees","HistGBR","Coordinate-MLP","DeepKriging","4N-adapted","KCN-adapted","PE-GNN-adapted"}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--datasets",nargs="+",default=["synthetic","pm25","lucas_large","california"]); p.add_argument("--seeds",nargs="+",type=int,default=[1042,1052,1062,1072,1082]); p.add_argument("--max-samples",type=int,default=None,help="Optional subsample size; omit for complete datasets"); p.add_argument("--test-size",type=float,default=.2); p.add_argument("--val-size",type=float,default=.1); p.add_argument("--include",nargs="*",default=[],help="Run only these method names"); p.add_argument("--exclude",nargs="*",default=[]); p.add_argument("--data-root",type=Path,default=Path(__file__).parents[1]/"data"); p.add_argument("--output-dir",type=Path,default=Path(__file__).parents[1]/"results"/"benchmarks"); a=p.parse_args(); rows=[]
 for dataset in a.datasets:
  for seed in a.seeds:
   S0,X0,y0=load_dataset(dataset,a.data_root,seed,a.max_samples); (S,X,y),(itr,iva,ite),scalers=make_spatial_splits(S0,X0,y0,seed=seed,test_size=a.test_size,val_size=a.val_size); Z=np.c_[S,X]
   for name,model in build_baselines(seed).items():
    if name not in MANUSCRIPT_BASELINES: continue
    if a.include and name not in a.include: continue
    if name in a.exclude: continue
    start=time.perf_counter()
    try:
     model.fit(Z[itr],y[itr]); pred=model.predict(Z[ite]); status="ok"; error=""
     pred_raw=scalers[2].inverse_transform(np.asarray(pred)[:,None]).ravel(); y_raw=np.asarray(y0)[ite]
     rows.append({"dataset":dataset,"seed":seed,"method":name,"rmse":mean_squared_error(y_raw,pred_raw)**.5,"mae":mean_absolute_error(y_raw,pred_raw),"r2":r2_score(y_raw,pred_raw),"seconds":time.perf_counter()-start,"status":status,"error":error})
    except Exception as exc: rows.append({"dataset":dataset,"seed":seed,"method":name,"status":"failed","error":repr(exc),"seconds":time.perf_counter()-start})
    a.output_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output_dir/"baseline_results.csv",index=False); print(dataset,seed,name,rows[-1]["status"])
 (a.output_dir/"run_config.json").write_text(json.dumps(vars(a),indent=2,default=str),encoding="utf-8")
if __name__=="__main__": main()
