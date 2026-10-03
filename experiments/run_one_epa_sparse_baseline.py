from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from datasets import load_dataset
from data_loader import preprocess_spatial_split
from baselines.baseline_models import build_baselines
p=argparse.ArgumentParser();p.add_argument("--method",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--training-fraction-pct",type=int,required=True);p.add_argument("--split-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--data-root",type=Path,default=Path(__file__).resolve().parents[1]/"data");a=p.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=True); out=a.output_dir/f"{a.method.replace('/','_')}_seed_{a.seed}.json"
if out.exists(): raise SystemExit(0)
S0,X0,y0=load_dataset("pm25",a.data_root,a.seed,None); split=np.load(a.split_dir/f"seed_{a.seed}.npz")
(S,X,y),(itr,iva,ite),scalers=preprocess_spatial_split(S0,X0,y0,split["train_idx"],split["val_idx"],split["test_idx"]); Z=np.c_[S,X]
lookup={"DeepKriging-adapted":"DeepKriging"}; key=lookup.get(a.method,a.method); model=build_baselines(a.seed)[key]
if a.method=="SparseGP-Nystrom": model.center_method="lloyd"
print("FIT_START",flush=True); start=time.perf_counter(); model.fit(Z[itr],y[itr]); print("FIT_DONE",flush=True); pred=model.predict(Z[ite]); print("PRED_DONE",flush=True); pred_raw=scalers[2].inverse_transform(np.asarray(pred)[:,None]).ravel(); target=np.asarray(y0)[ite]
row=dict(dataset="pm25",training_fraction_pct=a.training_fraction_pct,seed=a.seed,method=a.method,rmse=mean_squared_error(target,pred_raw)**.5,mae=mean_absolute_error(target,pred_raw),r2=r2_score(target,pred_raw),seconds=time.perf_counter()-start)
out.write_text(json.dumps(row,indent=2),encoding="utf-8");print(json.dumps(row),flush=True)
