from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/"results"/"epa_sparse_fraction_fixedtest"
raw=pd.read_csv(root/"asdkl_fraction_results.csv")
baseline=pd.read_csv(root/"fraction_40"/"baselines"/"baseline_results.csv").drop_duplicates(["method","seed"],keep="last")
asdkl=raw[raw.training_fraction_pct==40][["dataset","training_fraction_pct","seed","rmse","mae","r2"]].copy(); asdkl["method"]="ASDKL"
comparison=pd.concat([baseline,asdkl],ignore_index=True,sort=False)
comparison.to_csv(root/"fraction_40"/"baseline_comparison_complete.csv",index=False)
summary=comparison.groupby("method")[["rmse","mae","r2"]].agg(["mean","std"])
summary.to_csv(root/"fraction_40"/"baseline_comparison_summary.csv")
learning=raw.groupby("training_fraction_pct")[["rmse","mae","r2","nll","crps","picp90","mpiw90"]].agg(["mean","std"])
learning.to_csv(root/"learning_curve_summary.csv")
r10=learning.loc[10,("rmse","mean")]; r70=learning.loc[70,("rmse","mean")]
learning[("rmse","gain_fraction")]=(r10-learning[("rmse","mean")])/(r10-r70)
learning.to_csv(root/"learning_curve_summary.csv")
# Figure contract: show the fixed-test learning curve and G(p) without selecting or visually emphasizing a plateau fraction.
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"pdf.fonttype":42,"svg.fonttype":"none","font.size":7,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,"legend.frameon":False})
fig,axes=plt.subplots(1,2,figsize=(7.2,2.75),gridspec_kw={"width_ratios":[1.2,1]})
colors=["#4C78A8","#F58518","#54A24B","#B279A2","#E45756"]
for color,(seed,g) in zip(colors,raw.groupby("seed")):
    g=g.sort_values("training_fraction_pct"); axes[0].plot(g.training_fraction_pct,g.rmse,"o-",lw=.9,ms=2.8,color=color,alpha=.75,label=str(seed))
x=learning.index.to_numpy(float); mean=learning[("rmse","mean")].to_numpy(); sd=learning[("rmse","std")].to_numpy(); ci=2.776*sd/np.sqrt(5)
axes[0].plot(x,mean,"o-",color="black",lw=1.8,ms=3.8,label="Mean")
axes[0].fill_between(x,mean-ci,mean+ci,color="black",alpha=.12,lw=0)
axes[0].set(xlabel="Training observations (%)",ylabel="Test RMSE",xticks=x)
axes[0].legend(ncol=2,fontsize=6,loc="upper right",title="Seed",title_fontsize=6)
gain=(r10-mean)/(r10-r70)*100
axes[1].plot(x,gain,"o-",color="#4C78A8",lw=1.8,ms=3.8)
axes[1].set(xlabel="Training observations (%)",ylabel="Recovered RMSE improvement (%)",xticks=x,ylim=(-5,108))
for label,ax in zip("ab",axes): ax.text(-.14,1.04,label,transform=ax.transAxes,fontweight="bold",fontsize=8)
fig.tight_layout(w_pad=2)
out=Path(__file__).resolve().parents[1]/"results"/"generated_figures"/"epa_sparse_learning_curve"
out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight'); fig.savefig(out.with_suffix('.svg'),bbox_inches='tight'); fig.savefig(out.with_suffix('.tiff'),dpi=600,bbox_inches='tight'); fig.savefig(out.with_suffix('.png'),dpi=200,bbox_inches='tight')
print('baseline rows',len(comparison)); print(summary.to_string()); print(learning.to_string())
