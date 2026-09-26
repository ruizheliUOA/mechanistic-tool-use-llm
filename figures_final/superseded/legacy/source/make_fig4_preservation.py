"""F4 — E1 vs E2 preservation estimands and finite-sample bounds."""
import sys; sys.path.insert(0,'.')
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import beta
from sakiko_style import C, apply, save
apply()
def up(k,n): return 1.0 if k==n else float(beta.ppf(0.95,k+1,n-k))
S=[("Qwen3-8B",0,211,0,6,"ADMIT"),("Qwen3-4B",1,214,1,50,"DECLINE"),
   ("Gemma-2-9B",1,223,1,11,"DECLINE"),("Phi-3.5",52,264,52,93,"retrospective")]
fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),gridspec_kw={"width_ratios":[1.15,1],"wspace":0.30})
y=np.arange(len(S))[::-1]
# A — exposure dilution
ax=axes[0]
for yy,(nm,k1,n1,k2,n2,v) in zip(y,S):
    ax.barh(yy,n1,height=0.46,color=C["faint"],edgecolor=C["neutral"],linewidth=0.6)
    ax.barh(yy,n2,height=0.46,color=C["structure"],edgecolor="none")
    ax.text(n1+7,yy,f"{n2} exposed of {n1}",va="center",fontsize=7.4,color=C["ink"])
ax.set_yticks(y); ax.set_yticklabels([s[0] for s in S]); ax.set_xlim(0,340)
ax.set_xlabel("baseline-correct rows"); ax.set_title("A  The denominator is diluted",loc="left",fontweight="bold")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
h=[Rectangle((0,0),1,1,fc=C["faint"],ec=C["neutral"],lw=.6),Rectangle((0,0),1,1,fc=C["structure"],ec="none")]
ax.legend(h,["E1  all baseline-correct (frozen)","E2  exposed to the intervention"],
          loc="upper center",bbox_to_anchor=(0.5,-0.30),ncol=1,handlelength=1.1,handleheight=0.8)
# B — bounds
ax=axes[1]
ax.axvline(0.05,color=C["fail"],lw=1.0,ls=(0,(4,2)),zorder=1)
ax.text(0.058,y[0]+0.52,"α = 0.05",fontsize=7.4,color=C["fail"])
for yy,(nm,k1,n1,k2,n2,v) in zip(y,S):
    e1,e2,u=k1/n1,k2/n2,up(k2,n2)
    ax.plot([e2,u],[yy,yy],color=C["neutral"],lw=1.1,zorder=2,solid_capstyle="butt")
    ax.plot(u,yy,marker="|",ms=7,mew=1.3,color=C["neutral"],zorder=3)
    ax.scatter([e1],[yy],s=26,facecolor="white",edgecolor=C["structure"],lw=1.1,zorder=4)
    ax.scatter([e2],[yy],s=26,color=C["structure"],zorder=4)
    ax.text(u+0.018,yy,f"{u:.3f}",va="center",fontsize=7.4,color=C["ink"])
ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(-0.02,0.78); ax.set_xlabel("preservation risk")
ax.set_title("B  No setting certifies α = 0.05",loc="left",fontweight="bold")
for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
ax.tick_params(left=False)
h=[plt.Line2D([],[],marker='o',ls='',mfc='white',mec=C["structure"],ms=5.2),
   plt.Line2D([],[],marker='o',ls='',color=C["structure"],ms=5.2),
   plt.Line2D([],[],color=C["neutral"],lw=1.1)]
ax.legend(h,["E1 point estimate","E2 point estimate","95% one-sided upper bound on E2"],
          loc="upper center",bbox_to_anchor=(0.45,-0.30),ncol=1,handlelength=1.3)
print(save(fig,"fig4_preservation_exposure_audit","../main"))
