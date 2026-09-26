"""F2 — failure-localization matrix."""
import sys; sys.path.insert(0,'.')
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sakiko_style import C, apply, save
apply()
COLS=["Adjudic-\nability","Read-\nability","Speci-\nficity","Desti-\nnation","Preser-\nvation","Licens-\nability"]
P,F,I,N,H = "PASS","FAIL","INCONC","NOTREACH","HIST"
ROWS=[
 ("Qwen3-8B  ca→tc",            [P,P,P,P,P,P],           "formal ADMIT"),
 ("Qwen3-4B  ca→tc",            [P,P,P,F,P,F],           "destination certification"),
 ("Gemma-2-9B  ca→tc",          [P,P,P,F,P,F],           "destination certification"),
 ("Gemma-2-9B  ca→direct",      [P,P,P,P,I,F],           "collateral vacuous (0 exposed)"),
 ("Qwen3.5-9B  ca→tc",          [F,N,N,N,N,N],           "support 29/30 — split-sensitive"),
 ("Qwen3.5-9B  ca→direct",      [F,N,N,N,N,N],           "support — split-sensitive"),
 ("Qwen3.5-9B  rfi→tc",         [P,F,N,N,N,N],           "readability 0.727"),
 ("Mistral-7B  rfi→tc",         [H,H,F,N,N,N],           "17/20 randoms ≥ real"),
 ("Llama-3.1-8B  ca→tc",        [H,H,F,N,N,N],           "5/20 randoms ≥ real"),
 ("Phi-3.5  cascade",           [H,H,H,H,F,F],           "collateral 52/93 exposed"),
 ("ACEBench  leading channels", [F,N,N,N,N,N],           "framework instantiation only"),
]
FC={P:C["pass_"],F:C["fail"],I:C["caution"],N:"#F2F4F6",H:C["neutral"]}
TC={P:"white",F:"white",I:"white",N:C["neutral"],H:"white"}
TX={P:"✓",F:"✗",I:"~",N:"·",H:"H"}
fig,ax=plt.subplots(figsize=(7.2,3.65))
nr,nc=len(ROWS),len(COLS)
for i,(nm,st,note) in enumerate(ROWS):
    yy=nr-1-i
    for j,s in enumerate(st):
        ax.add_patch(Rectangle((j*1.16,yy),1.06,0.86,fc=FC[s],ec="white",lw=1.1))
        ax.text(j*1.16+0.53,yy+0.43,TX[s],ha="center",va="center",fontsize=8.2,color=TC[s],fontweight="bold")
        if i==0 and j==4: ax.text(j*1.16+0.90,yy+0.66,"†",ha="center",va="center",fontsize=8.0,color="white",fontweight="bold")
    ax.text(-0.22,yy+0.43,nm,ha="right",va="center",fontsize=8.0)
    ax.text(nc*1.16+0.16,yy+0.43,note,ha="left",va="center",fontsize=7.2,color=C["neutral"],style="italic")
for j,c in enumerate(COLS): ax.text(j*1.16+0.53,nr+0.10,c,ha="center",va="bottom",fontsize=7.6,linespacing=1.15)
ax.set_xlim(-3.4,nc*1.16+3.6); ax.set_ylim(-0.75,nr+0.85); ax.axis("off")
h=[Rectangle((0,0),1,1,fc=FC[k],ec="white") for k in (P,F,I,N,H)]
ax.legend(h,["pass","fail — the binding rung","inconclusive / vacuous","not reached","historical protocol generation"],
          loc="upper center",bbox_to_anchor=(0.42,-0.005),ncol=5,handlelength=1.0,handleheight=0.9,columnspacing=1.1)
ax.text(-3.2,-0.62,"†  Passed the frozen E1 endpoint (0/211). Under the retrospective exposure-conditional estimand E2 this setting is 0/6, upper bound 0.393 — not preservation-certified.",
        fontsize=6.8,color=C["neutral"],style="italic",ha="left",va="center")
print(save(fig,"fig2_failure_localization_matrix","../main"))
