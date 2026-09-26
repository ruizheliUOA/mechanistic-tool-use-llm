"""F3 — aggregate movement vs destination-resolved outcome."""
import json, sys
sys.path.insert(0,'.')
import numpy as np, matplotlib.pyplot as plt
from sakiko_style import C, apply, save
apply()
R="/root/autodl-tmp/sakiko-followup"
q8=json.load(open(f"{R}/final/results/qwen3_stage2_formal/QWEN3_STAGE2_FORMAL_RESULTS.json"))["primary"]["real"]
S=[("Qwen3-8B\nADMIT", q8["n_channel_error"], q8["source_exits"], q8["gold_arrivals"], q8["wrong_to_wrong"], q8["fixed"]-q8["broke"], True),
   ("Qwen3-4B\nDECLINE", 124, 64, 37, 27, 39, False),
   ("Gemma-2-9B\nDECLINE",  96, 27, 17, 10, 16, False)]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios":[1,1.5], "wspace":0.34})
# A — the conventional view
ax=axes[0]; y=np.arange(len(S))[::-1]
ax.barh(y, [s[5] for s in S], height=0.5, color=C["structure"], edgecolor="none")
for yy,s in zip(y,S): ax.text(s[5]+1.4, yy, f"+{s[5]}", va="center", fontsize=8, color=C["ink"])
ax.set_yticks(y); ax.set_yticklabels([s[0] for s in S]); ax.set_xlabel("Net (Fixed − Broke)")
ax.set_xlim(0,50); ax.set_title("A  Conventional aggregate view", loc="left", fontweight="bold")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.text(0.5,-0.42,"All three appear positive.",transform=ax.transAxes,ha="center",fontsize=7.6,style="italic",color=C["neutral"])
# B — destination-resolved
ax=axes[1]
for yy,s in zip(y,S):
    n,ex,g,o = s[1],s[2],s[3],s[4]; ret=n-ex
    L=0
    for w,col in ((ret,C["neutral"]),(g,C["pass_"]),(o,C["fail"])):
        ax.barh(yy, w/n, left=L/n, height=0.5, color=col, edgecolor="white", linewidth=0.6); L+=w
    ax.text(1.015, yy, f"{g}/{ex} to gold", va="center", fontsize=7.4, color=C["ink"])
ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0,1.0); ax.set_xlabel("Fraction of channel errors")
ax.set_title("B  Destination-resolved view", loc="left", fontweight="bold")
for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
ax.tick_params(left=False)
h=[plt.Rectangle((0,0),1,1,fc=c,ec="none") for c in (C["neutral"],C["pass_"],C["fail"])]
ax.legend(h,["SOURCE_RETAINED","GOLD_ARRIVAL","OTHER_WRONG"],loc="upper center",
          bbox_to_anchor=(0.42,-0.30),ncol=3,handlelength=1.1,handleheight=0.8,columnspacing=1.3)
print(save(fig,"fig3_destination_accounting","../main"))
