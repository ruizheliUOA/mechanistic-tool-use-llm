"""F1 — SAKIKO-Core. Panel A: the ladder inside its evidential wrapper.
   Panel B: what Correctable decomposes into."""
import sys; sys.path.insert(0,'.')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from sakiko_style import C, apply, save
apply()
fig = plt.figure(figsize=(7.2,4.15))
gs  = fig.add_gridspec(2,1,height_ratios=[1.30,1.0],hspace=0.10)
# ---------------- Panel A ----------------
ax = fig.add_subplot(gs[0]); ax.set_xlim(0,100); ax.set_ylim(12.2,45.5); ax.axis("off")
def box(x,y,w,h,label,fc,ec,tc="white",fs=8.8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=1.3",fc=fc,ec=ec,lw=0.9))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fs,color=tc,fontweight="bold")
def arrow(a,b,y,col=None,lw=0.9,ax=ax):
    ax.add_patch(FancyArrowPatch((a,y),(b,y),arrowstyle="-|>",mutation_scale=8,lw=lw,
                                 color=col or C["ink"],shrinkA=0,shrinkB=0))
ax.add_patch(Rectangle((2.0,16.6),96,22.4,fc="#FAFBFC",ec=C["neutral"],lw=0.8,ls=(0,(5,3))))
ax.text(3.4,36.6,"evidential layer  ·  properties of the study",fontsize=7.2,color=C["neutral"],style="italic")
ax.add_patch(Rectangle((21.0,19.5),56.0,14.5,fc="#F3F5F7",ec=C["structure"],lw=0.8))
ax.text(22.3,31.6,"scientific layer  ·  properties of the setting",fontsize=7.2,color=C["structure"],style="italic")
NODES=[(4.0,15.5,"Adjudicable",C["structure"],"white"),(22.5,15.5,"Readable","#FFFFFF",C["structure"]),
       (40.0,15.5,"Steerable","#FFFFFF",C["structure"]),(57.5,18.0,"Correctable","#FFFFFF",C["structure"]),
       (79.5,16.5,"Licensable",C["pass_"],"white")]
for x,w,lab,fc,tc in NODES: box(x,21.5,w,7.0,lab,fc,C["structure"] if fc=="#FFFFFF" else fc,tc)
for a,b in ((19.5,22.5),(37.5,40.0),(55.0,57.5),(75.5,79.5)): arrow(a,b,25.0)
EX=[(11.75,"Qwen3.5  ca→*\nsupport 29 / 30",C["fail"]),(30.25,"Qwen3.5  rfi→tc\nAUC 0.727",C["fail"]),
    (47.75,"Mistral  rfi→tc\n17 / 20 randoms",C["fail"]),(66.25,"Qwen3-4B · Gemma\nCI lower fails",C["fail"]),
    (87.75,"Qwen3-8B\nADMIT",C["pass_"])]
for x,lab,col in EX:
    ax.add_patch(FancyArrowPatch((x,21.3),(x,15.6),arrowstyle="-|>",mutation_scale=7,lw=0.8,color=col,shrinkA=0,shrinkB=0))
    ax.text(x,14.4,lab,ha="center",va="top",fontsize=6.9,color=col,linespacing=1.3)
ax.text(78.5,18.2,"preservation  ·  yield  ·  evidence",fontsize=7.1,color=C["neutral"],ha="right",va="center",style="italic")
ax.add_patch(FancyArrowPatch((79.5,18.4),(85.5,21.0),arrowstyle="-|>",mutation_scale=7,lw=0.8,color=C["neutral"],shrinkA=0,shrinkB=1))
ax.text(0,43.5,"A",fontsize=10,fontweight="bold",color=C["ink"])
ax.text(50,43.1,"Adjudicable  →  [ Readable → Steerable → Correctable ]  →  Licensable",
        ha="center",fontsize=9.2,fontweight="bold",color=C["ink"])
# ---------------- Panel B ----------------
bx = fig.add_subplot(gs[1]); bx.set_xlim(0,100); bx.set_ylim(0,30); bx.axis("off")
bx.text(0,27.0,"B",fontsize=10,fontweight="bold",color=C["ink"])
bx.text(6.0,27.0,"Correctable decomposes the fate of every channel error",fontsize=8.4,fontweight="bold",color=C["ink"],va="center")
bx.add_patch(FancyBboxPatch((4.0,10.0),19,7.0,boxstyle="round,pad=0,rounding_size=1.3",fc="#FFFFFF",ec=C["structure"],lw=0.9))
bx.text(13.5,13.5,"channel error\n$c=(g\\rightarrow s)$",ha="center",va="center",fontsize=8.0,color=C["structure"],linespacing=1.3)
DEST=[("SOURCE_RETAINED","still $s$ — no movement",C["neutral"],20.0),
      ("GOLD_ARRIVAL","reached $g$ — repair",C["pass_"],11.5),
      ("OTHER_WRONG","some $w\\in A_D\\setminus\\{g,s\\}$",C["fail"],3.0)]
for lab,sub,col,y in DEST:
    bx.add_patch(FancyArrowPatch((23.5,13.5),(43.0,y+3.0),arrowstyle="-|>",mutation_scale=8,lw=1.0,
                                 color=col,shrinkA=1,shrinkB=1,connectionstyle="arc3,rad=0.10"))
    bx.add_patch(FancyBboxPatch((44.0,y),26,6.0,boxstyle="round,pad=0,rounding_size=1.2",fc="#FFFFFF",ec=col,lw=1.0))
    bx.text(57.0,y+3.0,lab,ha="center",va="center",fontsize=8.0,color=col,fontweight="bold")
    bx.text(72.0,y+3.0,sub,ha="left",va="center",fontsize=7.3,color=C["ink"])
bx.text(50,-2.2,"Aggregate metrics score SOURCE_RETAINED against the other two; only destination resolution separates GOLD_ARRIVAL from OTHER_WRONG.",
        ha="center",fontsize=7.2,color=C["neutral"],style="italic")
print(save(fig,"fig1_sakiko_core","../main"))
