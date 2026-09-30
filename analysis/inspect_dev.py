"""Inspection board: candidate identities, labels, and their released reference means."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from src.common import ROOT,load_jsonl

def main():
    gs=load_jsonl(ROOT/'data/derived/dev.jsonl')
    selected=[gs[i] for i in range(0,len(gs),3)]
    dest=ROOT/'results/dev/inspection';dest.mkdir(parents=True,exist_ok=True)
    for num,g in enumerate(selected):
        fig,axes=plt.subplots(1,4,figsize=(12,3.8))
        for ax in axes:ax.axis('off')
        for label,c,ax in zip('ABCD',g['candidates'],axes):
            im=Image.open(ROOT/c['image']);im.thumbnail((448,448));ax.imshow(im)
            ax.set_title(f'{label}: {c["id"][-12:]}\nh={c["utility"]:.3f}; n={c["n_utility"]}',fontsize=9)
        fig.suptitle(g['country']+' | '+g['prompt'],fontsize=10)
        fig.tight_layout();fig.savefig(dest/f'{num:02d}.png',dpi=130);plt.close(fig)

if __name__=='__main__':main()
