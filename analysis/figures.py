"""Four exportable, vector research figures; never invent numerical data."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from src.common import ROOT

COLORS={'Qwen':'#007C91','Smol':'#D77A34','Random':'#747A83','Oracle':'#4D6954'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
                     'savefig.bbox':'tight'})
DEST=ROOT/'paper/figures'

def save(fig,name):
    DEST.mkdir(parents=True,exist_ok=True)
    fig.savefig(DEST/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(DEST/f'{name}.png',dpi=220,bbox_inches='tight')
    plt.close(fig)

def overview():
    fig,ax=plt.subplots(figsize=(7.1,2.85)); ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
    def box(x,y,w,h,text,color='#EAF3F5'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#63828A',linewidth=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.5)
    def arrow(x1,y1,x2,y2,dashed=False):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,
                                    color='#58717A',linestyle='--' if dashed else '-',linewidth=1.1))
    box(.12,2.4,1.45,.85,'Original\nprompt')
    box(2.0,2.15,2.2,1.35,'Candidate images\n\nA    B    C    D')
    box(4.7,2.4,1.7,.85,'VLM judge\nselect / abstain')
    box(7.,2.5,2.7,.7,'Simulated returned image')
    box(7.,1.4,2.7,.55,'Abstain:\nreturn unresolved','#F5F0E9')
    box(2.,.35,2.2,.8,'Existing public\nhuman annotations','#EEF1ED')
    box(5.,.15,4.7,1.0,'Retrospective audit\nRegret = max(h) − h(selected)\nHSR = P[h(selected) < mean(h)]','#EEF1ED')
    arrow(1.65,2.82,1.94,2.82);arrow(4.3,2.82,4.6,2.82);arrow(6.5,2.82,6.93,2.82)
    arrow(5.55,2.3,6.93,1.69);arrow(3.1,2.05,3.1,1.25,True)
    arrow(4.3,.75,4.91,.75,True)
    ax.plot([9.8,9.95,9.95],[2.85,2.85,.72],color='#58717A',ls='--',lw=1.1)
    arrow(9.95,.72,9.82,.72,True)
    ax.text(.12,.75,'Inputs:\nprompt + images',fontsize=7.5,color='#395560')
    save(fig,'overview')

def pending(name,title):
    fig,ax=plt.subplots(figsize=(3.35,2.5));ax.axis('off')
    ax.text(.5,.68,title,ha='center',fontsize=11,transform=ax.transAxes)
    ax.text(.5,.38,'Main inference pending\nNo empirical values plotted',ha='center',fontsize=9,
            color='#757575',transform=ax.transAxes)
    save(fig,name)

def main():
    overview()
    status_path=ROOT/'results/main/status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    if not status.get('complete_primary'):
        for name,title in [('main_regret','Policy-level selection regret'),('country_regret','Country-group regret'),('coverage_risk','Coverage–risk operating points')]:
            pending(name,title)
        return
    summary=json.loads((ROOT/'results/main/summary.json').read_text())
    by={r['policy']:r for r in summary}
    selected=[p for p in ['Random','Qwen','Smol','Oracle'] if p in by]
    fig,ax=plt.subplots(figsize=(3.4,2.5))
    for y,p in enumerate(selected):
        r=by[p];lo,hi=r['regret_ci'];val=r['regret']
        ax.errorbar(val,y,xerr=[[val-lo],[hi-val]],fmt='o',color=COLORS[p],capsize=3)
        ax.annotate(f'{val:.3f}',(val,y),xytext=(6,7),textcoords='offset points',fontsize=8)
    ax.set_yticks(range(len(selected)),selected);ax.invert_yaxis();ax.set_xlabel('Selection regret (lower is better)')
    ax.set_xlim(left=-.015);ax.grid(axis='x',alpha=.18);fig.tight_layout();save(fig,'main_regret')
    country=json.loads((ROOT/'results/main/country.json').read_text())
    labels=sorted({r['country'] for r in country})
    fig,ax=plt.subplots(figsize=(3.4,3.6))
    policies=['Qwen']+(['Smol'] if 'Smol' in by else [])
    for j,p in enumerate(policies):
        for i,c in enumerate(labels):
            r=next((r for r in country if r['country']==c and r['policy']==p),None)
            if r is None:continue
            val=r['regret'];lo,hi=r['regret_ci']
            ax.errorbar(val,i+(j-.5)*.18,xerr=[[val-lo],[hi-val]],fmt='o',ms=4,
                        color=COLORS[p],capsize=2,label=p if i==0 else None)
    ax.set_yticks(range(len(labels)),[s.replace('_',' ') for s in labels]);ax.invert_yaxis()
    ax.set_xlim(left=-.01);ax.set_xlabel('Selection regret');ax.grid(axis='x',alpha=.18)
    ax.legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,'country_regret')
    df=pd.read_csv(ROOT/'results/main/prompt_metrics.csv')
    fig,ax=plt.subplots(figsize=(3.4,2.75))
    xx=[];yy=[];rr=[]
    for name,label in [('Qwen unanimous','3/3'),('Qwen agree 2/3','≥2/3'),('Qwen majority','≥1/3')]:
        if name not in by:continue
        row=by[name];sub=df[df.policy==name]
        xx.append(row['coverage']);yy.append(row['regret'])
        rr.append(float((sub.oracle_utility-sub.random_utility).mean()))
        ax.annotate(label,(xx[-1],yy[-1]),xytext=(3,6),textcoords='offset points',fontsize=8)
        lo,hi=row['regret_ci'];v=row['regret']
        ax.errorbar(xx[-1],v,yerr=[[v-lo],[hi-v]],fmt='none',color=COLORS['Qwen'],capsize=2)
    if xx:
        ax.plot(xx,yy,'o-',color=COLORS['Qwen'],label='Qwen majority')
        ax.plot(xx,rr,'s--',color=COLORS['Random'],label='Random on same subset')
    if 'Cross-model' in by:
        r=by['Cross-model'];ax.scatter([r['coverage']],[r['regret']],marker='D',color=COLORS['Smol'],label='Cross-model agreement')
    ax.set_xlabel('Automation coverage');ax.set_ylabel('Selective regret');ax.set_xlim(0,1.05)
    ax.set_ylim(bottom=0);ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=7.3)
    fig.tight_layout();save(fig,'coverage_risk')

if __name__=='__main__':main()
