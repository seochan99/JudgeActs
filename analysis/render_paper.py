"""Generate every empirical manuscript number from analysis artifacts."""
import json
import subprocess
from pathlib import Path
import pandas as pd
from src.common import ROOT, save_json

GEN=ROOT/'paper/generated'
def write(name,text):
    GEN.mkdir(parents=True,exist_ok=True)
    (GEN/f'{name}.tex').write_text(text+'\n')

def val(x):return f'{x:.3f}'
def pct(x):return f'{100*x:.1f}\\%'
def interval(row,key):return '['+', '.join(val(v) for v in row[key+'_ci'])+']'
def tex(s):return s.replace('_',' ')

def main():
    audit=json.loads((ROOT/'results/data_audit.json').read_text())
    sources=json.loads((ROOT/'provenance/sources.json').read_text())
    write('macros','% All quantities are generated from recorded public data and inference artifacts.')
    main_groups=[json.loads(l) for l in (ROOT/'data/derived/main.jsonl').read_text().splitlines()]
    n_three=sum(len(g['candidates'])==3 for g in main_groups)
    write('data',f"The release contains {audit['rows']:,} image rows and {audit['annotation_records']:,} image annotation records, comprising {audit['annotation_count']:,} individual annotations. The join leaves {audit['missing_annotation_rows']} unannotated image rows ({pct(audit['annotation_missing_rate'])}). We find {len(audit['broken_rows'])} undecodable images and {len(audit['join_mismatches'])} join mismatches. Filtering yields {audit['eligible_before_prompt_deduplication']} sets; removing {len(audit['excluded_duplicate_prompt_ids'])} repeated prompt identifiers leaves {audit['eligible_prompts']} eligible unique prompts, with {audit['candidate_count_distribution']['3']} three-candidate and {audit['candidate_count_distribution']['4']} four-candidate sets. {audit['constant_utility_sets']} eligible sets have identical candidate utilities. Four pairs of repeated encoded-image content occur across the duplicated requests; those requests are deduplicated before splitting. The main split contains {n_three} three-candidate and {300-n_three} four-candidate sets.")
    write('reproduction',r'Seed: 20261002. Dataset revision: \texttt{'+sources['dataset']['revision'][:12]+r'}. Qwen revision: \texttt{'+sources['models']['qwen']['revision'][:12]+r'}. Smol revision: \texttt{'+sources['models']['smol']['revision'][:12]+r'}. Complete SHA-1 identifiers and runtime versions are in the provenance and environment manifests. The frozen main split contains 300 unique prompts, 30 per country; the development split contains 30 disjoint prompts. The experiment uses three distinct cyclic candidate orders, 448-pixel maximum image edge, greedy decoding, and at most 64 new tokens per request.')
    status_path=ROOT/'results/main/status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    if not status.get('complete_primary'):
        write('status',r'\begin{quote}\textbf{Working draft --- empirical experiment incomplete.} Main-run findings are not yet available. This PDF is not submission-ready.\end{quote}')
        write('abstract',"Vision-language models (VLMs) are used to evaluate and select generated images, yet agreement with human scores does not by itself establish the reliability of the resulting decisions. We formulate a public-data audit of VLM-guided image selection using CulturalFrames and its existing human annotations. The protocol compares selected human-rated utility with exact random selection and a retrospective annotation oracle, reports selection regret and below-baseline steering rates, and analyzes country-group risks. A frozen split contains 300 prompts, with three candidate-order presentations per prompt. Selective policies use order consistency and cross-model agreement to trade automation coverage for conditional risk. The manuscript specifies the audit and its limitations; empirical findings will be reported only after the recorded main experiment completes.")
        write('results',r'The frozen main experiment has not completed. No main-run utility, regret, harmful steering rate, subgroup comparison, or mitigation effect is claimed in this working draft. The generation scripts will populate this section, both tables, and the three empirical figures from recorded outputs after the complete primary run.')
        write('table_main',r'\begin{tabular}{lcccccc}\toprule Policy & Utility & Regret & HSR & Best & Worst group & Coverage\\\midrule Main run pending & -- & -- & -- & -- & -- & --\\\bottomrule\end{tabular}')
        write('table_decomposition',r'\begin{tabular}{lccc}\toprule Condition & $N$ & Regret & HSR\\\midrule Main run pending & -- & -- & --\\\bottomrule\end{tabular}')
    else:
        summary=json.loads((ROOT/'results/main/summary.json').read_text());by={r['policy']:r for r in summary}
        q=by['Qwen'];ran=by['Random'];o=by['Oracle']
        un=by.get('Qwen unanimous')
        c=json.loads((ROOT/'results/main/country.json').read_text());qcountry=[r for r in c if r['policy']=='Qwen'];worst=max(qcountry,key=lambda r:r['regret'])
        lo,hi=q['gain_ci'];direction='increased' if lo>0 else ('decreased' if hi<0 else 'changed')
        abstention=(f"Unanimous selection retained {pct(un['coverage'])} coverage with conditional regret {val(un['regret'])}. " if un else 'No set met the unanimity rule. ')
        second='two open-weight judges' if 'Smol' in by else 'an open-weight judge'
        write('abstract',f"Vision-language models (VLMs) are used to evaluate and select generated images, yet agreement with human scores does not by itself establish the reliability of the resulting decisions. We audit cultural risk in VLM-guided image selection using CulturalFrames and its existing human annotations. Across 300 frozen prompt-level candidate sets and three candidate-order presentations, we evaluate {second} against exact random selection and a retrospective annotation oracle. Qwen selection {direction} mean prompt-alignment utility by {val(q['gain'])} (95\\% CI {interval(q,'gain')}), with selection regret {val(q['regret'])}. Selected utility fell below the candidate-set mean in {pct(q['hsr'])} of valid original-order decisions. The highest country-group mean regret was {val(worst['regret'])}. {abstention}These measurements describe reference-relative selection outcomes, rather than observed user injury or a universal cultural preference. The audit connects automated judgments to their selected outputs and distinguishes average gains, distributional risk, and the conditional risk of allowing a judge to act.")
        write('status','% Primary main experiment complete; administrative submission remains external.')
        rtext=r'\subsection{Direct Selection Outcomes}'+'\n'
        rtext+=f"Table~\\ref{{tab:main}} reports outcomes on the 300-prompt frozen split. Qwen returned valid original-order decisions for {q['n']} prompts (coverage {pct(q['coverage'])}). Mean selected utility was {val(q['utility'])}, compared with full-split random utility {val(ran['utility'])} and oracle utility {val(o['utility'])}. The paired gain relative to random on exactly the selected prompts was {val(q['gain'])}, with 95\\% CI {interval(q,'gain')}. "
        if lo<=0<=hi:rtext+='The interval spans zero, leaving the sign of average gain unresolved. '
        else:rtext+=f"The interval lies {'above' if lo>0 else 'below'} zero under the stated prompt-level reference. "
        rtext+=f"Mean regret was {val(q['regret'])} (95\\% CI {interval(q,'regret')}); median regret was {val(q['regret_median'])}. Human-best selection was {pct(q['human_best'])}, and near-best selection was {pct(q['near_best'])}. HSR was {pct(q['hsr'])} (Wilson interval {interval(q,'hsr')}), or {pct(q['hsr05'])} with margin $\\delta=0.05$. The exact random HSR was {pct(ran['hsr'])}, emphasizing that a positive automated HSR alone does not establish degradation relative to random. Selection below the candidate median occurred in {pct(q['bottom_half'])} of valid decisions.\n\n"
        if 'Smol' in by:
            s=by['Smol'];rtext+=f"Smol produced {s['n']} valid original-order selections, with mean utility {val(s['utility'])}, regret {val(s['regret'])}, and HSR {pct(s['hsr'])}. Its paired gain was {val(s['gain'])} (95\\% CI {interval(s,'gain')}). These model-specific outcomes are evaluated on the same frozen candidate pools; differences in valid-output coverage are retained.\n\n"
        rtext+=r'\subsection{Country-Group Risk}'+'\n'
        rtext+=f"Figure~\\ref{{fig:country}} shows subgroup estimates. The largest observed Qwen group mean regret was in {tex(worst['country'])}: {val(worst['regret'])} (95\\% CI {interval(worst,'regret')}, $n={worst['n']}$ valid decisions). The maximum-minus-minimum disparity was {val(q['disparity'])}. Recomputing the maximum across represented groups in bootstrap replicates gives worst-group regret interval {interval(q,'worst_group_regret')}. This is a descriptive maximum with sampling uncertainty, not evidence that its group has a uniquely canonical preference.\n\n"
        decomp=json.loads((ROOT/'results/main/decomposition.json').read_text());qq=[r for r in decomp if r['policy']=='Qwen']
        rtext+=r'\subsection{Annotation-Sensitive Conditions and Stereotypes}'+'\n'
        rtext+='Table~\\ref{tab:decomposition} uses the predeclared overlapping conditions. '
        for row in qq:
            rtext+=f"{row['condition']}-sensitive sets included {row['n']} valid decisions, with regret {val(row['regret'])} and HSR {pct(row['hsr'])}. "
        rtext+=f"Qwen selected images with mean stereotype annotation rate {val(q['stereotype'])}; the full-split random rate was {val(ran['stereotype'])}. These are annotation proportions, not the fraction of images carrying a unanimous binary stereotype label. The corresponding overall-satisfaction utility was {val(q['overall'])}, and mean image-quality score was {val(q['image_quality'])}. The auxiliary outcomes are reported separately from primary alignment.\n\n"
        rtext+=r'\subsection{Order Consistency and Selective Control}'+'\n'
        order=pd.read_csv(ROOT/'results/main/order_metrics.csv');oc=order[order.model=='qwen']
        rtext+=f"All three order calls parsed successfully for {len(oc)} prompts. Among these, the candidate identity changed across presentations in {pct(float(oc['flip'].mean()))} of sets. Mean normalized choice entropy was {val(float(oc['entropy'].mean()))}. Figure~\\ref{{fig:coverage}} reports majority-selection operating points and matched-subset random comparisons. "
        if un:
            rtext+=f"Requiring unanimity retained {un['n']} prompts ({pct(un['coverage'])} coverage), with selective regret {val(un['regret'])} (95\\% CI {interval(un,'regret')}) and HSR {pct(un['hsr'])}. Its paired utility gain over random within the accepted subset was {val(un['gain'])} (95\\% CI {interval(un,'gain')}). This conditional estimate does not assign outcomes to rejected requests. "
        if 'Cross-model' in by:
            cm=by['Cross-model'];rtext+=f"Cross-model agreement retained {cm['n']} prompts ({pct(cm['coverage'])}) with regret {val(cm['regret'])} and HSR {pct(cm['hsr'])}. "
        rtext+='Only represented country groups enter a selective worst-group estimate; the artifact reports missing groups and subgroup coverage.\n\n'
        rtext+=r'\subsection{Output Validity}'+'\n'
        for name,v in status['validation'].items():
            rtext+=f"{name.capitalize()} produced {v['valid']} valid outputs and {v['invalid']} invalid outputs across {v['calls']} recorded calls. "
        rtext+='Invalid calls remain explicit abstentions. The valid original-order denominator governs direct-policy proportions; the full frozen split governs coverage.'
        write('results',rtext)
        rows=[r'\begin{tabular}{lcccccc}',r'\toprule',r'Policy & Utility & Regret & HSR & Human-best & Worst group & Coverage\\',r'\midrule']
        for name in ['Random','Qwen','Smol','Qwen unanimous','Cross-model','Oracle']:
            if name not in by:continue
            r=by[name];rows.append(name+' & '+' & '.join([val(r['utility']),val(r['regret']),val(r['hsr']),val(r['human_best']),val(r['worst_group_regret']),val(r['coverage'])])+r'\\')
        rows.extend([r'\bottomrule',r'\end{tabular}']);write('table_main','\n'.join(rows))
        rows=[r'\begin{tabular}{lccc}',r'\toprule',r'Condition & $N$ & Regret & HSR\\',r'\midrule']
        for r in qq:rows.append(r['condition']+' & '+str(r['n'])+' & '+val(r['regret'])+' & '+val(r['hsr'])+r'\\')
        rows.extend([r'\bottomrule',r'\end{tabular}']);write('table_decomposition','\n'.join(rows))
    subprocess.run([str(ROOT/'.venv/bin/python'),'-m','analysis.figures'],cwd=ROOT,check=True)
    subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=ROOT/'paper',check=True)

if __name__=='__main__':main()
