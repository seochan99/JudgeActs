"""Create a reviewable anonymous submission bundle and preserve numerical provenance."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from .common import ROOT,save_json

def main():
    sources=json.loads((ROOT/'provenance/sources.json').read_text())
    env=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],capture_output=True,text=True)
    if env.returncode:
        env=subprocess.run(['uv','pip','freeze'],cwd=ROOT,capture_output=True,text=True,check=True)
    (ROOT/'provenance/environment.lock.txt').write_text(env.stdout)
    for kind in ['dataset','qwen','smol']:
        source=ROOT/'data/raw/culturalframes/README.md' if kind=='dataset' else ROOT/'models'/kind/'README.md'
        if source.exists():(ROOT/f'provenance/{kind}_card.md').write_text(source.read_text())
    paths=[ROOT/'data/raw/culturalframes/culturalframes_human_annotations.json',ROOT/'prompts/judge_primary.txt',
           ROOT/'configs/protocol.json',ROOT/'src/run_judge.py',ROOT/'src/run_smol.py',
           ROOT/'src/common.py',ROOT/'src/prepare.py',ROOT/'analysis/analyze.py',
           ROOT/'analysis/metrics.py',ROOT/'analysis/render_paper.py',
           ROOT/'analysis/manuscript_text.py',ROOT/'src/check_paper.py',
           ROOT/'analysis/overview_figure.py',ROOT/'analysis/empirical_figures.py',ROOT/'analysis/hero_figure.py',ROOT/'analysis/framework_figure.py',ROOT/'analysis/gallery_figure.py',ROOT/'analysis/result_figures.py',ROOT/'analysis/appendix_tables.py',ROOT/'analysis/style.py']
    paths+=list((ROOT/'data/manifests').glob('*.json'))
    paths+=list((ROOT/'runs').glob('*.jsonl'))
    paths+=list((ROOT/'results/main').glob('*.json'))
    paths+=list((ROOT/'results/main').glob('*.csv'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    save_json(ROOT/'provenance/checksums.json',hashes)
    paper=ROOT/'paper'
    files=[paper/'main.tex',paper/'refs.bib',paper/'aaai2027.sty',paper/'aaai2027.bst',paper/'main.pdf']
    files+=sorted((paper/'generated').glob('*.tex'))
    # Only figures the manuscript actually references.
    import re
    refs=set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]+)\}',(paper/'main.tex').read_text()))
    files+=sorted(paper/r for r in refs)
    with zipfile.ZipFile(ROOT/'submission-source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(paper))
    abstract=(paper/'generated/abstract.tex').read_text().strip()
    plain=abstract.replace('\\%','%')
    title='When the Judge Acts: Auditing VLM-Guided Image Selection on Culturally Situated Prompts'
    (ROOT/'submission-metadata.txt').write_text(title+'\n\n'+plain+'\n')
    print('Anonymous source bundle and submission metadata created.')

if __name__=='__main__':main()
