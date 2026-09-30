"""Resume frozen main inference, then build analyses, figures, PDF, and source bundle."""
import json
import subprocess
import sys
from .common import ROOT,load_jsonl

def run(module,*args):
    subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,check=True)

def complete(path,expected):
    if not path.exists():return False
    rows=load_jsonl(path)
    return len({(r['prompt_id'],r['permutation']) for r in rows})==expected

def main():
    expected=len(load_jsonl(ROOT/'data/derived/main.jsonl'))*3
    if not complete(ROOT/'runs/qwen_main.jsonl',expected):
        run('src.run_judge','--model','qwen','--split','main','--permutations','3')
    if not complete(ROOT/'runs/smol_main.jsonl',expected):
        run('src.run_smol','--split','main','--permutations','3')
    run('src.validate','--split','main')
    run('analysis.analyze','--split','main')
    run('analysis.render_paper')
    run('src.check_paper','--require-complete')
    run('src.finalize')

if __name__=='__main__':main()
