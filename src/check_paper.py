"""Inspect page count, embedded fonts, unresolved references, blinding, and numeric provenance."""
import json
import argparse
import re
import subprocess
import pymupdf
from .common import ROOT,save_json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    doc=pymupdf.open(ROOT/'paper/main.pdf')
    texts=[p.get_text() for p in doc]
    # Only a standalone heading is treated as the start of references.
    refpage=next((i+1 for i,t in enumerate(texts) if re.search(r'(?m)^References\s*$',t)),None)
    appendixpage=next((i+1 for i,t in enumerate(texts) if re.search(r'(?m)^Reproduction Protocol\s*$',t)),None)
    log=(ROOT/'paper/main.log').read_text()
    source=(ROOT/'paper/main.tex').read_text()
    fonts=subprocess.run(['pdffonts',str(ROOT/'paper/main.pdf')],capture_output=True,text=True,check=True).stdout
    missing_fonts=[l for l in fonts.splitlines()[2:] if re.search(r'\sno\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',l)]
    forbidden_packages=['hyperref','geometry','authblk','balance','float','fullpage','multicol','setspace','titlesec','wrapfig']
    report={'pages_total':len(doc),'references_start_page':refpage,'appendix_start_page':appendixpage,
            'main_pages_upper_bound':refpage if refpage else len(doc),
            'author_metadata_empty':not doc.metadata.get('author'),
            'font_embedding_ok':not missing_fonts,
            'unresolved_references':bool(re.search(r'Citation .*undefined|Reference .*undefined|There were undefined references',log)),
            'overfull_boxes':bool(re.search(r'Overfull \\hbox',log)),
            'forbidden_packages':[p for p in forbidden_packages if re.search(r'\\usepackage(?:\[[^\]]*\])?\{'+p+r'\}',source)],
            'contributions_present':'Statement of Contributions' in ''.join(texts),
            'identifying_urls_present':bool(re.search(r'github\.com/seochan|andrew\.cmu|huichanseo\.com', ''.join(texts))),
            'draft_banner_present':'Working draft' in ''.join(texts),
            'em_dash_present':'\u2014' in ''.join(texts),
            'em_dash_source_files':[str(p.relative_to(ROOT)) for p in [ROOT/'paper/main.tex',ROOT/'paper/refs.bib',*sorted((ROOT/'paper/generated').glob('*.tex'))]
                                    if '\u2014' in p.read_text() or '---' in p.read_text() or '\\textemdash' in p.read_text()],
            'placeholder_tokens':re.findall(r'\[TODO\]|\[XX\]|\[X\]|\[MAIN RESULT\]', ''.join(texts))}
    report['pdf_technical_checks_pass']=report['main_pages_upper_bound']<=10 and report['author_metadata_empty'] and report['font_embedding_ok'] and not report['unresolved_references'] and not report['overfull_boxes'] and not report['forbidden_packages'] and not report['identifying_urls_present'] and not report['placeholder_tokens'] and not report['em_dash_present'] and not report['em_dash_source_files'] and report['contributions_present']
    status_path=ROOT/'results/main/status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    report['empirical_content_complete']=not report['draft_banner_present'] and status.get('complete_primary',False) and status.get('complete_secondary',False)
    save_json(ROOT/'results/pdf_checks.json',report)
    print(json.dumps(report,indent=2))
    if not report['pdf_technical_checks_pass']:
        raise SystemExit('PDF failed technical checks; do not submit.')
    if args.require_complete and not report['empirical_content_complete']:
        raise SystemExit('Main experiment is incomplete; do not package as final.')

if __name__=='__main__':main()
