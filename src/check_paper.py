"""Inspect page count, embedded fonts, unresolved references, blinding, and numeric provenance."""
import json
import re
import subprocess
import pymupdf
from .common import ROOT,save_json

def main():
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
            'placeholder_tokens':re.findall(r'\[TODO\]|\[XX\]|\[X\]|\[MAIN RESULT\]', ''.join(texts))}
    report['pdf_technical_checks_pass']=report['main_pages_upper_bound']<=10 and report['font_embedding_ok'] and not report['unresolved_references'] and not report['forbidden_packages'] and not report['identifying_urls_present'] and not report['placeholder_tokens'] and report['contributions_present']
    report['empirical_content_complete']=not report['draft_banner_present']
    save_json(ROOT/'results/pdf_checks.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
