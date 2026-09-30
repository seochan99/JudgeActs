"""Build a de-anonymized arXiv source package from the submission manuscript.

The anonymous submission (paper/main.tex) is left untouched. The arXiv copy differs only in
the author block, the removed review notice, and a code link.
"""
import re
import shutil
import subprocess
import zipfile

from .common import ROOT

PAPER = ROOT / "paper"
OUT = ROOT / "arxiv-source"
AUTHOR = r"""\author{Huichan Seo}
\affiliations{Independent Researcher\\
gmlcks00513@gmail.com}"""
CODE = r"""\begin{links}
\link{Code and project page}{https://github.com/seochan99/JudgeActs}
\end{links}"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    tex = (PAPER / "main.tex").read_text()
    tex = tex.replace(r"\usepackage[submission]{aaai2027}", r"\usepackage{aaai2027}" + "\n" + r"\nocopyright")
    tex = re.sub(r"\\author\{Anonymous Submission\}\s*\\affiliations\{\}", lambda _: AUTHOR, tex)
    tex = tex.replace(r"\maketitle", r"\maketitle" + "\n" + CODE, 1) if r"\begin{links}" in (PAPER / "aaai2027.sty").read_text() or r"\newenvironment{links}" in (PAPER / "aaai2027.sty").read_text() else tex
    (OUT / "main.tex").write_text(tex)
    for name in ["refs.bib", "aaai2027.sty", "aaai2027.bst"]:
        shutil.copy(PAPER / name, OUT / name)
    shutil.copytree(PAPER / "generated", OUT / "generated")
    refs = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]+)\}", tex))
    (OUT / "figures").mkdir()
    for r in refs:
        shutil.copy(PAPER / r, OUT / r)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=OUT, check=True,
                   stdout=subprocess.DEVNULL)
    keep = {"main.tex", "main.bbl", "refs.bib", "aaai2027.sty", "aaai2027.bst"}
    with zipfile.ZipFile(ROOT / "arxiv-source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            rel = p.relative_to(OUT)
            if p.is_file() and (str(rel) in keep or rel.parts[0] in {"generated", "figures"}):
                z.write(p, rel)
    if (ROOT / "docs").exists():
        # Web copy: downsampled images keep the project page light.
        subprocess.run(["gs", "-q", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.6", "-dPDFSETTINGS=/ebook",
                        "-dColorImageResolution=200", "-dNOPAUSE", "-dBATCH",
                        f"-sOutputFile={ROOT / 'docs' / 'paper.pdf'}", str(OUT / "main.pdf")], check=True)
    print("arXiv source:", ROOT / "arxiv-source.zip", "| PDF:", OUT / "main.pdf")


if __name__ == "__main__":
    main()
