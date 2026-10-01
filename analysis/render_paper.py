"""Build anonymous manuscript text, figures and PDF from recorded artifacts."""
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import ROOT
from .manuscript_text import pending, completed, pct

GEN = ROOT / "paper/generated"


def write(name, content):
    if "\u2014" in content or "---" in content or "\\textemdash" in content:
        raise ValueError(f"Prohibited punctuation in {name}")
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / f"{name}.tex").write_text(content + "\n")


def read(path):
    return json.loads(path.read_text())


def main():
    audit = read(ROOT / "results/data_audit.json")
    sources = read(ROOT / "provenance/sources.json")
    groups = [json.loads(line) for line in (ROOT / "data/derived/main.jsonl").read_text().splitlines()]
    n_three = sum(len(g["candidates"]) == 3 for g in groups)
    main_ties = sum(np.ptp([c["utility"] for c in g["candidates"]]) <= 1e-12 for g in groups)
    write("macros", "% Values are derived from the recorded public data and main inference.")
    write("data",
        f"The release contains {audit['rows']:,} image rows and {audit['annotation_records']:,} "
        f"image annotation records, comprising {audit['annotation_count']:,} individual ratings. "
        f"{audit['missing_annotation_rows']} image rows have no annotation record "
        f"({pct(audit['annotation_missing_rate'])}). Filtering leaves "
        f"{audit['eligible_before_prompt_deduplication']} eligible sets. Removing "
        f"{len(audit['excluded_duplicate_prompt_ids'])} repeated requests within countries leaves "
        f"{audit['eligible_prompts']} unique prompts: {audit['candidate_count_distribution']['3']} "
        f"three-candidate and {audit['candidate_count_distribution']['4']} four-candidate sets. "
        f"The main split has {n_three} three-candidate and {len(groups)-n_three} four-candidate sets; "
        f"{main_ties} sets have identical candidate alignment scores.")
    write("reproduction",
        r"Seed: 20261002. Dataset revision: \texttt{" + sources["dataset"]["revision"][:12] +
        r"}. Qwen revision: \texttt{" + sources["models"]["qwen"]["revision"][:12] +
        r"}. Smol revision: \texttt{" + sources["models"]["smol"]["revision"][:12] +
        r"}. Full revision identifiers and runtime versions are in the provenance manifest. "
        r"The main split contains 300 prompts, 30 per country; the dev split contains 30 disjoint prompts. "
        r"Each prompt has three cyclic candidate orders. Qwen uses a 448-pixel maximum image edge; "
        r"Smol uses one 384-pixel native view. Both use greedy decoding and a 64-token limit. "
        f"The image audit finds {len(audit['broken_rows'])} undecodable images and "
        f"{len(audit['join_mismatches'])} identifier/content mismatches. Four repeated encoded-image "
        r"pairs occur across the duplicate requests removed before splitting.")
    result_dir = ROOT / "results/main"
    status = read(result_dir / "status.json") if (result_dir / "status.json").exists() else {}
    if status.get("complete_primary") and status.get("complete_secondary"):
        fragments = completed(
            read(result_dir / "summary.json"),
            read(result_dir / "country.json"),
            read(result_dir / "decomposition.json"),
            pd.read_csv(result_dir / "order_metrics.csv"),
            status,
            read(result_dir / "category.json"),
            read(result_dir / "position.json"),
            read(result_dir / "baselines.json") if (result_dir / "baselines.json").exists() else None,
            read(result_dir / "ablations.json") if (result_dir / "ablations.json").exists() else None,
        )
    else:
        fragments = pending()
    for name, content in fragments.items():
        write(name, content)
    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "analysis.figures"], cwd=ROOT, check=True)
    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "analysis.extra_text"], cwd=ROOT, check=True)
    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "analysis.review_text"], cwd=ROOT, check=True)
    # Build in an isolated copy: editors that auto-compile paper/ (e.g. LaTeX Workshop) can race with this build.
    import shutil, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp) / "paper"
        shutil.copytree(ROOT / "paper", build, ignore=shutil.ignore_patterns(
            "main.aux", "main.bbl", "main.blg", "main.fdb_latexmk", "main.fls", "main.log", "main.pdf", "main.synctex.gz"))
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                       cwd=build, check=True, stdout=subprocess.DEVNULL)
        for name in ["main.pdf", "main.log", "main.bbl"]:
            shutil.copy(build / name, ROOT / "paper" / name)


if __name__ == "__main__":
    main()
