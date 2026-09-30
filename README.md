This project uses only independently released public datasets and public model weights.

# When the Judge Acts

Independent IASEAI'27 submission project. Public CulturalFrames images and annotations, public Qwen3-VL-4B-Instruct and SmolVLM2-2.2B-Instruct weights, newly written code and analyses. No previous laboratory repositories, private datasets, annotations, prompts, or experimental outputs are inputs to this project.

Seed: `20261002`. Independent unit: prompt candidate set. Primary utility: mean public human prompt-alignment score. Results are generated from recorded inference outputs; no synthetic empirical results are permitted.

The public dataset card does not declare a dataset license. Public availability is recorded separately from permission to redistribute. Raw images and annotator demographics are excluded from git and the manuscript figures. Confirm terms with the dataset maintainers before redistribution.

Project artifacts:

- `paper/main.pdf`: anonymous AAAI 2027 manuscript.
- `paper/main.tex`: complete editable source; generated empirical text is in `paper/generated`.
- `paper/figures`: four vector PDF figures and PNG previews.
- `data/manifests`: frozen development, main, and reserve prompt IDs and presentation orders.
- `results/main`: prompt-level CSVs, country/decomposition estimates, bootstrap intervals, and output validity.
- `provenance`: public revisions, source cards, protocol decisions, checksums, and pinned environment.

Reproduce:

```sh
uv sync
uv run python -m src.fetch dataset
uv run python -m src.fetch qwen
uv run python -m src.fetch smol
uv run python -m src.prepare
uv run python -m src.run_judge --model qwen --split main --permutations 3
uv run python -m src.run_smol --split main --permutations 3
uv run python -m src.validate --split main
uv run python -m analysis.analyze --split main
uv run python -m analysis.render_paper
uv run python -m src.check_paper
uv run python -m src.finalize
```

The working PDF explicitly labels an incomplete primary experiment. Empirical figures and text remain pending until the 900-call Qwen main run finishes. Never interpret development metrics or blank empirical panels as main results.

OpenReview account approval and portal connection were confirmed by the user. October 2 is the internal submission target; the supplied official guide states October 5, 2026 AOE. The final author list, track choice, financial disclosures, attendance commitment, and actual upload remain the authors' submission tasks.
