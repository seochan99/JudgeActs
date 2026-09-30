# JudgeActs

**When the Judge Acts: Auditing VLM-Guided Image Selection on Culturally Situated Prompts**
Huichan Seo · Independent Researcher

[Project page](https://seochan99.github.io/JudgeActs/) · [Data](https://huggingface.co/datasets/seochan99/JudgeActs) · Paper (coming soon)

When a vision-language model (VLM) picks which generated image a user receives, its choice is an action, not a score. This repository audits such choices on 300 culturally situated prompts from CulturalFrames. Each returned image is compared with released human ratings the judge never sees, and with exact random choice on the same candidate pool, under three rotations of candidate order.

Main findings, all regenerated from the recorded outputs in this repository:

- A 4B-parameter judge (Qwen3-VL-4B) beats random choice only slightly (gain 0.039, 95% CI [0.016, 0.062]) and less than a CLIP similarity scorer (0.069).
- It picks the first-listed image in 48.8% of calls (27.8% expected) and changes its choice under reordering on 59.7% of prompts.
- Requiring agreement across orders keeps 40.3% of prompts, on which the judge beats random by 0.141, and rejects prompts on which it falls below random. Agreement with a weaker second judge does the reverse.
- Both selectors reduce missing cultural expectations but slightly raise stereotype ratings.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Data preparation, frozen split construction, judge runners, validation, packaging |
| `analysis/` | Metrics, bootstrap analysis, figures, manuscript text generation |
| `configs/`, `prompts/` | Frozen protocol configuration and judge instructions |
| `data/manifests/`, `data/derived/` | Frozen prompt IDs, presentation orders, aggregated ratings (no images) |
| `runs/` | Recorded judge outputs for every call, including raw text |
| `results/main/` | Prompt-level metrics and every reported estimate |
| `paper/` | LaTeX source; numbers in `paper/generated/` are produced by the analysis code |
| `tools/audit_browser/` | Offline browser for inspecting every recorded decision |
| `provenance/` | Source revisions, model and dataset cards, protocol decisions, checksums |
| `tests/` | Unit tests for metrics and manuscript generation |

The experimental split and both judge interfaces were frozen in separate commits before main inference; see the commit history.

## Reproduce

```sh
uv sync
uv run python -m src.fetch dataset
uv run python -m src.fetch qwen
uv run python -m src.fetch smol
uv run python -m src.prepare
uv run python -m src.run_judge --model qwen --split main --permutations 3
uv run python -m src.run_smol --split main --permutations 3
uv run python -m analysis.analyze --split main
uv run python -m analysis.render_paper
uv run python -m src.check_paper
```

The recorded outputs in `runs/` let you rerun every analysis without inference. Seed: `20261002`.

To browse decisions locally:

```sh
uv run python tools/audit_browser/export.py
cd tools/audit_browser && python -m http.server 8765
```

## Data and images

This project uses the public CulturalFrames release and public model weights. The CulturalFrames dataset card does not declare a license, so source images and raw annotation records are not redistributed here. Figures reproduce a small number of dataset images with attribution for illustration. Please cite CulturalFrames if you use this work.

## Citation

```bibtex
@misc{seo2026judgeacts,
  title  = {When the Judge Acts: Auditing VLM-Guided Image Selection on Culturally Situated Prompts},
  author = {Seo, Huichan},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Code is released under the MIT License. Dataset images and annotations remain subject to their original terms.
