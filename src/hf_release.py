"""Assemble (and optionally upload) the Hugging Face dataset release.

The release contains recorded judge outputs, aggregated ratings per candidate, and every
analysis output. Source images are not included; they are identified by CulturalFrames ids.
"""
import argparse
import json
import shutil

from .common import ROOT, load_jsonl

OUT = ROOT / "release" / "hf_dataset"
REPO = "seochan99/JudgeActs"

CARD = """---
license: cc-by-4.0
pretty_name: JudgeActs
language:
- en
tags:
- vision-language-models
- llm-as-a-judge
- position-bias
- cultural-bias
- evaluation
- text-to-image
size_categories:
- 1K<n<10K
configs:
- config_name: pools
  data_files: pools.jsonl
- config_name: judge_calls
  data_files:
  - split: qwen
    path: judge_calls/qwen_main.jsonl
  - split: smol
    path: judge_calls/smol_main.jsonl
- config_name: ablations
  data_files:
  - split: choice_only
    path: ablations/choice_only.jsonl
  - split: opaque_labels
    path: ablations/opaque_labels.jsonl
  - split: rotation4
    path: ablations/rotation4.jsonl
- config_name: scale_extension
  data_files:
  - split: qwen4b_8bit_control
    path: extension/qwen4b_mlx_main.jsonl
  - split: qwen8b_8bit
    path: extension/qwen8b_mlx_main.jsonl
- config_name: prompt_metrics
  data_files: metrics/prompt_metrics.csv
---

# JudgeActs: recorded VLM selection decisions on culturally situated prompts

Data release for **When the Judge Acts: Auditing VLM-Guided Image Selection on Culturally Situated Prompts** (Huichan Seo).

- Project page: https://seochan99.github.io/JudgeActs/
- Code: https://github.com/seochan99/JudgeActs

A vision-language model judge chooses one of three or four generated images for each of 300 culturally situated prompts, under three cyclic presentation orders. Every choice is joined to released human ratings that the judge never saw.

## Contents

| Config / file | Description |
| --- | --- |
| `pools` | 300 frozen candidate pools: prompt, country, category, and per-candidate aggregated human ratings (prompt alignment, stereotype, missing explicit and implicit expectations, image quality, overall satisfaction) |
| `judge_calls` | Every recorded call for Qwen3-VL-4B-Instruct (900) and SmolVLM2-2.2B-Instruct (900): presentation mapping, raw output, parsed choice, validity flags, timing |
| `ablations` | Post-hoc Qwen runs: choice-only output format, opaque labels, and a fourth rotation for four-image pools |
| `scale_extension` | Qwen3-VL-8B (8-bit, MLX) on the same prompts and orders, plus an 8-bit Qwen3-VL-4B quantization control |
| `prompt_metrics` | Prompt-level outcomes for every policy (regret, gain over exact random choice, below-mean indicator, cultural-error ratings) |
| `results/` | Every summary estimate reported in the paper, with bootstrap intervals |
| `manifests/` | Frozen development, main, and reserve splits and presentation orders |

The key `hsr` in result files stores the below-mean rate (BMR) reported in the paper.

## Images and ratings

Images are **not** included. Candidate ids match the public CulturalFrames release (Nayak et al., Findings of EMNLP 2025), from which images and the underlying annotations can be obtained under its terms. Aggregated ratings here are derived from that release; the CulturalFrames dataset card does not declare a license, so please respect its original terms. The judge outputs and analysis files are released under CC BY 4.0.

## Citation

```bibtex
@misc{seo2026judgeacts,
  title  = {When the Judge Acts: Auditing VLM-Guided Image Selection on Culturally Situated Prompts},
  author = {Seo, Huichan},
  year   = {2026},
  note   = {Preprint}
}
```
"""


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "judge_calls").mkdir(parents=True)
    pools = []
    for g in load_jsonl(ROOT / "data/derived/main.jsonl"):
        pools.append({"prompt_id": g["prompt_id"], "prompt": g["prompt"], "country": g["country"],
                      "category": g["category"],
                      "candidates": [{k: v for k, v in c.items() if k != "image"} for c in g["candidates"]]})
    with open(OUT / "pools.jsonl", "w") as f:
        for p in pools:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    for m in ["qwen", "smol"]:
        shutil.copy(ROOT / f"runs/{m}_main.jsonl", OUT / "judge_calls" / f"{m}_main.jsonl")
    (OUT / "ablations").mkdir()
    for p in sorted((ROOT / "runs/ablations").glob("*.jsonl")):
        shutil.copy(p, OUT / "ablations" / p.name)
    (OUT / "extension").mkdir()
    for p in sorted((ROOT / "runs/extension").glob("*mlx_main.jsonl")):
        shutil.copy(p, OUT / "extension" / p.name)
    (OUT / "metrics").mkdir()
    for name in ["prompt_metrics.csv", "order_metrics.csv"]:
        shutil.copy(ROOT / "results/main" / name, OUT / "metrics" / name)
    (OUT / "results").mkdir()
    for p in sorted((ROOT / "results/main").glob("*.json")):
        shutil.copy(p, OUT / "results" / p.name)
    shutil.copytree(ROOT / "data/manifests", OUT / "manifests")
    (OUT / "README.md").write_text(CARD)
    print("built", OUT)


def upload():
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=REPO, repo_type="dataset",
                      commit_message="Release recorded judge outputs, pools, and analysis results")
    print("uploaded https://huggingface.co/datasets/" + REPO)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    build()
    if args.upload:
        upload()
