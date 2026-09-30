"""Content-based reference selector (reviewer request): CLIP prompt-image similarity, argmax per prompt.

Exploratory, added after main inference. The selector sees only the prompt text and the candidate
pixels (never generator identity or human ratings) and is position-invariant by construction: every
candidate is scored independently, so presentation order cannot affect the choice.
"""
import json

import numpy as np
import pandas as pd

from src.common import ROOT, load_jsonl, save_json
from .analyze import summarize
from .appendix_tables import tab, f3, ci, pc
from .metrics import candidate_metrics, bootstrap_mean

MODEL_ID = "openai/clip-vit-large-patch14"
RES = ROOT / "results/main"
CACHE = RES / "clip_scores.json"
AXES = ["stereotype", "missing_explicit", "missing_implicit", "image_quality", "overall"]


def clip_scores(groups):
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        want = {c["id"] for g in groups for c in g["candidates"]}
        if want <= set(cached["scores"]):
            return cached
    import torch
    from PIL import Image
    from huggingface_hub import model_info
    from transformers import CLIPModel, CLIPProcessor
    revision = model_info(MODEL_ID).sha
    # CPU on purpose: a concurrent job holds the accelerator.
    torch.set_num_threads(4)
    model = CLIPModel.from_pretrained(MODEL_ID, revision=revision).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID, revision=revision)
    scores = {}
    with torch.no_grad():
        for g in groups:
            cs = g["candidates"]
            ims = [Image.open(ROOT / c["image"]).convert("RGB") for c in cs]
            inp = proc(text=[g["prompt"]], images=ims, return_tensors="pt", padding=True, truncation=True, max_length=77)
            o = model(**inp)
            # text_embeds / image_embeds are the projected, L2-normalised CLIP embeddings.
            for c, v in zip(cs, (o.image_embeds @ o.text_embeds.T).squeeze(-1).tolist()):
                scores[c["id"]] = float(v)
    out = {"model": MODEL_ID, "revision": revision, "score": "cosine(text_embed, image_embed)",
           "device": "cpu", "precision": "float32", "text_truncation": 77, "scores": scores}
    save_json(CACHE, out)
    return out


def select(group, scores):
    # Candidates are stored in canonical (sorted id) order; the first maximum wins ties.
    ids = sorted(c["id"] for c in group["candidates"])
    vals = [scores[i] for i in ids]
    return ids[int(np.argmax(vals))]


def policy_rows(df, groups, choice, policy):
    """Rows with the same columns as build_rows(), reusing the per-prompt Random base columns."""
    base = df[df.policy == "Random"].set_index("prompt_id")
    base_cols = list(df.columns[:list(df.columns).index("policy")])
    rows = []
    for g in groups:
        if g["prompt_id"] not in choice:
            continue
        cs = g["candidates"]; ids = [c["id"] for c in cs]; sc = [c["utility"] for c in cs]
        i = ids.index(choice[g["prompt_id"]])
        row = {k: base.loc[g["prompt_id"], k] for k in base_cols if k != "prompt_id"}
        row.update({"prompt_id": g["prompt_id"], "policy": policy, **candidate_metrics(sc, i), "selected_id": ids[i]})
        for a in AXES:
            row[a] = cs[i][a]
        ov = [c["overall"] for c in cs]
        if all(v is not None for v in ov):
            row["overall_regret"] = max(ov) - ov[i]
            row["overall_gain"] = ov[i] - float(np.mean(ov))
        rows.append(row)
    return pd.DataFrame(rows)[list(df.columns)]


def brief(r):
    keys = ["n", "coverage", "utility", "regret", "regret_ci", "gain", "gain_ci", "hsr", "hsr_ci", "random_hsr",
            "hsr_delta", "hsr_delta_ci", "human_best", "human_best_ci", "random_regret", "near_best", "bottom_half"]
    out = {k: r[k] for k in keys}
    for a in AXES:
        out[a + "_delta"] = r[a + "_delta"]; out[a + "_delta_ci"] = r[a + "_delta_ci"]
        out[a + "_comparison_n"] = r[a + "_comparison_n"]
    return out


def main():
    groups = load_jsonl(ROOT / "data/derived/main.jsonl")
    df = pd.read_csv(RES / "prompt_metrics.csv")
    cache = clip_scores(groups)
    clip_choice = {g["prompt_id"]: select(g, cache["scores"]) for g in groups}
    clip_df = policy_rows(df, groups, clip_choice, "CLIP")
    full = pd.concat([df[df.policy.isin(["Random", "Qwen", "Smol", "Oracle"])], clip_df], ignore_index=True)
    summ = {r["policy"]: r for r in summarize(full, len(groups))}
    out = {"note": "Exploratory reference selector added after main inference. CLIP scores each candidate "
                   "independently, so it is position-invariant by construction (no order effects, no flips).",
           "model": cache["model"], "revision": cache["revision"], "score": cache["score"],
           "tie_rule": "argmax cosine similarity; ties broken by canonical candidate id order",
           "policies": {p: brief(summ[p]) for p in ["Random", "CLIP", "Qwen", "Smol", "Oracle"]}}

    # Agreement between CLIP and each judge's original-order choice.
    q = df[df.policy == "Qwen"].set_index("prompt_id")
    agree = pd.Series({pid: float(q.loc[pid, "selected_id"] == clip_choice[pid]) for pid in q.index})
    chance = float(np.mean([1 / len(g["candidates"]) for g in groups if g["prompt_id"] in q.index]))
    out["agreement"] = {}
    for name in ["Qwen", "Smol"]:
        s = df[df.policy == name].set_index("prompt_id")
        a = np.array([float(s.loc[p, "selected_id"] == clip_choice[p]) for p in s.index])
        ch = float(np.mean([1 / len(g["candidates"]) for g in groups if g["prompt_id"] in s.index]))
        out["agreement"][name] = {"n": int(len(a)), "rate": float(a.mean()), "ci": bootstrap_mean(a),
                                  "chance_rate": ch}
    # Qwen gain split by agreement with CLIP (exploratory; conditioning on a post-hoc event).
    split = {}
    for label, flag in [("agree", 1.0), ("disagree", 0.0)]:
        ids = agree.index[agree == flag]
        sub = q.loc[ids]
        csub = clip_df.set_index("prompt_id").loc[ids]
        split[label] = {"n": int(len(sub)), "qwen_gain": float(sub.gain.mean()), "qwen_gain_ci": bootstrap_mean(sub.gain),
                        "qwen_regret": float(sub.regret.mean()), "qwen_regret_ci": bootstrap_mean(sub.regret),
                        "qwen_hsr": float(sub.hsr.mean()), "random_hsr": float(sub.random_hsr.mean()),
                        "clip_gain": float(csub.gain.mean()), "clip_gain_ci": bootstrap_mean(csub.gain)}
    out["qwen_by_clip_agreement"] = split
    # Paired CLIP minus Qwen difference on the same prompts.
    cq = clip_df.set_index("prompt_id").loc[q.index]
    d = (cq.gain - q.gain).to_numpy()
    out["clip_minus_qwen_gain"] = {"n": int(len(d)), "mean": float(d.mean()), "ci": bootstrap_mean(d)}
    h = (cq.hsr - q.hsr).to_numpy()
    out["clip_minus_qwen_hsr"] = {"n": int(len(h)), "mean": float(h.mean()), "ci": bootstrap_mean(h)}
    save_json(RES / "baselines.json", out)

    rows = []
    for p in ["Random", "CLIP", "Qwen", "Smol", "Oracle"]:
        r = summ[p]
        rows.append([p, str(r["n"]), f3(r["regret"]) + " " + ci(r["regret_ci"]), f3(r["gain"]) + " " + ci(r["gain_ci"]),
                     pc(r["hsr"]), pc(r["random_hsr"]), pc(r["human_best"]),
                     f3(r["stereotype_delta"]) + " " + ci(r["stereotype_delta_ci"]) if p != "Random" else "--"])
    t1 = tab("lrllrrrl", r"Selector & $n$ & Regret [95\% CI] & Gain [95\% CI] & BMR & Rand. BMR & Best & $\Delta$Stereotype [95\% CI]", rows, r"\scriptsize")
    rows = []
    for p in ["CLIP", "Qwen"]:
        r = summ[p]
        rows.append([p] + [f3(r[a + "_delta"]) + " " + ci(r[a + "_delta_ci"]) for a in ["stereotype", "missing_explicit", "missing_implicit"]])
    t2 = tab("llll", r"Selector & $\Delta$Stereotype & $\Delta$Miss. explicit & $\Delta$Miss. implicit", rows)
    ag = out["agreement"]
    rows = [[f"Qwen agrees with CLIP", str(split["agree"]["n"]), f3(split["agree"]["qwen_gain"]) + " " + ci(split["agree"]["qwen_gain_ci"]),
             f3(split["agree"]["qwen_regret"])],
            [f"Qwen disagrees with CLIP", str(split["disagree"]["n"]), f3(split["disagree"]["qwen_gain"]) + " " + ci(split["disagree"]["qwen_gain_ci"]),
             f3(split["disagree"]["qwen_regret"])]]
    t3 = tab("lrlr", r"Prompts & $n$ & Qwen gain [95\% CI] & Qwen regret", rows)
    note = (f"% CLIP {MODEL_ID}@{cache['revision']}; agreement Qwen {pc(ag['Qwen']['rate'])}\\%, "
            f"Smol {pc(ag['Smol']['rate'])}\\%, chance {pc(ag['Qwen']['chance_rate'])}\\%; "
            f"paired CLIP$-$Qwen gain {f3(out['clip_minus_qwen_gain']['mean'])} {ci(out['clip_minus_qwen_gain']['ci'])}.\n")
    (ROOT / "paper/generated/appendix_baselines.tex").write_text(
        note + t1 + "\n\n\\vspace{4pt}\n" + t2 + "\n\n\\vspace{4pt}\n" + t3 + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
