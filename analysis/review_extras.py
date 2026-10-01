"""POST-HOC reviewer extras -> results/main/review_extras.json, paper/generated/appendix_extras_*.tex,
paper/figures/risk_coverage.{pdf,png}.

Exploratory analyses added after main inference. Every number is computed from frozen run files,
human annotations, and recorded CLIP scores; nothing is hand-entered. Frozen inputs are read only.

  1. CLIP sensitivity: second CLIP backbone (ViT-B/32) next to the recorded ViT-L/14 selector.
  2. Cost/latency of single-order vs three-order (order-voting) protocols from recorded wall-clock seconds.
  3. Pool difficulty vs Qwen order sensitivity (flip across the 3 orders).
  4. Stereotype decomposition of Qwen's choices (generator, alignment ties, generator x slot).
  5. Cultural profiles of the Qwen majority and Qwen unanimous policies.
  6. Risk-coverage of agreement thresholds (3 orders on 300 prompts; 4 orders on 200 four-image pools).
"""
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from src.common import ROOT, load_jsonl, save_json
from .analyze import summarize
from .appendix_tables import tab, f3, ci, pc, normalize
from .content_baselines import policy_rows, select
from .metrics import bootstrap_mean, candidate_metrics, choice_consistency

RES = ROOT / "results/main"
GEN = ROOT / "paper/generated"
FIG = ROOT / "paper/figures"
L14 = "openai/clip-vit-large-patch14"
B32 = "openai/clip-vit-base-patch32"
B32_CACHE = RES / "clip_scores_b32.json"
CULT = ["stereotype", "missing_explicit", "missing_implicit"]
EPS = 1e-12


def load_run(path):
    rows = load_jsonl(path)
    run = {(r["prompt_id"], r["permutation"]): r for r in rows}
    assert len(run) == len(rows), f"duplicate keys in {path}"
    return run


# ---------------------------------------------------------------- 1. CLIP sensitivity
def clip_config(model_id, revision):
    """Preprocessing/scoring details read from the processor objects themselves (config files only)."""
    from transformers import CLIPProcessor
    proc = CLIPProcessor.from_pretrained(model_id, revision=revision)
    ip = proc.image_processor
    tok = proc.tokenizer
    return {"model": model_id, "revision": revision,
            "image_processor": type(ip).__name__,
            "resize_shortest_edge": int(dict(ip.size)["shortest_edge"]),
            "resample": __import__("PIL.Image").Image.Resampling(int(ip.resample)).name,
            "center_crop": {k: int(v) for k, v in dict(ip.crop_size).items() if v is not None}, "do_center_crop": bool(ip.do_center_crop),
            "rescale_factor": float(ip.rescale_factor), "image_mean": list(ip.image_mean),
            "image_std": list(ip.image_std), "convert_rgb": bool(getattr(ip, "do_convert_rgb", True)),
            "tokenizer": type(tok).__name__, "text_max_length": int(tok.model_max_length),
            "text_input": "raw prompt string (no template), padding=True, truncation=True, max_length=77",
            "score": "cosine similarity of the projected, L2-normalised text_embeds and image_embeds "
                     "(CLIPModel outputs; logit scale not applied)",
            "selection": "argmax over candidates; ties broken by canonical (sorted) candidate id",
            "device": "cpu", "precision": "float32"}, tok


def clip_b32_scores(groups):
    if B32_CACHE.exists():
        cached = json.loads(B32_CACHE.read_text())
        if {c["id"] for g in groups for c in g["candidates"]} <= set(cached["scores"]):
            return cached
    import torch
    from PIL import Image
    from huggingface_hub import model_info
    from transformers import CLIPModel, CLIPProcessor
    revision = model_info(B32).sha
    # CPU and few threads on purpose: a concurrent job holds the accelerator.
    torch.set_num_threads(2)
    model = CLIPModel.from_pretrained(B32, revision=revision).eval()
    proc = CLIPProcessor.from_pretrained(B32, revision=revision)
    scores = {}
    with torch.no_grad():
        for g in groups:
            cs = g["candidates"]
            ims = [Image.open(ROOT / c["image"]).convert("RGB") for c in cs]
            inp = proc(text=[g["prompt"]], images=ims, return_tensors="pt", padding=True, truncation=True, max_length=77)
            o = model(**inp)
            for c, v in zip(cs, (o.image_embeds @ o.text_embeds.T).squeeze(-1).tolist()):
                scores[c["id"]] = float(v)
    out = {"model": B32, "revision": revision, "score": "cosine(text_embed, image_embed)", "device": "cpu",
           "precision": "float32", "text_truncation": 77, "torch": torch.__version__,
           "computed_utc": datetime.now(timezone.utc).isoformat(), "scores": scores}
    save_json(B32_CACHE, out)
    return out


def clip_sensitivity(groups, df):
    l14 = json.loads((RES / "clip_scores.json").read_text())
    b32 = clip_b32_scores(groups)
    choices = {name: {g["prompt_id"]: select(g, c["scores"]) for g in groups}
               for name, c in [("CLIP ViT-L/14", l14), ("CLIP ViT-B/32", b32)]}
    parts = [df[df.policy.isin(["Random", "Qwen", "Oracle"])]]
    parts += [policy_rows(df, groups, ch, name) for name, ch in choices.items()]
    full = pd.concat(parts, ignore_index=True)
    summ = {r["policy"]: r for r in summarize(full, len(groups))}
    keys = ["n", "regret", "regret_ci", "gain", "gain_ci", "hsr", "hsr_ci", "random_hsr", "human_best",
            "stereotype_delta", "stereotype_delta_ci", "missing_explicit_delta", "missing_explicit_delta_ci",
            "missing_implicit_delta", "missing_implicit_delta_ci"]
    pol = {p: {k: summ[p][k] for k in keys} for p in ["Random", "CLIP ViT-L/14", "CLIP ViT-B/32", "Qwen", "Oracle"]}
    ids = [g["prompt_id"] for g in groups]
    a, b = choices["CLIP ViT-L/14"], choices["CLIP ViT-B/32"]
    same = np.array([float(a[p] == b[p]) for p in ids])
    rows = {n: full[full.policy == n].set_index("prompt_id").loc[ids] for n in ["CLIP ViT-L/14", "CLIP ViT-B/32"]}
    d_gain = (rows["CLIP ViT-B/32"].gain - rows["CLIP ViT-L/14"].gain).to_numpy()
    d_st = (rows["CLIP ViT-B/32"].stereotype - rows["CLIP ViT-L/14"].stereotype).to_numpy()
    # Score-level agreement across backbones: within-prompt rank correlation of candidate scores.
    rho = [spearmanr([l14["scores"][c["id"]] for c in g["candidates"]],
                     [b32["scores"][c["id"]] for c in g["candidates"]]).statistic for g in groups]
    rho = np.array([r for r in rho if np.isfinite(r)])
    cfg_l, tok = clip_config(L14, l14["revision"])
    cfg_b, _ = clip_config(B32, b32["revision"])
    n_trunc = sum(len(tok(g["prompt"])["input_ids"]) > 77 for g in groups)
    cfg_text = (f"Both CLIP selectors ({L14}@{l14['revision']}; {B32}@{b32['revision']}) score each candidate "
                f"independently with the Hugging Face CLIPProcessor defaults: RGB conversion, resize of the shortest "
                f"edge to {cfg_b['resize_shortest_edge']} px (bicubic), center crop to {cfg_b['center_crop']['height']}x"
                f"{cfg_b['center_crop']['width']}, rescale by 1/255 and normalisation with the CLIP mean/std. "
                f"The text input is the raw prompt string without a template, tokenised with padding and truncation "
                f"at 77 tokens ({n_trunc} of {len(groups)} prompts exceed 77 tokens). The score is the cosine "
                f"similarity of the projected, L2-normalised text and image embeddings (logit scale not applied); "
                f"the selector takes the argmax per prompt, breaking ties by canonical candidate id. "
                f"Inference ran on CPU in float32.")
    return {"policies": pol,
            "backbone_choice_agreement": {"n": len(ids), "rate": float(same.mean()), "ci": bootstrap_mean(same)},
            "within_prompt_score_spearman": {"n": int(len(rho)), "mean": float(rho.mean()), "median": float(np.median(rho))},
            "b32_minus_l14_gain": {"mean": float(d_gain.mean()), "ci": bootstrap_mean(d_gain)},
            "b32_minus_l14_stereotype": {"mean": float(np.nanmean(d_st)), "ci": bootstrap_mean(d_st[~np.isnan(d_st)])},
            "prompts_over_77_tokens": int(n_trunc),
            "config": {"ViT-L/14": cfg_l, "ViT-B/32": cfg_b}, "config_text": cfg_text}


# ---------------------------------------------------------------- 2. Cost / latency
def _concurrent_flags(rows, others):
    """A call is concurrent if its wall-clock interval overlaps any call of the other run."""
    iv = lambda r: (datetime.fromisoformat(r["utc"]).timestamp() - r["seconds"], datetime.fromisoformat(r["utc"]).timestamp())
    other = sorted(iv(r) for r in others if "utc" in r)
    starts = np.array([s for s, _ in other]); ends = np.array([e for _, e in other])
    flags = []
    for r in rows:
        s, e = iv(r)
        flags.append(bool(np.any((starts < e) & (ends > s))))
    return np.array(flags)


def latency(groups, runs):
    out = {}
    for m, run in runs.items():
        rows = list(run.values())
        sec = np.array([r["seconds"] for r in rows], dtype=float)
        other = list(runs["smol" if m == "qwen" else "qwen"].values())
        conc = _concurrent_flags(rows, other)
        per_prompt = []
        for g in groups:
            rs = [run.get((g["prompt_id"], k)) for k in range(3)]
            if all(rs):
                per_prompt.append({"single": rs[0]["seconds"], "three": sum(r["seconds"] for r in rs),
                                   "size": len(g["candidates"])})
        pp = pd.DataFrame(per_prompt)
        by_size = {}
        for size in [3, 4]:
            s = np.array([r["seconds"] for r in rows if len(r["mapping"]) == size])
            by_size[str(size)] = {"calls": int(len(s)), "median": float(np.median(s)), "mean": float(s.mean())}
        solo = sec[~conc]
        out[m] = {"calls": int(len(sec)), "median_s": float(np.median(sec)), "mean_s": float(sec.mean()),
                  "p90_s": float(np.quantile(sec, .9)), "total_h": float(sec.sum() / 3600),
                  "by_pool_size": by_size,
                  "concurrent_calls": int(conc.sum()),
                  "solo_calls": int(len(solo)),
                  "solo_median_s": float(np.median(solo)) if len(solo) else None,
                  "solo_mean_s": float(solo.mean()) if len(solo) else None,
                  "per_prompt": {"n": int(len(pp)),
                                 "single_order_median_s": float(pp.single.median()), "single_order_mean_s": float(pp.single.mean()),
                                 "three_order_median_s": float(pp.three.median()), "three_order_mean_s": float(pp.three.mean()),
                                 "three_over_single_mean_ratio": float(pp.three.mean() / pp.single.mean()),
                                 "three_over_single_median_ratio": float(pp.three.median() / pp.single.median()),
                                 "note": "single order = permutation 0 only; it carries the session warm-up outliers, "
                                         "so the mean ratio is below 3 while the median ratio is ~3",
                                 "extra_seconds_per_prompt_mean": float((pp.three - pp.single).mean()),
                                 "single_order_total_h": float(pp.single.sum() / 3600),
                                 "three_order_total_h": float(pp.three.sum() / 3600)}}
    out["hardware"] = (f"Apple Silicon, PyTorch MPS backend (transformers-mps, bfloat16, greedy decoding, max 64 new tokens); "
                       f"analysis host: {platform.machine()} {platform.platform()}, "
                       f"{_chip()}. Seconds are recorded wall-clock per call including image preprocessing; "
                       f"calls whose interval overlaps a call of the other judge ran concurrently on the same accelerator "
                       f"and are slower (see provenance/decisions.md).")
    return out


def _chip():
    import subprocess
    try:
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
        return f"{brand}, {mem / 2**30:.0f} GB unified memory"
    except Exception:
        return "chip unknown"


# ---------------------------------------------------------------- 3. Difficulty vs order sensitivity
def _logit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    nll = lambda w: np.sum(np.logaddexp(0, X1 @ w) - y * (X1 @ w))
    grad = lambda w: X1.T @ (1 / (1 + np.exp(-(X1 @ w))) - y)
    return minimize(nll, np.zeros(X1.shape[1]), jac=grad, method="BFGS").x


def difficulty(groups, qrun, seed=20261002):
    recs = []
    for g in groups:
        rs = [qrun.get((g["prompt_id"], k)) for k in range(3)]
        if not all(r and r["parse_ok"] for r in rs):
            continue
        u = np.array([c["utility"] for c in g["candidates"]])
        recs.append({"prompt_id": g["prompt_id"], "flip": float(len({r["selected_id"] for r in rs}) > 1),
                     "range": float(np.ptp(u)), "near_ties": int(np.sum(u >= u.max() - .05 - EPS)),
                     "best_ties": int(np.sum(u >= u.max() - EPS)), "size": len(u)})
    d = pd.DataFrame(recs)
    q1, q2 = np.quantile(d.range, [1 / 3, 2 / 3])
    # Tertile cut points from the data; tied values stay together (in the lower bin).
    d["tertile"] = np.where(d.range <= q1 + EPS, "low", np.where(d.range <= q2 + EPS, "mid", "high"))
    tert = {}
    for t in ["low", "mid", "high"]:
        s = d[d.tertile == t]
        tert[t] = {"n": int(len(s)), "range_min": float(s.range.min()), "range_max": float(s.range.max()),
                   "mean_range": float(s.range.mean()), "flip_rate": float(s.flip.mean()), "flip_ci": bootstrap_mean(s.flip)}
    by_ties = {}
    for k, s in d.groupby("near_ties"):
        by_ties[str(k)] = {"n": int(len(s)), "flip_rate": float(s.flip.mean()), "flip_ci": bootstrap_mean(s.flip)}
    by_size = {str(k): {"n": int(len(s)), "flip_rate": float(s.flip.mean()), "flip_ci": bootstrap_mean(s.flip)}
               for k, s in d.groupby("size")}
    corr = {}
    for v in ["range", "near_ties", "size"]:
        r = spearmanr(d[v], d.flip)
        corr[v] = {"spearman_rho": float(r.statistic), "p_value": float(r.pvalue)}
    # Multivariable logistic regression on standardised predictors; prompt-bootstrap CIs (2000 resamples).
    cols = ["range", "near_ties", "size"]
    X = d[cols].to_numpy(float); mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd; y = d.flip.to_numpy()
    w = _logit(Z, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(2000):
        ix = rng.integers(len(y), size=len(y))
        boots.append(_logit(Z[ix], y[ix]))
    boots = np.array(boots)
    lo, hi = np.quantile(boots, [.025, .975], axis=0)
    logit = {"predictors": "standardised (per SD)", "sd": dict(zip(cols, sd.tolist())),
             "coef": {n: {"log_odds_per_sd": float(w[i]), "ci": [float(lo[i]), float(hi[i])],
                          "odds_ratio_per_sd": float(np.exp(w[i]))}
                      for i, n in enumerate(["intercept"] + cols)},
             "bootstrap": {"resamples": 2000, "seed": seed, "unit": "prompt"}}
    return {"n": int(len(d)), "flip_rate": float(d.flip.mean()), "tertile_cuts": [float(q1), float(q2)],
            "by_range_tertile": tert, "by_near_ties": by_ties, "by_pool_size": by_size,
            "spearman": corr, "logistic": logit,
            "definitions": {"flip": "Qwen choice differs across the three presentation orders",
                            "range": "best minus worst human alignment utility in the pool",
                            "near_ties": "candidates with utility >= best - 0.05 (includes the best)"}}


# ---------------------------------------------------------------- 4. Stereotype decomposition
def stereotype_decomposition(groups, df, qrun):
    q = df[df.policy == "Qwen"].copy()
    gen = {c["id"]: c["source_model"] for g in groups for c in g["candidates"]}
    q["generator"] = q.selected_id.map(gen)
    q["d_st"] = q.stereotype - q.random_stereotype
    total = q.d_st.dropna()
    overall = {"n": int(len(total)), "delta": float(total.mean()), "ci": bootstrap_mean(total)}
    cands = [(c["source_model"], c["stereotype"], c["utility"], len(g["candidates"])) for g in groups for c in g["candidates"]]
    gens = sorted({c[0] for c in cands})
    per_gen = {}
    for m in gens:
        s = q[q.generator == m].d_st.dropna()
        st = np.array([c[1] for c in cands if c[0] == m], float)
        per_gen[m] = {"chosen_n": int((q.generator == m).sum()), "chosen_share": float((q.generator == m).mean()),
                      "uniform_share": float(sum(1 / c[3] for c in cands if c[0] == m) / len(groups)),
                      "mean_stereotype_all_images": float(st.mean()), "mean_stereotype_ci": bootstrap_mean(st),
                      "mean_utility_all_images": float(np.mean([c[2] for c in cands if c[0] == m])),
                      "delta_when_chosen": float(s.mean()) if len(s) else None,
                      "delta_when_chosen_ci": bootstrap_mean(s),
                      # Contribution to the overall mean delta (sums over generators to the overall delta).
                      "contribution_to_overall_delta": float(s.sum() / len(total))}
    # Within-pool generator effect: each generator's stereotype minus its pool mean, averaged over pools.
    for m in gens:
        rel = [c["stereotype"] - np.mean([x["stereotype"] for x in g["candidates"]])
               for g in groups for c in g["candidates"] if c["source_model"] == m]
        per_gen[m]["within_pool_stereotype_vs_pool_mean"] = float(np.mean(rel))
        per_gen[m]["within_pool_ci"] = bootstrap_mean(rel)
    # Alignment ties: does the alignment-best set contain more than one candidate?
    ties = {}
    nb = {g["prompt_id"]: int(sum(c["utility"] >= max(x["utility"] for x in g["candidates"]) - EPS for c in g["candidates"]))
          for g in groups}
    q["best_tied"] = q.prompt_id.map(lambda p: nb[p] > 1)
    o = df[df.policy == "Oracle"].set_index("prompt_id")
    for label, flag in [("best_tied", True), ("unique_best", False)]:
        s = q[q.best_tied == flag]
        dd = s.d_st.dropna()
        od = (o.loc[s.prompt_id].stereotype - o.loc[s.prompt_id].random_stereotype).dropna()
        ties[label] = {"n": int(len(s)), "qwen_delta": float(dd.mean()), "qwen_delta_ci": bootstrap_mean(dd),
                       "oracle_delta": float(od.mean()), "oracle_delta_ci": bootstrap_mean(od),
                       "qwen_regret": float(s.regret.mean())}
    # Generator x slot: P(chosen | generator shown in slot), pooled over the three orders.
    shown, chosen = Counter(), Counter()
    for r in qrun.values():
        if not r["parse_ok"]:
            continue
        for k, (lab, cid) in enumerate(r["mapping"].items()):
            shown[(gen[cid], "ABCD"[k])] += 1
            chosen[(gen[cid], "ABCD"[k])] += cid == r["selected_id"]
    slot = {m: {s: {"shown": shown[(m, s)], "chosen": chosen[(m, s)],
                    "rate": chosen[(m, s)] / shown[(m, s)] if shown[(m, s)] else None} for s in "ABCD"} for m in gens}
    for m in gens:
        tot_s = sum(shown[(m, s)] for s in "ABCD"); tot_c = sum(chosen[(m, s)] for s in "ABCD")
        slot[m]["all"] = {"shown": tot_s, "chosen": tot_c, "rate": tot_c / tot_s}
    return {"overall": overall, "by_chosen_generator": per_gen, "by_alignment_ties": ties,
            "generator_by_slot_choice_rate": slot,
            "note": "Qwen original-order choice; delta = chosen stereotype minus exact random (pool mean) "
                    "on the same prompt. Slot table pools all valid calls over the three orders."}


# ---------------------------------------------------------------- 5. Majority vs unanimity
def cultural_profiles(df, total):
    summ = {r["policy"]: r for r in summarize(df[df.policy.isin(["Qwen", "Qwen majority", "Qwen unanimous", "Qwen, not unanimous"])], total)}
    out = {}
    for p in ["Qwen", "Qwen majority", "Qwen unanimous", "Qwen, not unanimous"]:
        r = summ[p]
        out[p] = {"n": r["n"], "coverage": r["coverage"], "regret": r["regret"], "regret_ci": r["regret_ci"],
                  **{a + k: r[a + k] for a in CULT for k in ["_delta", "_delta_ci", "_comparison_n",
                                                             "_delta_ci_bonf3"]}}
    return out


# ---------------------------------------------------------------- 6. Risk-coverage
def _threshold_rows(groups, run, perms, label):
    n = len(groups)
    per = []
    for g in groups:
        rs = [run.get((g["prompt_id"], k)) for k in perms]
        if not all(r and r["parse_ok"] for r in rs):
            continue
        ids = [c["id"] for c in g["candidates"]]; u = [c["utility"] for c in g["candidates"]]
        cons = choice_consistency([r["selected_id"] for r in rs], ids)
        m = candidate_metrics(u, ids.index(cons["majority_id"]))
        per.append({"agreement": cons["agreement"], "regret": m["regret"], "rand": float(max(u) - np.mean(u))})
    d = pd.DataFrame(per)
    k = len(perms)
    rows = []
    for j in range(1, k + 1):
        s = d[d.agreement >= j / k - EPS]
        diff = (s.regret - s.rand).to_numpy()
        rows.append({"protocol": label, "threshold": f"{j}/{k}", "q": j / k, "n": int(len(s)), "pool_n": n,
                     "coverage": len(s) / n, "regret": float(s.regret.mean()), "regret_ci": bootstrap_mean(s.regret),
                     "random_regret": float(s.rand.mean()), "random_regret_ci": bootstrap_mean(s.rand),
                     "difference": float(diff.mean()), "difference_ci": bootstrap_mean(diff)})
    return rows


def risk_coverage(groups, qrun):
    rot = load_run(ROOT / "runs/ablations/rotation4.jsonl")
    four = [g for g in groups if len(g["candidates"]) == 4]
    combined = {**{k: v for k, v in qrun.items() if k[1] in (0, 1, 2)}, **rot}
    full = [g for g in four if all((g["prompt_id"], k) in combined for k in range(4))]
    rows = (_threshold_rows(groups, qrun, [0, 1, 2], "3 orders, all prompts")
            + _threshold_rows(full, combined, [0, 1, 2], "3 orders, 4-image pools")
            + _threshold_rows(full, combined, [0, 1, 2, 3], "4 orders, 4-image pools"))
    return {"rows": rows, "four_image_pools": len(four), "pools_with_4_orders": len(full),
            "rule": "accept if the plurality choice's share of orders >= q; plurality ties broken by canonical "
                    "candidate order; matched random = exact expected random regret on the accepted prompts"}


def risk_figure(rc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    from . import style
    style.apply()
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    fig.subplots_adjust(left=.155, right=.975, bottom=.18, top=.84)
    specs = [("3 orders, all prompts", "o", "-"), ("4 orders, 4-image pools", "s", "--")]
    for proto, mk, ls in specs:
        rs = [r for r in rc["rows"] if r["protocol"] == proto]
        x = [r["coverage"] for r in rs]
        for key, col, z in [("random_regret", style.RANDOM, 2), ("regret", style.QWEN, 3)]:
            y = [r[key] for r in rs]
            lo = [y[i] - rs[i][key + "_ci"][0] for i in range(len(rs))]
            hi = [rs[i][key + "_ci"][1] - y[i] for i in range(len(rs))]
            ax.errorbar(x, y, yerr=[lo, hi], color=col, marker=mk, ms=3.6, lw=1.0, ls=ls, elinewidth=.6,
                        capsize=0, mfc=col if ls == "-" else "white", mec=col, mew=.9, zorder=z)
        for r in rs:
            ax.annotate(r["threshold"], (r["coverage"], r["regret"]), xytext=(-4, -8 if ls == "-" else 4),
                        textcoords="offset points", ha="right", fontsize=6.2, color=style.MUTED)
    ax.set_xlim(0, 1.04)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Coverage (share of prompts automated)")
    ax.set_ylabel("Selective regret")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=style.GRID, lw=.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=style.QWEN, lw=1.2, label="Qwen plurality"),
               Line2D([], [], color=style.RANDOM, lw=1.2, label="Random, same prompts"),
               Line2D([], [], color=style.INK, marker="o", ms=3.4, lw=1, ls="-", label="3 orders, 300 prompts"),
               Line2D([], [], color=style.INK, marker="s", ms=3.4, lw=1, ls="--", mfc="white",
                      label="4 orders, 200 pools")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.13, 1.0), ncol=2, frameon=False,
               fontsize=6.6, handlelength=2.0, columnspacing=1.0, labelspacing=.3)
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIG / f"risk_coverage.{suffix}", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- tables
def tables(out):
    t = {}
    c = out["clip_sensitivity"]
    rows = []
    for p in ["Random", "CLIP ViT-L/14", "CLIP ViT-B/32", "Qwen", "Oracle"]:
        r = c["policies"][p]
        rows.append([p, str(r["n"]), f3(r["regret"]) + " " + ci(r["regret_ci"]), f3(r["gain"]) + " " + ci(r["gain_ci"]),
                     pc(r["hsr"]), pc(r["random_hsr"]),
                     f3(r["stereotype_delta"]) + " " + ci(r["stereotype_delta_ci"]) if p != "Random" else "--"])
    a = c["backbone_choice_agreement"]
    t["clip"] = (f"% CLIP backbones agree on {pc(a['rate'])}\\% of prompts; B/32$-$L/14 gain "
                 f"{f3(c['b32_minus_l14_gain']['mean'])} {ci(c['b32_minus_l14_gain']['ci'])}.\n"
                 + tab("lrllrrl", r"Selector & $n$ & Regret [95\% CI] & Gain [95\% CI] & BMR & Rand. BMR & $\Delta$Stereotype [95\% CI]", rows))

    rows = []
    for m in ["qwen", "smol"]:
        r = out["latency"][m]; pp = r["per_prompt"]
        rows.append([m.capitalize(), str(r["calls"]), f"{r['median_s']:.1f}", f"{r['mean_s']:.1f}",
                     f"{pp['single_order_mean_s']:.1f}", f"{pp['three_order_mean_s']:.1f}",
                     f"{pp['three_over_single_mean_ratio']:.2f}", f"{pp['single_order_total_h']:.2f}",
                     f"{pp['three_order_total_h']:.2f}"])
    t["latency"] = tab("lrrrrrrrr", r"Judge & Calls & Med. s/call & Mean s/call & 1-order s/prompt & 3-order s/prompt & Ratio & 1-order h & 3-order h", rows)

    dfc = out["difficulty"]
    rows = [[t_.capitalize(), f"{r['range_min']:.2f}--{r['range_max']:.2f}", str(r["n"]), pc(r["flip_rate"]),
             "[" + ", ".join(pc(v) for v in r["flip_ci"]) + "]"] for t_, r in dfc["by_range_tertile"].items()]
    rows += [[f"{k} near-tie" + ("s" if k != "1" else ""), "--", str(r["n"]), pc(r["flip_rate"]),
              "[" + ", ".join(pc(v) for v in r["flip_ci"]) + "]"] for k, r in dfc["by_near_ties"].items()]
    lg = dfc["logistic"]["coef"]
    t["difficulty"] = ("% logistic log-odds per SD: " + "; ".join(f"{k} {f3(v['log_odds_per_sd'])} {ci(v['ci'])}"
                                                                  for k, v in lg.items() if k != "intercept") + "\n"
                       + tab("lrrrl", r"Stratum & Range & $n$ & Flip (\%) & 95\% CI", rows))

    sd = out["stereotype_decomposition"]
    rows = []
    for m, r in sd["by_chosen_generator"].items():
        rows.append([m, pc(r["uniform_share"]), pc(r["chosen_share"]), f3(r["mean_stereotype_all_images"]),
                     f3(r["within_pool_stereotype_vs_pool_mean"]),
                     f3(r["delta_when_chosen"]) + " " + ci(r["delta_when_chosen_ci"]), f3(r["contribution_to_overall_delta"])])
    rows.append(["All", "100.0", "100.0", "--", "--", f3(sd["overall"]["delta"]) + " " + ci(sd["overall"]["ci"]),
                 f3(sd["overall"]["delta"])])
    t["stereo_generator"] = tab("lrrrrlr", r"Chosen generator & Uniform (\%) & Qwen (\%) & Stereo. & Within-pool & $\Delta$Stereo. when chosen [95\% CI] & Contribution", rows)
    rows = []
    for lab, name in [("best_tied", "Alignment-best tied"), ("unique_best", "Unique alignment-best")]:
        r = sd["by_alignment_ties"][lab]
        rows.append([name, str(r["n"]), f3(r["qwen_delta"]) + " " + ci(r["qwen_delta_ci"]),
                     f3(r["oracle_delta"]) + " " + ci(r["oracle_delta_ci"]), f3(r["qwen_regret"])])
    t["stereo_ties"] = tab("lrllr", r"Prompts & $n$ & Qwen $\Delta$Stereo. [95\% CI] & Oracle $\Delta$Stereo. [95\% CI] & Qwen regret", rows)
    rows = []
    for m, r in sd["generator_by_slot_choice_rate"].items():
        rows.append([m] + [pc(r[s]["rate"]) + f" ({r[s]['shown']})" if r[s]["shown"] else "--" for s in "ABCD"]
                    + [pc(r["all"]["rate"])])
    t["gen_slot"] = tab("lrrrrr", r"Generator & A (\%) & B (\%) & C (\%) & D (\%) & All (\%)", rows)

    rows = []
    for p, r in out["cultural_profiles"].items():
        rows.append([p, str(r["n"])] + [f3(r[a + "_delta"]) + " " + ci(r[a + "_delta_ci"]) for a in CULT])
    t["profiles"] = tab("lrlll", r"Policy & $n$ & $\Delta$Stereotype & $\Delta$Miss. explicit & $\Delta$Miss. implicit", rows)

    rows = []
    for r in out["risk_coverage"]["rows"]:
        rows.append([r["protocol"], r["threshold"], str(r["n"]), pc(r["coverage"]),
                     f3(r["regret"]) + " " + ci(r["regret_ci"]), f3(r["random_regret"]) + " " + ci(r["random_regret_ci"]),
                     f3(r["difference"]) + " " + ci(r["difference_ci"])])
    t["risk_coverage"] = tab("lrrrlll", r"Protocol & $q\ge$ & $n$ & Cov. (\%) & Regret [95\% CI] & Rand. regret [95\% CI] & Difference [95\% CI]", rows, r"\scriptsize")
    for name, content in t.items():
        (GEN / f"appendix_extras_{name}.tex").write_text(normalize(content + "\n"))


def main():
    groups = load_jsonl(ROOT / "data/derived/main.jsonl")
    df = pd.read_csv(RES / "prompt_metrics.csv")
    runs = {m: load_run(ROOT / f"runs/{m}_main.jsonl") for m in ["qwen", "smol"]}
    out = {"post_hoc": True, "bootstrap": {"resamples": 10000, "seed": 20261002, "unit": "prompt"},
           "judge": "Qwen/Qwen3-VL-4B-Instruct main runs unless stated",
           "clip_sensitivity": clip_sensitivity(groups, df),
           "latency": latency(groups, runs),
           "difficulty": difficulty(groups, runs["qwen"]),
           "stereotype_decomposition": stereotype_decomposition(groups, df, runs["qwen"]),
           "cultural_profiles": cultural_profiles(df, len(groups)),
           "risk_coverage": risk_coverage(groups, runs["qwen"])}
    save_json(RES / "review_extras.json", out)
    tables(out)
    risk_figure(out["risk_coverage"])
    print(json.dumps({k: v for k, v in out.items() if k != "clip_sensitivity"}, indent=1, default=str)[:20000])
    print(json.dumps({k: v for k, v in out["clip_sensitivity"].items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
