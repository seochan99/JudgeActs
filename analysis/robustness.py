"""Reviewer robustness checks (exploratory, added after main inference).

(a) Annotation consensus: per-prompt annotator disagreement = mean over candidates (with >=2 ratings)
    of the population variance of individual alignment ratings; median split into high/low consensus.
(b) Prompts whose human-best candidate is unique (no utility tie at the top).
(c) Presentation-slot preference and order flips by prompt category.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from src.common import ROOT, load_jsonl, save_json
from .analyze import summarize
from .appendix_tables import tab, f3, ci, pc, nice
from .metrics import wilson, bootstrap_mean

RES = ROOT / "results/main"
POLICIES = ["Qwen", "Smol", "Qwen unanimous"]


def disagreement(g):
    v = [float(np.var(c["alignment_ratings"])) for c in g["candidates"] if len(c["alignment_ratings"]) >= 2]
    return float(np.mean(v)) if v else np.nan


def unique_best(g):
    u = [c["utility"] for c in g["candidates"]]
    return sum(x >= max(u) - 1e-12 for x in u) == 1


def brief(r):
    keys = ["n", "regret", "regret_ci", "gain", "gain_ci", "hsr", "hsr_ci", "random_hsr", "hsr_delta", "hsr_delta_ci",
            "human_best", "random_regret", "stereotype_delta", "stereotype_delta_ci", "missing_explicit_delta",
            "missing_explicit_delta_ci", "missing_implicit_delta", "missing_implicit_delta_ci"]
    return {k: r[k] for k in keys}


def subset(df, pids, total):
    sub = df[df.prompt_id.isin(pids)]
    return {r["policy"]: brief(r) for r in summarize(sub, total) if r["policy"] in ["Random"] + POLICIES}


def main():
    groups = load_jsonl(ROOT / "data/derived/main.jsonl")
    df = pd.read_csv(RES / "prompt_metrics.csv")
    dis = {g["prompt_id"]: disagreement(g) for g in groups}
    med = float(np.nanmedian(list(dis.values())))
    # High consensus = disagreement strictly below the median; ties at the median go to low consensus.
    high = [p for p, d in dis.items() if d < med]
    low = [p for p, d in dis.items() if d >= med]
    uniq = [g["prompt_id"] for g in groups if unique_best(g)]
    out = {"note": "Exploratory robustness checks added after main inference.",
           "disagreement_definition": "mean over candidates with >=2 ratings of population variance of alignment_ratings",
           "disagreement_median": med,
           "disagreement_mean": float(np.nanmean(list(dis.values()))),
           "consensus_split": {"high_consensus": {"n_prompts": len(high), **subset(df, high, len(high))},
                               "low_consensus": {"n_prompts": len(low), **subset(df, low, len(low))}},
           "unique_best": {"n_prompts": len(uniq), "n_tied_best": len(groups) - len(uniq), **subset(df, uniq, len(uniq))}}
    # Rank correlation between disagreement and Qwen gain (descriptive).
    from scipy.stats import spearmanr
    q = df[df.policy == "Qwen"].set_index("prompt_id")
    rho = spearmanr([dis[p] for p in q.index], q.gain)
    out["spearman_disagreement_vs_qwen_gain"] = {"rho": float(rho.statistic), "p_value": float(rho.pvalue)}

    # Slot preference and flips by category.
    cat = {g["prompt_id"]: g["category"] for g in groups}
    order = pd.read_csv(RES / "order_metrics.csv")
    slots = {}
    for m in ["qwen", "smol"]:
        rows = [r for r in load_jsonl(ROOT / f"runs/{m}_main.jsonl") if r["parse_ok"]]
        by = defaultdict(list)
        for r in rows:
            by[cat[r["prompt_id"]]].append(r)
        by["all"] = rows
        res = {}
        for c, rs in sorted(by.items()):
            a = sum(next(k for k, v in r["mapping"].items() if v == r["selected_id"]) == "A" for r in rs)
            fl = order[(order.model == m) & ((order.category == c) | (c == "all"))]["flip"].to_numpy()
            res[c] = {"calls": len(rs), "slot_a_count": int(a), "slot_a_share": a / len(rs),
                      "slot_a_ci": wilson(a, len(rs)),
                      "uniform_share": float(np.mean([1 / len(r["mapping"]) for r in rs])),
                      "flip_n": int(len(fl)), "flip_rate": float(fl.mean()), "flip_ci": bootstrap_mean(fl)}
        slots[m] = res
    out["slot_by_category"] = slots
    save_json(RES / "robustness.json", out)

    # LaTeX.
    rows = []
    for label, block in [("High consensus", out["consensus_split"]["high_consensus"]),
                         ("Low consensus", out["consensus_split"]["low_consensus"]),
                         ("Unique best", out["unique_best"])]:
        for p in ["Qwen", "Qwen unanimous"]:
            r = block.get(p)
            if not r:
                continue
            rows.append([label, p, str(r["n"]), f3(r["random_regret"]), f3(r["regret"]) + " " + ci(r["regret_ci"]),
                         f3(r["gain"]) + " " + ci(r["gain_ci"]), pc(r["hsr"]), pc(r["random_hsr"]),
                         f3(r["stereotype_delta"]) + " " + ci(r["stereotype_delta_ci"])])
    t1 = tab("llrrllrrl", r"Subset & Policy & $n$ & Rand. regret & Regret [95\% CI] & Gain [95\% CI] & BMR & Rand. BMR & $\Delta$Stereotype [95\% CI]", rows, r"\scriptsize")
    rows = []
    for c in sorted(slots["qwen"], key=lambda k: (k == "all", k)):
        qq, ss = slots["qwen"][c], slots["smol"].get(c)
        rows.append([nice(c) if c != "all" else "All", str(qq["calls"]), pc(qq["slot_a_share"]), pc(qq["uniform_share"]),
                     pc(qq["flip_rate"]), pc(ss["slot_a_share"]), pc(ss["uniform_share"]), pc(ss["flip_rate"])])
    t2 = tab("lrrrrrrr", r"Category & Qwen calls & Qwen A (\%) & Unif. (\%) & Qwen flip (\%) & Smol A (\%) & Unif. (\%) & Smol flip (\%)", rows)
    note = (f"% Disagreement median {med:.4f}; high consensus n={len(high)}, low n={len(low)}; unique best n={len(uniq)}.\n")
    (ROOT / "paper/generated/appendix_robustness.tex").write_text(note + t1 + "\n\n\\vspace{4pt}\n" + t2 + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
