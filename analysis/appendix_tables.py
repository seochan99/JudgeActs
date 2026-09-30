"""Appendix tables generated from recorded artifacts; no hand-entered values."""
import json
from collections import Counter, defaultdict

import numpy as np

from src.common import ROOT, load_jsonl
from .metrics import bootstrap_mean

RES = ROOT / "results/main"
GEN = ROOT / "paper/generated"


def f3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{0.0 if abs(x) < 5e-4 else x:.3f}".replace("-", "$-$")


def ci(pair):
    return "[" + ", ".join(f3(v) for v in pair) + "]"


def pc(x):
    return f"{100 * x:.1f}"


def tab(spec, header, rows, size=r"\footnotesize"):
    lines = [size, r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{" + spec + "}", r"\toprule",
             header + r"\\", r"\midrule"]
    lines += [" & ".join(r) + r"\\" for r in rows]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def nice(name):
    return name.replace("_", " ").replace("-", " ").replace("&", r"\&").capitalize()


def main():
    summary = {r["policy"]: r for r in json.loads((RES / "summary.json").read_text())}
    country = json.loads((RES / "country.json").read_text())
    category = json.loads((RES / "category.json").read_text())
    status = json.loads((RES / "status.json").read_text())
    groups = load_jsonl(ROOT / "data/derived/main.jsonl")
    runs = {m: load_jsonl(ROOT / f"runs/{m}_main.jsonl") for m in ["qwen", "smol"]}
    out = {}

    # Dataset composition per country.
    rows = []
    for c in sorted({g["country"] for g in groups}):
        gs = [g for g in groups if g["country"] == c]
        cands = [x for g in gs for x in g["candidates"]]
        ties = sum(np.ptp([x["utility"] for x in g["candidates"]]) <= 1e-12 for g in gs)
        rows.append([c.replace("_", " "), str(len(gs)), str(sum(len(g["candidates"]) == 3 for g in gs)),
                     str(sum(len(g["candidates"]) == 4 for g in gs)), str(ties),
                     f3(float(np.mean([x["utility"] for x in cands]))),
                     f3(float(np.mean([np.ptp([x["utility"] for x in g["candidates"]]) for g in gs]))),
                     f3(float(np.mean([x["stereotype"] for x in cands if x["stereotype"] is not None])))])
    out["appendix_dataset"] = tab("lrrrrrrr", r"Country & Prompts & 3-img & 4-img & Ties & Mean align. & Mean range & Stereo.", rows)

    # Generator availability, human ratings, and choice share across all orders.
    util = defaultdict(list)
    avail = Counter()
    for g in groups:
        for x in g["candidates"]:
            util[x["source_model"]].append(x["utility"])
            avail[x["source_model"]] += 1 / len(g["candidates"])
    lookup = {x["id"]: x["source_model"] for g in groups for x in g["candidates"]}
    share = {}
    for m, rs in runs.items():
        ok = [r for r in rs if r["parse_ok"]]
        cnt = Counter(lookup[r["selected_id"]] for r in ok)
        share[m] = {k: v / len(ok) for k, v in cnt.items()}
    rows = [[k, str(len(v)), f3(float(np.mean(v))), pc(avail[k] / len(groups)),
             pc(share["qwen"].get(k, 0)), pc(share["smol"].get(k, 0))] for k, v in sorted(util.items())]
    out["appendix_generators"] = tab("lrrrrr", r"Generator & Images & Mean align. & Uniform (\%) & Qwen (\%) & Smol (\%)", rows)

    # Presentation slot by pool size.
    rows = []
    for m, rs in runs.items():
        for size in [3, 4]:
            ok = [r for r in rs if r["parse_ok"] and len(r["mapping"]) == size]
            cnt = Counter(next(k for k, v in r["mapping"].items() if v == r["selected_id"]) for r in ok)
            cells = [pc(cnt.get(s, 0) / len(ok)) if s in "ABCD"[:size] else "--" for s in "ABCD"]
            rows.append([m.capitalize(), str(size), str(len(ok))] + cells + [pc(1 / size)])
    out["appendix_position"] = tab("lrrrrrrr", r"Judge & Pool & Calls & A (\%) & B (\%) & C (\%) & D (\%) & Uniform (\%)", rows)

    # Every policy, including gate complements.
    order = ["Random", "Qwen", "Qwen majority", "Qwen agree 2/3", "Qwen unanimous", "Qwen, not unanimous",
             "Smol", "Smol majority", "Smol agree 2/3", "Smol unanimous", "Cross-model", "Qwen, judges disagree", "Oracle"]
    rows = []
    for p in order:
        r = summary.get(p)
        if not r:
            continue
        rows.append([p, str(r["n"]), pc(r["coverage"]), f3(r["regret"]) + " " + ci(r["regret_ci"]),
                     f3(r["gain"]) + " " + ci(r["gain_ci"]), pc(r["hsr"]), pc(r["random_hsr"]), pc(r["human_best"])])
    out["appendix_policies"] = tab("lrrllrrr", r"Policy & $n$ & Cov. & Regret [95\% CI] & Gain [95\% CI] & BMR & Rand. BMR & Best", rows)

    # Secondary outcomes as paired differences from matched random selection.
    rows = []
    for p in ["Qwen", "Smol", "Qwen unanimous", "Cross-model"]:
        r = summary[p]
        cells = [f3(r[a + "_delta"]) + " " + ci(r[a + "_delta_ci"]) for a in
                 ["stereotype", "missing_explicit", "missing_implicit", "image_quality"]]
        rows.append([p] + cells + [f3(r["overall_gain"]) + " " + ci(r["overall_gain_ci"])])
    q = summary["Qwen"]
    rows.append([r"Qwen, Bonf.", ci(q["stereotype_delta_ci_bonf3"]), ci(q["missing_explicit_delta_ci_bonf3"]),
                 ci(q["missing_implicit_delta_ci_bonf3"]), "--", "--"])
    out["appendix_outcomes"] = tab("lccccc", r"Policy & $\Delta$Stereotype & $\Delta$Miss. explicit & $\Delta$Miss. implicit & $\Delta$Quality & Overall gain", rows, r"\scriptsize")

    # Per-country table.
    idx = {(r["country"], r["policy"]): r for r in country}
    rows = []
    for c in sorted({r["country"] for r in country}):
        qr, sr = idx.get((c, "Qwen")), idx.get((c, "Smol"))
        un = idx.get((c, "Qwen unanimous"))
        rows.append([c.replace("_", " "), f3(qr["random_regret"]), f3(qr["regret"]),
                     f3(qr["gain"]) + " " + ci(qr["gain_ci"]), pc(qr["hsr"]),
                     f3(sr["regret"]) if sr else "--", f3(sr["gain"]) if sr else "--",
                     (str(un["n"]) + " / " + f3(un["regret"])) if un else "0 / --"])
    out["appendix_country"] = tab("lrrlrrrr", r"Country & Rand. & Qwen & Qwen gain [95\% CI] & BMR & Smol & Smol gain & 3/3: $n$ / regret", rows, r"\scriptsize")

    # Category table.
    cidx = {(r["category"], r["policy"]): r for r in category}
    rows = []
    for c in sorted({r["category"] for r in category}):
        qr = cidx[(c, "Qwen")]
        rows.append([nice(c), str(qr["n"]), f3(qr["random_regret"]), f3(qr["regret"]) + " " + ci(qr["regret_ci"]),
                     f3(qr["gain"]) + " " + ci(qr["gain_ci"])])
    out["appendix_category"] = tab("lrrll", r"Category & $n$ & Rand. regret & Qwen regret [95\% CI] & Qwen gain [95\% CI]", rows)

    # Sensitivity of the direct policies to alternative definitions.
    rows = []
    for p in ["Random", "Qwen", "Smol", "Qwen unanimous"]:
        r = summary[p]
        rows.append([p, f3(r["regret_median"]), pc(r["hsr05"]), pc(r["random_hsr05"]), pc(r["near_best"]),
                     pc(r["bottom_half"]), f3(r["overall_regret"])])
    out["appendix_sensitivity"] = tab("lrrrrrr", r"Policy & Med. regret & BMR$_{.05}$ & Rand. BMR$_{.05}$ & Near-best & Below median & Overall regret", rows)

    # Output validity.
    rows = []
    for m, v in status["validation"].items():
        rows.append([m.capitalize(), str(v["calls"]), str(v["valid"]), str(v["parse_failed"]), str(v["runtime_failed"]),
                     str(v["strict_schema_valid"]), str(v["recovered"]), str(v["json_invalid"]), f"{v['seconds'] / 3600:.2f}"])
    out["appendix_validity"] = tab("lrrrrrrrr", r"Judge & Calls & Valid & Parse fail & Runtime fail & Exact schema & Recovered & Non-JSON & Hours", rows)

    for name, content in out.items():
        (GEN / f"{name}.tex").write_text(content + "\n")
    normalize_all()


if __name__ == "__main__":
    main()


def normalize(tex):
    """Uniform appendix tables: full text width, spread columns, one font size."""
    import re
    if "tabular*" in tex:
        return tex
    tex = re.sub(r"\\(scriptsize|footnotesize|small)\b", "", tex)
    tex = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", tex)
    def size(block):
        rows = [r for r in block.split("\\\\") if "&" in r]
        return "\\scriptsize" if max((len(r) for r in rows), default=0) > 150 else "\\footnotesize"
    parts = re.split(r"(\\begin\{tabular\}\{[^}]*\}.*?\\end\{tabular\})", tex, flags=re.S)
    for k, part in enumerate(parts):
        m = re.match(r"\\begin\{tabular\}\{([^}]*)\}", part)
        if m:
            parts[k] = (size(part) + "\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}" + m.group(1) + "}"
                        + part[m.end():])
    tex = "".join(parts)
    tex = tex.replace("\\end{tabular}", "\\end{tabular*}")
    return re.sub(r"\n{3,}", "\n\n", tex)


def normalize_all():
    for path in sorted(GEN.glob("appendix_*.tex")) + [GEN / "table_decomposition.tex"]:
        if path.exists():
            path.write_text(normalize(path.read_text()))
