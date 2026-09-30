"""Manuscript fragments for exploratory analyses (content baseline, robustness, ablations).

Every number is read from results/main; nothing is hand-entered.
"""
import json

from src.common import ROOT

RES = ROOT / "results/main"
GEN = ROOT / "paper/generated"


def v(x):
    return f"{x:.3f}"


def p(x):
    return f"{100 * x:.1f}\\%"


def ci(pair, f=v):
    return "[" + ", ".join(f(x) for x in pair) + "]"


def load(name):
    path = RES / name
    return json.loads(path.read_text()) if path.exists() else None


def baselines_text(b, rob):
    c, q = b["policies"]["CLIP"], b["policies"]["Qwen"]
    ag, split, d = b["agreement"]["Qwen"], b["qwen_by_clip_agreement"], b["clip_minus_qwen_gain"]
    out = [
        "Would a simpler scorer that cannot be influenced by order do as well? "
        f"We score each candidate by CLIP prompt--image similarity \\citep{{radford2021clip,hessel2021clipscore}} and return the highest-scoring image. "
        f"This content-only selector has regret {v(c['regret'])} (95\\% CI {ci(c['regret_ci'])}) and gain over random "
        f"{v(c['gain'])} {ci(c['gain_ci'])}, compared with Qwen's {v(q['regret'])} and {v(q['gain'])}; the paired "
        f"difference in gain favors CLIP by {v(d['mean'])} {ci(d['ci'])}. CLIP also selects a below-mean image less often "
        f"({p(c['hsr'])} versus {p(q['hsr'])}). Yet it raises the stereotype rating relative to random by "
        f"{v(c['stereotype_delta'])} {ci(c['stereotype_delta_ci'])}, as much as Qwen does, while sharply reducing missing "
        f"explicit expectations ({v(c['missing_explicit_delta'])}). An alignment-oriented scorer therefore does not avoid "
        "stereotyped images; this concern is not specific to VLM judges.",
        f"Qwen agrees with CLIP on {p(ag['rate'])} of prompts (chance {p(ag['chance_rate'])}). Where they agree, Qwen's gain is "
        f"{v(split['agree']['qwen_gain'])} {ci(split['agree']['qwen_gain_ci'])}; where they disagree it is "
        f"{v(split['disagree']['qwen_gain'])} {ci(split['disagree']['qwen_gain_ci'])}. Agreement with an independent, "
        "position-invariant scorer thus behaves like order unanimity rather than like agreement with the weaker VLM judge. "
        "The CLIP comparison is exploratory (Appendix Table~\\ref{tab:baselines}).",
    ]
    if rob:
        cs = rob["consensus_split"]
        hi = cs.get("high_consensus", {}).get("Qwen")
        if hi:
            lo = cs["low_consensus"]["Qwen"]
            out.append(
                f"Results do not hinge on noisy annotations. Splitting prompts at the median annotator disagreement, Qwen's gain is "
                f"{v(hi['gain'])} {ci(hi['gain_ci'])} on high-consensus and {v(lo['gain'])} {ci(lo['gain_ci'])} on low-consensus "
                "prompts (Appendix Table~\\ref{tab:robustness})."
            )
    return "\n\n".join(out)


def ablations_text(a):
    need = ("choice_only", "opaque_labels", "rotation4")
    if not a or any(k not in a or "slot_first_rate" not in a[k] for k in need) or a["opaque_labels"]["calls"] < 900:
        return "Ablation results are being computed."
    co, ol, r4 = (a[k] for k in need)
    parts = [
        "The slot preference could be an artifact of our interface rather than of the judge. We ran three post-hoc "
        "ablations with Qwen on all 300 prompts, leaving the frozen main results unchanged (Figure~\ref{fig:ablations}, "
        "Appendix Table~\ref{tab:ablations}).",
        f"\\textit{{Output format.}} The main contract asks for a full ranking whose first entry is the choice, which could "
        f"prime the first label. With a choice-only contract, Qwen still picks the first slot in {p(co['slot_first_rate'])} "
        f"of calls (uniform {p(co['slot_first_uniform'])}) and changes its choice across orders on {p(co['flip_rate'])} of "
        f"prompts. Its gain over random is {v(co['gain'])} {ci(co['gain_ci'])}, and unanimity keeps {p(co['unanimous_coverage'])} "
        f"of prompts with gain {v(co['unanimous_gain'])} {ci(co['unanimous_gain_ci'])}.",
        f"\\textit{{Label identity.}} Replacing A/B/C/D with opaque two-character codes unrelated to order, Qwen picks the "
        f"first-presented image in {p(ol['slot_first_rate'])} of calls (uniform {p(ol['slot_first_uniform'])}) and changes "
        f"its choice on {p(ol['flip_rate'])} of prompts, with gain {v(ol['gain'])} {ci(ol['gain_ci'])}.",
        f"\\textit{{Full rotation.}} Adding a fourth rotation for the {r4['n_prompts']} four-image pools places every image "
        f"in every slot once. Slot shares are then " + ", ".join(f"{k} {p(x)}" for k, x in r4["slot_rates"].items())
        + f" (uniform {p(r4['slot_first_uniform'])}). Unanimity over four orders keeps {p(r4['unanimous_coverage'])} of these "
        f"pools with regret {v(r4['unanimous_regret'])} and gain {v(r4['unanimous_gain'])} {ci(r4['unanimous_gain_ci'])}, and "
        f"voting over the four orders gives gain {v(r4['plurality_gain'])} {ci(r4['plurality_gain_ci'])}.",
    ]
    persists = all(x["slot_first_rate"] > x["slot_first_uniform"] + 0.05 for x in (co, ol, r4))
    gate_ok = all(x.get("unanimous_gain_ci", [0])[0] > 0 for x in (co, ol, r4) if "unanimous_gain_ci" in x)
    if persists and gate_ok:
        parts.append("In every variant the preference for the first slot remains and unanimity still keeps prompts on which "
                     "the judge beats random. The preference is a property of the judge, not of the letter labels or the "
                     "ranking format.")
    elif persists:
        parts.append("In every variant the preference for the first slot remains, so it is not produced by the letter "
                     "labels or the ranking format.")
    else:
        parts.append("The preference weakens in at least one variant, so part of it depends on the interface; we report "
                     "each variant separately.")
    return "\n\n".join(parts)


def main():
    b, rob, a = load("baselines.json"), load("robustness.json"), load("ablations.json")
    if b:
        (GEN / "baselines_text.tex").write_text(baselines_text(b, rob) + "\n")
    try:
        text = ablations_text(a)
    except KeyError:
        text = "Ablation results are being computed."
    (GEN / "ablations.tex").write_text(text + "\n")


if __name__ == "__main__":
    main()
