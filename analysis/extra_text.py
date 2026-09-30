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
    if not a:
        return "Ablation results are being computed."
    parts = []
    base = a.get("main_reference", {})
    co, ol, r4 = a.get("choice_only"), a.get("opaque_labels"), a.get("rotation4")
    parts.append(
        "The slot preference could be induced by our interface rather than by the judge. We ran three post-hoc ablations "
        "with Qwen, leaving the frozen main results unchanged."
    )
    if co:
        parts.append(
            f"\\textit{{Output schema.}} The main contract asks for a full ranking whose first entry is the choice, which could "
            f"prime the first label. With a choice-only contract, Qwen still selects the first slot in {p(co['slot_first_rate'])} "
            f"of calls (uniform {p(co['slot_first_uniform'])}), its choice changes across orders for {p(co['flip_rate'])} of prompts, "
            f"and its gain over random is {v(co['gain'])} {ci(co['gain_ci'])}."
        )
    if ol:
        parts.append(
            f"\\textit{{Label identity.}} Replacing A/B/C/D with opaque two-character codes that carry no order, Qwen selects the "
            f"first-presented image in {p(ol['slot_first_rate'])} of calls (uniform {p(ol['slot_first_uniform'])}) and changes its "
            f"choice across orders for {p(ol['flip_rate'])} of prompts, with gain {v(ol['gain'])} {ci(ol['gain_ci'])}."
        )
    if r4:
        parts.append(
            f"\\textit{{Full rotation.}} Adding the fourth rotation for the {r4['n_prompts']} four-image pools places every image in "
            f"every slot once. Slot shares are then {', '.join(f'{k} {p(x)}' for k, x in r4['slot_rates'].items())} "
            f"(uniform 25.0\\%); unanimity over four orders keeps {p(r4['unanimous_coverage'])} of these pools with regret "
            f"{v(r4['unanimous_regret'])}, and the order-plurality policy has gain {v(r4['plurality_gain'])} "
            f"{ci(r4['plurality_gain_ci'])}."
        )
    parts.append(a.get("conclusion", ""))
    return "\n\n".join(x for x in parts if x)


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
