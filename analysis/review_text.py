"""Manuscript fragments answering reviewer questions; all values read from results/main/review_extras.json."""
import json

from src.common import ROOT
from .extra_text import v, p, ci

RES = ROOT / "results/main"
GEN = ROOT / "paper/generated"


def main():
    path = RES / "review_extras.json"
    if not path.exists():
        return
    r = json.loads(path.read_text())
    cs = r["clip_sensitivity"]
    b32 = cs["policies"]["CLIP ViT-B/32"]
    clip_text = (
        "The CLIP baseline embeds the raw prompt (no template; all prompts fit the 77-token limit) and each image with the "
        "default CLIP preprocessing (resize and center crop to 224 pixels), and returns the image with the highest cosine "
        "similarity; ties follow canonical order. Our main baseline uses ViT-L/14. A smaller ViT-B/32 backbone gives nearly "
        f"the same gain ({v(b32['gain'])} {ci(b32['gain_ci'])}; difference {v(cs['b32_minus_l14_gain']['mean'])} "
        f"{ci(cs['b32_minus_l14_gain']['ci'])}), but no stereotype increase ({v(b32['stereotype_delta'])} "
        f"{ci(b32['stereotype_delta_ci'])}), so the stereotype effect of a similarity scorer depends on the backbone "
        "(Appendix Table~\\ref{tab:extras-clip})."
    )
    sd = r["stereotype_decomposition"]
    g = sd["by_chosen_generator"]["SD-3.5-Large"]
    prof = r["cultural_profiles"]
    maj, un, rest = prof["Qwen majority"], prof["Qwen unanimous"], prof["Qwen, not unanimous"]
    ties = sd["by_alignment_ties"]
    stereo_text = (
        "Where does the stereotype increase come from? Most of it traces to one generator: Qwen returns SD-3.5-Large images "
        f"in {p(g['chosen_share'])} of prompts against {p(g['uniform_share'])} under uniform choice, and these images carry the "
        f"highest stereotype ratings (Appendix Table~\\ref{{tab:extras-stereo}}). The increase is also larger when several "
        f"candidates tie for the best alignment ({v(ties['best_tied']['qwen_delta'])} {ci(ties['best_tied']['qwen_delta_ci'])}) "
        f"than when one candidate is clearly best ({v(ties['unique_best']['qwen_delta'])} "
        f"{ci(ties['unique_best']['qwen_delta_ci'])}). Aggregating across orders leaves no detectable increase: the stereotype difference is "
        f"{v(maj['stereotype_delta'])} {ci(maj['stereotype_delta_ci'])} for voting over the three orders and "
        f"{v(un['stereotype_delta'])} {ci(un['stereotype_delta_ci'])} on unanimous prompts, whereas on the prompts unanimity "
        f"rejects it is {v(rest['stereotype_delta'])} {ci(rest['stereotype_delta_ci'])}. Both aggregated policies still reduce "
        f"missing explicit expectations ({v(maj['missing_explicit_delta'])} and {v(un['missing_explicit_delta'])})."
    )
    lat = r["latency"]["qwen"]["per_prompt"]
    rows = {(x["protocol"], x["threshold"]): x for x in r["risk_coverage"]["rows"]}
    four = [x for x in r["risk_coverage"]["rows"] if x["protocol"].startswith("4")]
    f_top = max(four, key=lambda x: x["q"]) if four else None
    cost_text = (
        "Order checks are cheap relative to their value. On our hardware a single Qwen decision takes a median of "
        f"{lat['single_order_median_s']:.1f}~s and three orders {lat['three_order_median_s']:.1f}~s, roughly three times the "
        "cost, which buys either the voting gain or the unanimity filter. Adding a fourth order lowers coverage without "
        "lowering regret further"
        + (f" (four-order unanimity keeps {p(f_top['coverage'])} of four-image pools at regret {v(f_top['regret'])})" if f_top else "")
        + " (Appendix Figure~\\ref{fig:risk-coverage})."
    )
    d = r["difficulty"]["by_range_tertile"]
    diff_text = (
        "Order sensitivity is only weakly related to how close the candidates are: Qwen's choice changes across orders on "
        f"{p(d['low']['flip_rate'])} of prompts in the lowest third of rating range and {p(d['high']['flip_rate'])} in the "
        "highest (Appendix Table~\\ref{tab:extras-difficulty}), so the bias is not confined to near-ties."
    )
    for name, text in [("review_clip", clip_text), ("review_stereo", stereo_text),
                       ("review_cost", cost_text), ("review_difficulty", diff_text)]:
        (GEN / f"{name}.tex").write_text(text + "\n")


if __name__ == "__main__":
    main()
