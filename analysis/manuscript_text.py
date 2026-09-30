"""Manuscript prose generated only from complete, recorded main analyses."""


def val(x):
    return "NA" if x is None else f"{x:.3f}"


def pct(x):
    return f"{100*x:.1f}\\%"


def ci(row, key, percent=False):
    formatter = pct if percent else val
    return "[" + ", ".join(formatter(x) for x in row[key + "_ci"]) + "]"


def label(text):
    return text.replace("_", " ").replace("&", r"\&")


def pending():
    return {
        "status": r"\begin{quote}\textbf{Working draft: main experiments incomplete.} This PDF is not submission-ready.\end{quote}",
        "abstract": "A vision-language model that selects a generated image decides which candidate a user would receive. We specify an audit of this choice using CulturalFrames and its released human ratings. For each prompt, two open-weight judges select from three or four existing images without access to generator names or annotations. The primary outcome is prompt-alignment regret relative to the highest-rated candidate; exact random selection provides a paired baseline. Separate analyses examine missing expectations, stereotypes, country groups, and abstention based on agreement across candidate orders or judges. The frozen main split contains 300 prompts, with three orders per prompt. Main-run findings remain pending.",
        "results": "The main experiments are still running. Results, tables, and empirical figures will be populated only after both judges complete all frozen requests.",
        "discussion_results": "% Main findings pending.",
        "conclusion_result": "% Main findings pending.",
        "category_results": "Category estimates will be reported after both main runs finish.",
        "table_main": r"\begin{tabular}{lccccccc}\toprule Policy & $n$ & Align. & Regret & BMR (\%) & Best (\%) & Worst & Cov. (\%)\\\midrule Main run pending & -- & -- & -- & -- & -- & -- & --\\\bottomrule\end{tabular}",
        "table_decomposition": r"\begin{tabular}{lcccc}\toprule Condition & $n$ & Regret & Selected & Random\\\midrule Main run pending & -- & -- & -- & --\\\bottomrule\end{tabular}",
    }


def completed(summary, country, decomposition, order, status, categories, position, baselines=None, ablations=None):
    by = {r["policy"]: r for r in summary}
    q, oracle = by["Qwen"], by["Oracle"]
    un, majority, cross = (by.get(p) for p in ["Qwen unanimous", "Qwen majority", "Cross-model"])
    rest, split = by.get("Qwen, not unanimous"), by.get("Qwen, judges disagree")
    s = by["Smol"]
    oc = order[order.model == "qwen"]
    flip = float(oc["flip"].mean())
    qa, sa = position["qwen"]["slots"]["A"], position["smol"]["slots"]
    s_front = sa["A"]["rate"] + sa["B"]["rate"]
    clip = baselines["policies"]["CLIP"] if baselines else None
    dclip = baselines["clip_minus_qwen_gain"] if baselines else None

    ratio = qa["rate"] / qa["uniform_rate"]
    abstract = (
        "Vision-language models (VLMs) are increasingly used as judges that pick the best of several generated images, "
        "so their choices decide what users see. Such judges are usually validated by how well their scores agree with "
        "human ratings, not by the quality of the images they actually choose. We audit VLM judges as decision-makers. "
        "On 300 culturally situated prompts, we compare the image a judge returns with human ratings it never sees and with "
        "random choice from the same candidates, and we repeat every decision with the candidates reordered. "
        "A 4B-parameter judge improves only marginally on random selection"
        + (" and falls short of a simple CLIP similarity baseline" if clip and clip["gain"] > q["gain"] else "")
        + f". Its choices show strong position bias: it picks the first image shown {ratio:.1f} times as often as chance, "
        f"and reordering the same candidates changes its choice on {round(100 * flip)}\\% of prompts. "
        "Self-consistency is nonetheless informative: decisions that survive reordering are much better than random, "
        "whereas agreement with a second, weaker judge keeps the wrong decisions. Both the judge and the CLIP baseline "
        "reduce missing cultural details but slightly increase stereotyped content. We argue that judges should be "
        "evaluated by what they choose, with order sensitivity reported and cultural errors measured separately from "
        "prompt alignment."
    )

    direct = [
        f"Table~\\ref{{tab:main}} and Figure~\\ref{{fig:results}}a summarize the direct policies. Qwen returns a valid "
        f"original-order choice for all {q['n']} prompts. Its mean alignment is {val(q['utility'])}, against "
        f"{val(q['random_utility'])} for random choice and {val(oracle['oracle_utility'])} for the oracle, a gain of "
        f"{val(q['gain'])} (95\\% CI {ci(q, 'gain')}). Mean regret is {val(q['regret'])} {ci(q, 'regret')}; Qwen returns a "
        f"best-rated image for {pct(q['human_best'])} of prompts and a below-mean image for {pct(q['hsr'])}, against "
        f"{pct(q['random_hsr'])} for random choice (paired difference {val(q['hsr_delta'])}, 95\\% CI {ci(q, 'hsr_delta')}).",
        f"Smol does worse than random: gain {val(s['gain'])} {ci(s, 'gain')}, regret {val(s['regret'])}, and BMR "
        f"{pct(s['hsr'])}. It rarely follows the output format (Appendix Table~\\ref{{tab:validity}}), and its interface and "
        "image resolution differ from Qwen's, so we treat it as a weak secondary judge rather than as a capability comparison. "
        "Appendix Table~\\ref{tab:sensitivity} shows that the conclusions hold with a below-mean margin of 0.05, "
        "with the median in place of the mean, and with overall satisfaction as the utility.",
    ]

    position_text = [
        f"Both judges favor early slots (Figure~\\ref{{fig:results}}c). Qwen picks slot A in {pct(qa['rate'])} of its "
        f"{position['qwen']['calls']} calls, where uniform choice given the pool sizes would give {pct(qa['uniform_rate'])} "
        f"($\\chi^2_{{{position['qwen']['df']}}}={position['qwen']['chi2']:.0f}$, $p<10^{{-10}}$); Smol places {pct(s_front)} of "
        f"its choices in slots A or B. Consistent with this, Qwen's choice changes across the three orders for {pct(flip)} of prompts. "
        "Figure~\\ref{fig:hero} shows the extreme case: the judge returns whatever image is shown first.",
    ]
    qg, ug = position["qwen_generators"], position["_generator_uniform"]
    top = max(qg, key=qg.get)
    position_text.append(
        "The judge still responds to content. Canonical order places one generator first in most pools, so we pool all "
        f"three rotations, which spreads each generator across slots: Qwen then chooses images from the generator with the "
        f"highest mean human rating in {pct(qg[top])} of calls, against {pct(ug[top])} under uniform choice "
        "(Appendix Table~\\ref{tab:generators}). Its choices mix a content signal with a strong slot preference."
    )

    gates = []
    if un and rest:
        gates.append(
            "A judge that follows position alone picks a different image in each rotation, so it can never be unanimous. "
            f"Unanimity keeps {un['n']} prompts ({pct(un['coverage'])}). On them Qwen's gain over random is {val(un['gain'])} "
            f"{ci(un, 'gain')} and regret {val(un['regret'])}; on the {rest['n']} rejected prompts its gain is "
            f"{val(rest['gain'])} {ci(rest, 'gain')}, below random (Figure~\\ref{{fig:results}}b). The kept prompts are not "
            f"easier: their candidates differ more in rating than the rejected ones (mean best-minus-worst range "
            f"{val(un['score_range'])} versus {val(rest['score_range'])}), leaving more room for error. The gate only filters "
            "prompts; on kept prompts the choice is unchanged."
        )
    if majority:
        gates.append(
            f"Voting over the three orders instead of abstaining, a simple form of order averaging, raises the gain on all "
            f"prompts from {val(q['gain'])} to {val(majority['gain'])} {ci(majority, 'gain')}."
        )
    if cross and split:
        gates.append(
            f"Agreement between judges behaves differently. Qwen and Smol agree on {cross['n']} prompts "
            f"({pct(cross['coverage'])}), where Qwen's gain is {val(cross['gain'])} {ci(cross, 'gain')}; on the {split['n']} "
            f"prompts where they disagree it is {val(split['gain'])} {ci(split, 'gain')}. Two judges that both prefer early "
            "slots can agree for reasons unrelated to content, so this gate keeps the wrong prompts. The result is specific to "
            "pairing with a weak judge. Appendix Table~\\ref{tab:policies} lists every gate with its complement."
        )

    conditions = [r for r in decomposition if r["policy"] == "Qwen"]
    axes = {"Explicit": "missing_explicit", "Implicit": "missing_implicit", "Stereotype": "stereotype"}
    cultural = [
        "Figure~\\ref{fig:cultural} compares the cultural-error ratings of Qwen's choices with random choice on the same pools. "
        f"Qwen lowers missing explicit expectations by {val(-q['missing_explicit_delta'])} {ci(q, 'missing_explicit_delta')} and "
        f"missing implicit expectations by {val(-q['missing_implicit_delta'])} {ci(q, 'missing_implicit_delta')}, and both "
        "reductions survive a Bonferroni adjustment across the three error types. It raises the stereotype rating from "
        f"{val(q['matched_random_stereotype'])} to {val(q['matched_stereotype'])} (difference {val(q['stereotype_delta'])}, "
        f"pointwise 95\\% CI {ci(q, 'stereotype_delta')}); this interval includes zero after adjustment "
        f"([{val(q['stereotype_delta_ci_bonf3'][0])}, {val(q['stereotype_delta_ci_bonf3'][1])}]), so we treat it as "
        "exploratory. On the pools where stereotype ratings differ across candidates, the difference is "
        + next((f"{val(r['stereotype_delta'])} {ci(r, 'stereotype_delta')}" for r in conditions if r['condition'] == 'Stereotype'), "--")
        + " (Appendix Table~\\ref{tab:decomposition}). An alignment gain therefore need not carry over to every cultural outcome."
    ]

    qc = [r for r in country if r["policy"] == "Qwen"]
    worst = max(qc, key=lambda r: r["regret"])
    best_c = max(qc, key=lambda r: r["gain"])
    low_c = min(qc, key=lambda r: r["gain"])
    n_pos = sum(r["gain"] > 0 for r in qc)
    country_text = [
        f"Qwen's point estimate of gain is positive in {n_pos} of {len(qc)} country groups (Appendix Figure~\\ref{{fig:country}}), "
        f"ranging from {val(best_c['gain'])} in {label(best_c['country'])} to {val(low_c['gain'])} in {label(low_c['country'])}. "
        f"Regret is highest in {label(worst['country'])} ({val(worst['regret'])}, pointwise 95\\% CI {ci(worst, 'regret')}), "
        f"where random choice also has the highest regret ({val(worst['random_regret'])}). With 30 prompts per group, "
        "these intervals are wide and overlap; country labels describe dataset contexts, not cultural preferences, and "
        "the analysis is not powered to rank groups."
    ]

    table = [r"\begin{tabular}{lrrcllrrc}", r"\toprule",
             r"Policy & $n$ & Cov.\,(\%) & Align. & Regret [95\% CI] & Gain over random [95\% CI] & BMR\,(\%) & Best\,(\%) & Worst\\",
             r"\midrule"]
    rows = [("Random", "Random"), ("Qwen", "Qwen"), ("Qwen majority", "Qwen, majority of 3 orders"),
            ("Qwen unanimous", "Qwen, unanimous gate"), ("Smol", "Smol"), ("Cross-model", "Cross-model gate"),
            ("CLIP", "CLIP similarity"), ("Oracle", "Oracle")]
    for key, name in rows:
        r = by.get(key) or (baselines["policies"].get(key) if baselines else None)
        if not r:
            continue
        cov = r.get("coverage", r["n"] / 300)
        cells = [str(r["n"]), f"{100 * cov:.1f}", val(r.get("utility", r.get("random_utility", 0) + r["gain"])),
                 f"{val(r['regret'])} {ci(r, 'regret')}", f"{val(r['gain'])} {ci(r, 'gain')}",
                 f"{100 * r['hsr']:.1f}", f"{100 * r['human_best']:.1f}",
                 val(r["worst_group_regret"]) if "worst_group_regret" in r else "--"]
        table.append(name + " & " + " & ".join(cells) + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    decomposition_table = [r"\begin{tabular}{lrrrrl}", r"\toprule",
                           r"Condition & $n$ & Regret & Selected & Random & Difference [95\% CI]\\", r"\midrule"]
    for r in conditions:
        axis = axes[r["condition"]]
        cells = [str(r["n"]), val(r["regret"]), val(r["matched_" + axis]), val(r["matched_random_" + axis]),
                 f"{val(r[axis + '_delta'])} {ci(r, axis + '_delta')}"]
        decomposition_table.append(r["condition"] + " & " + " & ".join(cells) + r"\\")
    decomposition_table += [r"\bottomrule", r"\end{tabular}"]
    return {
        "status": "% Both frozen main runs are complete; this is not an upload confirmation.",
        "abstract": abstract,
        "results_direct": "\n\n".join(direct),
        "results_position": "\n\n".join(position_text),
        "results_gates": "\n\n".join(gates),
        "results_cultural": "\n\n".join(cultural),
        "results_country": "\n\n".join(country_text),
        "conclusion_result": (
            f"A small VLM judge that chooses which image to return improves alignment over random choice only slightly "
            f"({val(q['gain'])}), and less than a CLIP similarity scorer. " if clip else
            f"A small VLM judge improves alignment over random choice only slightly ({val(q['gain'])}). "
        ) + (
            f"Its choices depend strongly on presentation order: it picks the first-listed image in {pct(qa['rate'])} of calls. "
            "Agreement across orders separates prompts where the judge adds value from prompts where it falls below "
            "chance, whereas agreement with a weaker judge does not. Neither selector avoids stereotyped images."
        ),
        "table_main": "\n".join(table),
        "table_decomposition": "\n".join(decomposition_table),
    }
