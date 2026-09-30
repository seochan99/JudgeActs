# Protocol decisions before main inference

- Dataset and model revisions are pinned in `sources.json`.
- All code in this repository is newly written. The only copied local files are the provided AAAI Press 2027 style and bibliography style.
- Before any main inference, validation discovered two repeated prompt texts and four pairs of duplicate encoded-image bytes. The lexicographically first base identifier is retained per repeated country/prompt pair. The initial manifests are preserved in `initial_manifests`; corrected manifests are frozen in git.
- The first development output instruction contained a concrete JSON example with choice A and alphabetic ranking. The model repeatedly copied that ranking, sometimes disagreeing with its own choice. The example was removed for parser/schema debugging before main inference. The initial development outputs are retained locally in `runs/dev_v1_example` and are excluded from all empirical tables.
- Primary output validation is strict and never repairs a choice or ranking. Invalid output causes abstention. Runtime exceptions permit one identical retry.
- No main-run result is used to change a prompt, endpoint, threshold, or split.
- Qwen's final development interface completed 90/90 calls with strict schema validity before the primary main run started.
- Smol development showed inability to satisfy full rankings and inconsistent JSON field names. Its secondary interface requests a choice only and normalizes an explicit single label (bare or in one recognized one-field structured object). JSON validity, exact-schema validity, and recovered-decision validity are reported separately. No explanatory prose or ambiguous choice is recovered.
- Smol's default preprocessor upsamples images to 1536 pixels and creates 17 views per square image, imposing excessive memory use when all candidates are supplied. The secondary adapter instead uses one native 384-pixel view per image, documented as a model-specific visual-budget limitation. These decisions were made on the dev split before secondary main inference; Qwen's primary main interface is unchanged.
- OpenReview approval and IASEAI portal connection were confirmed directly by the user during preparation. No author-list or track choice is inferred from that confirmation.
