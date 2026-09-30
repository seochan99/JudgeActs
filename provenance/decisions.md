# Protocol decisions before main inference

- Dataset and model revisions are pinned in `sources.json`.
- All code in this repository is newly written. The only copied local files are the provided AAAI Press 2027 style and bibliography style.
- Before any main inference, validation discovered two repeated prompt texts and four pairs of duplicate encoded-image bytes. The lexicographically first base identifier is retained per repeated country/prompt pair. The initial manifests are preserved in `initial_manifests`; corrected manifests are frozen in git.
- The first development output instruction contained a concrete JSON example with choice A and alphabetic ranking. The model repeatedly copied that ranking, sometimes disagreeing with its own choice. The example was removed for parser/schema debugging before main inference. The initial development outputs are retained locally in `runs/dev_v1_example` and are excluded from all empirical tables.
- Primary output validation is strict and never repairs a choice or ranking. Invalid output causes abstention. Runtime exceptions permit one identical retry.
- No main-run result is used to change a prompt, endpoint, threshold, or split.
