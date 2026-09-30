# Development-only visual integrity check

Ten frozen development sets were inspected, one per country (dev indices 0, 3, ..., 27). The inspection boards are stored locally in `results/dev/inspection`. The boards show every available candidate, canonical labels, candidate IDs, and released annotation means/counts.

The inspection checked that decoded images correspond to the displayed request, that three- versus four-candidate layouts preserve identity, and that visible labels agree with the manifest. Tensor records additionally confirm that all candidates enter Qwen (`image_grid_thw` has one entry per candidate). No new human annotation or cultural preference adjudication was produced. The inspection did not consult the main split's judge outputs or change an endpoint.
