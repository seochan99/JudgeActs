# Selection Audit Browser

Offline, dependency-free viewer for the selection audit: an overview dashboard,
a filterable prompt explorer, and a per-prompt detail panel (three presentation
orders, per-candidate human ratings, selection metrics, agreement gates).
Every number shown is read from the repository's data, run, and result files.

## Build and run

```bash
uv run python tools/audit_browser/export.py      # writes data.js and thumbs/ (~1 min first time)
cd tools/audit_browser && python -m http.server 8765
# open http://localhost:8765/   (opening index.html via file:// also works)
```

Routes (URL hash): `#overview`, `#explore?country=Japan&category=etiquette&outcome=below&judge=smol&q=tea&sort=regret`,
`#detail=<prompt-id prefix>[&judge=qwen|smol]` (append `?country=...` to set the explorer filters behind the panel).
Outcome keys: `below`, `best`, `unanimous`, `disagree`, `agree`.

## Inputs (read-only)

`data/derived/main.jsonl`, `runs/qwen_main.jsonl`, `runs/smol_main.jsonl`,
`results/main/summary.json`, `results/main/position.json`, `results/main/country.json`.

## Paper screenshots

Serve as above, then capture at 1600x1000, device scale factor 2, with headless Chrome, e.g.
`"<chrome>" --headless=new --hide-scrollbars --window-size=1600,1000 --force-device-scale-factor=2 --virtual-time-budget=12000 --screenshot=paper/figures/browser_detail.png "http://localhost:8765/index.html#detail=7c51b057?country=Chile"`.
`uv run python tools/audit_browser/compose_panels.py` combines the three PNGs into `paper/figures/browser_panels.pdf`.

## Images

Candidate images are CulturalFrames outputs. They are **not redistributed**: `thumbs/`
is generated locally from `data/images/` and is git-ignored. Do not publish it.
