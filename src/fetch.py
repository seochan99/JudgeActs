"""Download only pinned, public third-party artifacts."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from .common import ROOT, save_json

DATASET = 'mair-lab/CulturalFrames'
MODELS = {'qwen': 'Qwen/Qwen3-VL-4B-Instruct', 'smol': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['metadata', 'dataset', 'qwen', 'smol'])
    args = parser.parse_args()
    api = HfApi()
    dest = ROOT/'provenance/sources.json'
    if not dest.exists():
        info = api.dataset_info(DATASET)
        card = info.card_data.to_dict() if info.card_data else {}
        sources = {'fetched_utc': datetime.now(timezone.utc).isoformat(),
                   'dataset': {'id': DATASET, 'revision': info.sha, 'license': card.get('license', 'NOT_DECLARED_IN_CARD')},
                   'models': {}}
        for name, repo in MODELS.items():
            info = api.model_info(repo)
            sources['models'][name] = {'id': repo, 'revision': info.sha, 'license': info.card_data.to_dict().get('license')}
        save_json(dest, sources)
    import json
    sources = json.loads(dest.read_text())
    if args.kind in ['metadata', 'dataset']:
        d = sources['dataset']
        for filename in ['README.md', 'culturalframes_human_annotations.json']:
            hf_hub_download(d['id'], filename, repo_type='dataset', revision=d['revision'], local_dir=ROOT/'data/raw/culturalframes')
        if args.kind == 'dataset':
            snapshot_download(d['id'], repo_type='dataset', revision=d['revision'], allow_patterns=['data/*.parquet'], local_dir=ROOT/'data/raw/culturalframes', max_workers=4)
    else:
        m = sources['models'][args.kind]
        snapshot_download(m['id'], revision=m['revision'], local_dir=ROOT/'models'/args.kind, max_workers=3)
    print('download complete:', args.kind, flush=True)

if __name__ == '__main__':
    main()
