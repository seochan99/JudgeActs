import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261002

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

def orders(group):
    ids = [c['id'] for c in group['candidates']]
    # Three distinct cyclic orders; identity is always stored separately from labels.
    offset = int(digest(f'{SEED}:{group["prompt_id"]}:order')[:16], 16) % (len(ids)-1) + 1
    shifts = [0, offset] + [i for i in range(1, len(ids)) if i != offset][:1]
    return [ids[s:]+ids[:s] for s in shifts]

def strict_parse(raw, labels):
    data = json.loads(raw.strip())
    if not isinstance(data, dict) or set(data) != {'choice', 'ranking'}:
        raise ValueError('expected only choice and ranking')
    ranking = data['ranking']
    if not isinstance(ranking, list) or len(ranking) != len(labels) or set(ranking) != set(labels):
        raise ValueError('ranking must contain every candidate once')
    if data['choice'] != ranking[0]:
        raise ValueError('choice must be top of ranking')
    return data
