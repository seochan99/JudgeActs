"""Export small JSON files for the project page (docs/static/data)."""
import csv, json
from pathlib import Path
from PIL import Image

R = Path("results/main")
OUT = Path("docs/static/data"); OUT.mkdir(parents=True, exist_ok=True)
IMG = Path("docs/static/images/example"); IMG.mkdir(parents=True, exist_ok=True)
r4 = lambda x: round(float(x), 4)

summary = {p["policy"]: p for p in json.load(open(R / "summary.json"))}
position = json.load(open(R / "position.json"))
abl = json.load(open(R / "ablations.json"))
hero = json.load(open(R / "hero_example.json"))
pid = hero["prompt_id"]

# 1. hero example ------------------------------------------------------------
rec = next(json.loads(l) for l in open("data/derived/main.jsonl") if json.loads(l)["prompt_id"] == pid)
picks = sorted((json.loads(l) for l in open("runs/qwen_main.jsonl") if json.loads(l)["prompt_id"] == pid),
               key=lambda d: d["permutation"])
gen_short = {"SD35": "sd35", "flux": "flux", "gpt-image": "gpt-image"}
cands = []
for c in rec["candidates"]:
    key = c["id"].split("_", 1)[1]
    fn = f"{gen_short[key]}.jpg"
    im = Image.open(c["image"]).convert("RGB"); im.thumbnail((480, 480), Image.LANCZOS)
    im.save(IMG / fn, "JPEG", quality=82, optimize=True, progressive=True)
    cands.append({"id": c["id"], "key": key, "generator": c["source_model"], "src": f"static/images/example/{fn}",
                  "rating": r4(c["utility"]), "ratings": c["alignment_ratings"], "n": c["n_utility"]})
orders = []
for d in picks:
    slot = d["parsed"]["choice"]
    orders.append({"slots": {s: v.split("_", 1)[1] for s, v in d["mapping"].items()},
                   "choice": slot, "selected": d["selected_id"].split("_", 1)[1]})
json.dump({"prompt_id": pid, "prompt": rec["prompt"], "country": rec["country"], "category": rec["category"],
           "judge": "Qwen3-VL-4B", "candidates": cands, "orders": orders},
          open(OUT / "example.json", "w"), indent=1)

# 2. slots ---------------------------------------------------------------------
def slots_from_position(m):
    s = position[m]["slots"]
    return {"calls": position[m]["calls"], "share": {k: r4(v["rate"]) for k, v in s.items()},
            "uniform": {k: r4(v["uniform_rate"]) for k, v in s.items()}}
def slots_from_abl(v):
    s = v["slot_choice"]["slots"]
    return {"calls": v["calls"], "share": {k[0]: r4(x["share"]) for k, x in s.items()},
            "uniform": {k[0]: r4(x["uniform"]) for k, x in s.items()}}
variants = [{"key": "main", "label": "Main", "note": "choice and ranking, 3 orders", **slots_from_abl(abl["reference_main"]),
             "first": r4(abl["reference_main"]["slot_first_rate"]), "first_uniform": r4(abl["reference_main"]["slot_first_uniform"])}]
expected = {"choice_only": 900, "opaque_labels": 900, "rotation4": 200}
labels = {"choice_only": ("Choice only", "no ranking in the output"),
          "opaque_labels": ("Opaque labels", "two-character codes instead of letters"),
          "rotation4": ("4 orders", "fourth rotation, 200 four-image pools")}
omitted = []
for k, exp in expected.items():
    v = abl.get(k)
    if not v or v.get("calls") != exp or "slot_first_rate" not in v:
        omitted.append({"key": k, "label": labels[k][0], "calls": (v or {}).get("calls", 0), "expected": exp})
        continue
    entry = {"key": k, "label": labels[k][0], "note": labels[k][1], "calls": v["calls"],
             "first": r4(v["slot_first_rate"]), "first_uniform": r4(v["slot_first_uniform"])}
    if k == "rotation4":
        entry["share"] = {s: r4(x) for s, x in v["slot_rates"].items()}
        entry["uniform"] = {s: 0.25 for s in "ABCD"}
    else:
        entry.update({kk: vv for kk, vv in slots_from_abl(v).items() if kk != "calls"})
    variants.append(entry)
json.dump({"models": {"qwen": {"label": "Qwen3-VL-4B", **slots_from_position("qwen")},
                      "smol": {"label": "SmolVLM2-2.2B", **slots_from_position("smol")}},
           "chi2": {"qwen": round(position["qwen"]["chi2"], 1), "smol": round(position["smol"]["chi2"], 1)},
           "variants": variants, "omitted": omitted}, open(OUT / "slots.json", "w"), indent=1)

# 3. gates ---------------------------------------------------------------------
prompts = [json.loads(l) for l in open("data/derived/main.jsonl")]
prompts.sort(key=lambda p: (p["country"], p["category"], p["prompt_id"]))
members = {}
for row in csv.DictReader(open(R / "prompt_metrics.csv")):
    members.setdefault(row["policy"], set()).add(row["prompt_id"])
def g(policy):
    p = summary[policy]
    return {"n": p["n"], "gain": r4(p["gain"]), "ci": [r4(x) for x in p["gain_ci"]]}
gates = {
    "none": {"label": "No gate", "kept": g("Qwen"), "rejected": None,
             "keep": [1] * len(prompts)},
    "unanimity": {"label": "Order unanimity", "kept": g("Qwen unanimous"), "rejected": g("Qwen, not unanimous"),
                  "keep": [int(p["prompt_id"] in members["Qwen unanimous"]) for p in prompts]},
    "crossmodel": {"label": "Cross-model agreement", "kept": g("Cross-model"), "rejected": g("Qwen, judges disagree"),
                   "keep": [int(p["prompt_id"] in members["Cross-model"]) for p in prompts]},
}
for k, v in gates.items():
    assert sum(v["keep"]) == v["kept"]["n"], (k, sum(v["keep"]), v["kept"]["n"])
json.dump({"n_prompts": len(prompts), "countries": [p["country"] for p in prompts], "gates": gates,
           "reference": {"clip": {"gain": r4(json.load(open(R / "baselines.json"))["policies"]["CLIP"]["gain"])},
                         "oracle": {"gain": r4(summary["Oracle"]["gain"])}}},
          open(OUT / "gates.json", "w"), separators=(",", ":"))
print("ok", len(prompts), [v["label"] for v in variants])
