"""Wait for the existing primary worker, then safely resume and build artifacts.

This coordinator never starts a second copy of the active primary experiment.
If that worker exits early, src.complete resumes only missing frozen request keys.
"""
import argparse
import os
import subprocess
import sys
import time

from .common import ROOT, load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-pid", type=int, required=True)
    args = parser.parse_args()
    expected = 3 * len(load_jsonl(ROOT / "data/derived/main.jsonl"))
    path = ROOT / "runs/qwen_main.jsonl"
    last_report = 0
    while True:
        # Count only complete JSON lines; the runner writes one record per request.
        try:
            rows = load_jsonl(path) if path.exists() else []
        except ValueError:
            time.sleep(1)
            continue
        keys = {(r["prompt_id"], r["permutation"]) for r in rows}
        if len(keys) != len(rows):
            raise RuntimeError("Duplicate primary requests; refusing automatic finalization")
        if len(keys) == expected:
            break
        try:
            os.kill(args.primary_pid, 0)
        except ProcessLookupError:
            print("Primary worker exited; resuming missing frozen requests.", flush=True)
            break
        if time.monotonic() - last_report >= 50:
            print(f"Waiting for active primary worker: {len(keys)}/{expected}", flush=True)
            last_report = time.monotonic()
        time.sleep(5)
    subprocess.run([sys.executable, "-m", "src.complete"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
