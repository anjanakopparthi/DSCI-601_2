"""Rebuild the sweep summary table from the per-run log.

Run from the repo root:  python evaluation/rebuild_summary.py

metrics/sweep_summary.json currently holds only the Malayalam entries: the
Malayalam sweep was run in a separate session and wrote the file rather than
merging into it, so the Tamil entries were lost. The per-run log,
metrics/sweep_results.json, still has every run, so the summary is
reconstructible.

Selection rule, matching the sweep: the learning rate is chosen on DEV at the
grid seed, then that learning rate is replicated across the extra seeds and the
TEST figures are reported as a mean and standard deviation over seeds. Dev
decides, test reports.

Writes the merged summary back to metrics/sweep_summary.json, keeping any
existing keys it does not recompute.
"""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
GRID_SEED = 42

runs = json.load(open(ROOT / "metrics/sweep_results.json"))["runs"]
print(f"{len(runs)} runs in the log")

stages = defaultdict(int)
for r in runs:
    stages[r.get("stage")] += 1
print("by stage:  " + ",  ".join(f"{k}: {v}" for k, v in sorted(
    stages.items(), key=lambda kv: str(kv[0]))))

SHORT = {"xlm-roberta-base": "XLM-R", "google/muril-base-cased": "MuRIL"}

# ---- the learning-rate grid, decided on dev at the grid seed ---------------
print("\nStage 1 -- learning-rate grid, dev macro F1 at seed "
      f"{GRID_SEED}\n")
grid = defaultdict(dict)
for r in runs:
    if r["seed"] == GRID_SEED and r.get("dev_macro_f1") is not None:
        key = (SHORT.get(r["model"], r["model"]), r["lang"])
        grid[key].setdefault(r["lr"], r["dev_macro_f1"])

lrs = sorted({lr for v in grid.values() for lr in v})
print(f"  {'model':<7} {'language':<11} " +
      " ".join(f"{lr:>9.0e}" for lr in lrs) + f" {'selected':>11} {'spread':>8}")
print("  " + "-" * (20 + 10 * len(lrs) + 21))
selected = {}
for key in sorted(grid):
    row = grid[key]
    best = max(row, key=row.get)
    selected[key] = best
    spread = max(row.values()) - min(row.values())
    cells = " ".join(f"{row.get(lr, float('nan')):>9.4f}" for lr in lrs)
    print(f"  {key[0]:<7} {key[1]:<11} {cells} {best:>11.0e} {spread:>8.4f}")

# ---- multi-seed replication at the selected learning rate ------------------
print("\nStage 2 -- test macro F1 across seeds at the selected rate\n")
agg = defaultdict(list)
for r in runs:
    key = (SHORT.get(r["model"], r["model"]), r["lang"])
    if key in selected and r["lr"] == selected[key] and \
       r.get("test_macro_f1") is not None:
        agg[key].append((r["seed"], r["test_macro_f1"], r.get("test_hope_f1"),
                         r.get("degenerate")))

print(f"  {'model':<7} {'language':<11} {'lr':>8} {'seeds':>6} "
      f"{'test macro F1':>16} {'hope F1':>9}   runs")
print("  " + "-" * 86)
summary = {}
for key in sorted(agg):
    rows = sorted(agg[key])
    f1s = [x[1] for x in rows]
    hope = [x[2] for x in rows if x[2] is not None]
    sd = stdev(f1s) if len(f1s) > 1 else 0.0
    deg = sum(1 for x in rows if x[3])
    print(f"  {key[0]:<7} {key[1]:<11} {selected[key]:>8.0e} {len(f1s):>6} "
          f"{mean(f1s):>9.4f} +- {sd:.4f} {mean(hope):>9.4f}   "
          + ", ".join(f"{v:.4f}" for v in f1s)
          + (f"   [{deg} degenerate]" if deg else ""))
    summary[f"{key[0]}|{key[1]}"] = dict(
        lr=selected[key], n_seeds=len(f1s),
        test_macro_f1_mean=round(mean(f1s), 4),
        test_macro_f1_std=round(sd, 4),
        test_macro_f1_runs=[round(v, 4) for v in f1s],
        test_hope_f1_mean=round(mean(hope), 4) if hope else None)

# ---- seed spread against model gap ----------------------------------------
print("\nSeed spread vs model gap, per language\n")
by_lang = defaultdict(dict)
for k, v in summary.items():
    m, l = k.split("|")
    by_lang[l][m] = v
for l, d in sorted(by_lang.items()):
    if len(d) < 2:
        continue
    gap = abs(d["XLM-R"]["test_macro_f1_mean"] - d["MuRIL"]["test_macro_f1_mean"])
    worst = max(v["test_macro_f1_std"] for v in d.values())
    spread = max(max(v["test_macro_f1_runs"]) - min(v["test_macro_f1_runs"])
                 for v in d.values())
    verdict = "model gap is inside seed noise" if gap < spread else \
              "model gap exceeds seed noise"
    print(f"  {l:<11} model gap {gap:.4f}   worst seed sd {worst:.4f}   "
          f"seed range {spread:.4f}   -> {verdict}")

# ---- single-run English metrics, if present -------------------------------
print("\nEnglish (no sweep was run; single-run metrics files)\n")
found = False
for p in sorted(ROOT.glob("metrics/*english_metrics.json")):
    d = json.load(open(p))
    found = True
    print(f"  {p.name}")
    print("    " + json.dumps(d)[:400] + ("..." if len(json.dumps(d)) > 400 else ""))
if not found:
    print("  none found")

# ---- merge, do not overwrite ----------------------------------------------
out = ROOT / "metrics/sweep_summary.json"
existing = json.load(open(out)) if out.exists() else {}
merged = dict(existing)
merged.update(summary)
json.dump(merged, open(out, "w"), indent=2)
print(f"\nsweep_summary.json now holds {len(merged)} entries: "
      f"{', '.join(sorted(merged))}")
