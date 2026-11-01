import json
import numpy as np

res = json.loads(open('metrics/sweep_results.json').read())
runs = res['runs']

print(f"{'model':<7} {'lang':<10} {'lr':<8} {'seed':<6} {'best ep':>8} {'best dev':>9} {'last':>8}  {'capped?':<8}")
print("-" * 72)

capped = []
for r in sorted(runs, key=lambda x: (x['lang'], x['model'], x['lr'], x['seed'])):
    c = r['dev_curve']
    best = int(np.argmax(c)) + 1
    rising = best >= len(c) - 1
    if rising and not r['degenerate']:
        capped.append(r)
    short = 'XLM-R' if 'xlm' in r['model'] else 'MuRIL'
    tag = 'DEGEN' if r['degenerate'] else ('YES' if rising else '')
    print(f"{short:<7} {r['lang']:<10} {r['lr']:<8g} {r['seed']:<6} "
          f"{best:>4}/{len(c):<3} {max(c):>9.4f} {c[-1]:>8.4f}  {tag:<8}")

print()
print(f"{len(capped)} of {len(runs)} runs peaked at or next to the final epoch "
      f"and were stopped by the cap, not by convergence.")
print()
if capped:
    print("Configurations needing a longer run:")
    seen = set()
    for r in capped:
        k = (r['model'], r['lang'], r['lr'])
        if k not in seen:
            seen.add(k)
            short = 'XLM-R' if 'xlm' in r['model'] else 'MuRIL'
            print(f"  {short:<7} {r['lang']:<10} lr={r['lr']:g}")