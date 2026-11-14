"""Wall-pruning figure (G=20).

Usage (from anywhere):
    python plot_wallprune.py [extra_results.txt ...]

Data sources (merged, de-duplicated by the `output` name):
  * wallprune_results_G20.csv   - original single-trial results (run 1)
  * wallprune_results.txt       - raw file appended by the Slurm jobs
                                  (copy it from Snellius next to this script,
                                  or pass its path as an argument)
Any `RESULT_LINE: key=value ...` line is parsed, so the raw Slurm results
file can be used as-is. Lines without a `_runN` suffix count as run 1.
"""
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
DT = 0.01
V0 = 5.0
R  = 1.0
G  = 20
BOX_SIZE = 60.0 * G   # 20×20 maze: 60 * G sim-units
CELL_SIZE = BOX_SIZE / G
PAYLOAD_START = np.array([CELL_SIZE / 2, CELL_SIZE / 2])
GOAL          = np.array([BOX_SIZE - CELL_SIZE / 2, BOX_SIZE - CELL_SIZE / 2])
STRAIGHT_DIST = np.linalg.norm(GOAL - PAYLOAD_START)  # 1140√2 ≈ 1612
T_STAR_STRAIGHT = STRAIGHT_DIST / R  # straight-line crossing time in t* units

# ── Load data ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(_HERE, 'wallprune_results_G20.csv')
RAW_FILES = [os.path.join(_HERE, 'wallprune_results.txt')] + sys.argv[1:]

records = {}  # output name -> (prune_step, final_step)


def add(output, prune_step, final_step, goal_reached):
    if goal_reached:
        records[output] = (int(prune_step), int(final_step))


if os.path.exists(CSV_FILE):
    with open(CSV_FILE, newline='') as f:
        for row in csv.DictReader(f):
            add(row['output'], row['prune_step'], row['final_step'],
                row['goal_reached'] == 'True')

for path in RAW_FILES:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for line in f:
            if not line.startswith('RESULT_LINE:'):
                continue
            kv = dict(re.findall(r'(\w+)=(\S+)', line))
            add(kv['output'], kv['prune_step'], kv['final_step'],
                kv['goal_reached'] == 'True')

if not records:
    sys.exit("No results found - put wallprune_results.txt next to this script "
             "or pass its path as an argument.")

steps = np.array([v[0] for v in records.values()])
t_all = np.array([v[1] for v in records.values()], dtype=float) * DT * V0 / R

unique_steps = np.unique(steps)
means = np.array([t_all[steps == s].mean() for s in unique_steps])
stds  = np.array([t_all[steps == s].std(ddof=1) if (steps == s).sum() > 1 else 0.0
                  for s in unique_steps])
counts = np.array([(steps == s).sum() for s in unique_steps])

print("prune_step  n   mean t*   std")
for s, n, m, sd in zip(unique_steps, counts, means, stds):
    print(f"{s:>10}  {n}  {m:>9.0f}  {sd:>8.0f}")

COLOR = '#7B68EE'

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))

px_all  = steps + 1           # shift so 0 → 1 (log-safe); tick labels show original values
px_mean = unique_steps + 1

ax.scatter(px_all, t_all, color=COLOR, s=18, alpha=0.35, zorder=2, label='individual trials')
ax.errorbar(px_mean, means, yerr=stds, fmt='o', color=COLOR, markersize=8,
            capsize=4, zorder=3, label='mean ± std')

for x, y in zip(px_mean, means):
    ax.annotate(f'{y:,.0f}', (x, y), textcoords='offset points',
                xytext=(6, 4), va='bottom', fontsize=8)

ax.axhline(T_STAR_STRAIGHT, color='gray', linestyle=':', linewidth=1.5,
           label=rf'straight-line distance ($d/r = {T_STAR_STRAIGHT:.0f}$)')
ax.legend(fontsize=9)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(0.85, (unique_steps.max() + 1) * 1.5)
ax.set_ylim(1000, 100000)
ax.set_xticks(px_mean)
ax.set_xticklabels([str(v) for v in unique_steps], rotation=45, ha='right')
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

ax.set_xlabel('Interior Walls Pruned (cumulative)')
ax.set_ylabel(r'Time to Reach Goal  ($t^* = t\, v_0 / r$)')
n_min, n_max = counts.min(), counts.max()
trials = f'{n_max}' if n_min == n_max else f'{n_min}-{n_max}'
ax.set_title('Navigation Time vs Wall Density\n'
             rf'($t^* = t\,v_0/r$,  20×20 maze, {trials} trials per condition)')

ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.grid(True, which='both', linestyle='--', alpha=0.4)
plt.tight_layout()

outfile = os.path.join(_HERE, 'wallprune_G20.png')
plt.savefig(outfile, dpi=150)
print(f"Saved {outfile}")
plt.show()
