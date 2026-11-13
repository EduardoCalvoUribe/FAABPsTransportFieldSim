import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv
import os

# ── Physical constants ────────────────────────────────────────────────────────
DT = 0.01
V0 = 5.0
R  = 1.0
BOX_SIZE = 1200.0   # 20×20 maze: 60 * G sim-units
CELL_SIZE = BOX_SIZE / 20
PAYLOAD_START = np.array([CELL_SIZE / 2, CELL_SIZE / 2])
GOAL          = np.array([BOX_SIZE - CELL_SIZE / 2, BOX_SIZE - CELL_SIZE / 2])
STRAIGHT_DIST = np.linalg.norm(GOAL - PAYLOAD_START)  # 1140√2 ≈ 1612
T_STAR_STRAIGHT = STRAIGHT_DIST / R  # straight-line crossing time in t* units

# ── Load data ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(_HERE, 'wallprune_results_G20.csv')

walls_remaining = []
prune_steps = []
final_steps = []

with open(DATA_FILE, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['goal_reached'] == 'True':
            walls_remaining.append(int(row['walls_remaining']))
            prune_steps.append(int(row['prune_step']))
            final_steps.append(int(row['final_step']))

M = 361  # total interior walls for G=20
prune_steps = np.array(prune_steps)
t_star = np.array(final_steps, dtype=float) * DT * V0 / R

# Sort by prune_step ascending (full maze → open)
order = np.argsort(prune_steps)
px = prune_steps[order]
ty = t_star[order]

COLOR = '#7B68EE'

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))

px_plot = px + 1  # shift so 0 → 1 (log-safe); tick labels show original values
ax.scatter(px_plot, ty, color=COLOR, s=64, zorder=3)

for x, y in zip(px_plot, ty):
    ax.annotate(f'{y:,.0f}', (x, y), textcoords='offset points',
                xytext=(6, 4), va='bottom', fontsize=8)

ax.axhline(T_STAR_STRAIGHT, color='gray', linestyle=':', linewidth=1.5,
           label=rf'straight-line distance ($d/r = {T_STAR_STRAIGHT:.0f}$)')
ax.legend(fontsize=9)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(0.85, (px.max() + 1) * 1.5)
ax.set_xticks(px_plot)
ax.set_xticklabels([str(v) for v in px], rotation=45, ha='right')
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

ax.set_xlabel('Interior Walls Pruned (cumulative)')
ax.set_ylabel(r'Time to Reach Goal  ($t^* = t\, v_0 / r$)')
ax.set_title('Navigation Time vs Wall Density\n'
             r'($t^* = t\,v_0/r$,  20×20 maze, 1 trial each)')

ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.grid(True, which='both', linestyle='--', alpha=0.4)
plt.tight_layout()

outfile = os.path.join(_HERE, 'wallprune_G20.png')
plt.savefig(outfile, dpi=150)
plt.show()
print(f"Saved {outfile}")
