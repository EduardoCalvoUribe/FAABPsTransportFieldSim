"""
fig3/render.py
==============
Render fig3/data.npz as a static PNG.

Draws:
  - Static payload (darkgreen ring, style from src/visualization.py)
  - Full particle path as a thin low-alpha line
  - Position shadows every SHADOW_EVERY steps (small circles, low alpha)
    — illustrate movement through time
  - Final particle positions (full opacity, larger)
  - Orientation arrow at the final position

Run from project root:
    python fig3/render.py

Output:
    fig3/fig3.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

SHADOW_EVERY = 200   # draw one shadow dot per particle every N steps

# ── Load data ─────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(HERE, 'data.npz'))

positions      = data['positions']        # (T, 2, 2)
orientations   = data['orientations']     # (T, 2, 2)
PAYLOAD_CENTER = data['payload_center']   # (2,)
PAYLOAD_R      = float(data['payload_radius'])
BOX_SIZE       = float(data['box_size'])
PARTICLE_R     = float(data['particle_radius'])
DT             = float(data['dt'])
CURVITY_VALS   = data['curvity_values']   # (2,)

n_frames = positions.shape[0]

# ── Curvity colours: negative → orange, positive → blue ──────────────────────

def curvity_color(c: float):
    return (0.924, 0.390, 0.124) if c < 0 else (0.124, 0.658, 0.924)

COLORS = [curvity_color(float(k)) for k in CURVITY_VALS]

# ── Figure / axis (style from src/visualization.py) ──────────────────────────

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
ax.axis('off')
ax.set_position([0, 0, 1, 1])

# ── Static payload ────────────────────────────────────────────────────────────

ax.add_patch(Circle(PAYLOAD_CENTER, PAYLOAD_R,
                    facecolor='#737BA8', edgecolor='none', alpha=1.0, zorder=3))

# ── Full path lines ────────────────────────────────────────────────────────────

# for i in range(2):
#     ax.plot(positions[:, i, 0], positions[:, i, 1],
#             color=COLORS[i], alpha=0.35, linewidth=1.0, zorder=1)

# ── Shadow dots every SHADOW_EVERY steps ──────────────────────────────────────

SHADOW_S   = np.pi * (PARTICLE_R * 9) ** 2
ALPHA_MIN  = 0.08   # alpha of the oldest shadow
ALPHA_MAX  = 0.50   # alpha of the most recent shadow

shadow_steps = list(range(0, n_frames, SHADOW_EVERY))
n_shadows    = len(shadow_steps)

for idx, step in enumerate(shadow_steps):
    alpha = ALPHA_MIN + (idx / max(n_shadows - 1, 1)) * (ALPHA_MAX - ALPHA_MIN)
    for i in range(2):
        ax.scatter(positions[step, i, 0], positions[step, i, 1],
                   s=SHADOW_S, color=COLORS[i], alpha=alpha,
                   linewidths=0, zorder=2)

# ── Final particle positions ───────────────────────────────────────────────────

FINAL_S = np.pi * (PARTICLE_R * 9) ** 2
ax.scatter(positions[-1, :, 0], positions[-1, :, 1],
           s=FINAL_S, c=COLORS, alpha=0.92,
           linewidths=0.8, edgecolors='white', zorder=6)

# ── Orientation arrows at final position ──────────────────────────────────────

ARROW_SCALE = 1.0
for i in range(2):
    px, py = positions[-1, i]
    ox, oy = orientations[-1, i]
    ax.quiver(px, py, ox * ARROW_SCALE, oy * ARROW_SCALE,
              angles='xy', scale_units='xy', scale=1,
              color='black', alpha=0.85,
              width=0.0015, headwidth=3, headlength=4,
              zorder=7)

# ── Labels ────────────────────────────────────────────────────────────────────

label_anchors = [(4, 25), (4, 175)]
for i, (lx, ly) in enumerate(label_anchors):
    ax.text(lx, ly,
            rf'$\kappa = {float(CURVITY_VALS[i]):+.1f}$',
            color=COLORS[i], fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor=COLORS[i], alpha=0.75),
            zorder=8)

# ── Save ──────────────────────────────────────────────────────────────────────

out = os.path.join(HERE, 'fig3.png')
fig.savefig(out, dpi=150)
plt.close()
print(f"Saved → {out}")
