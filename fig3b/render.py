"""
fig3b/render.py
===============
Render fig3b/data.npz as a static PNG.

Draws:
  - Payload trajectory (dashed path) + ghost at start + filled at end
  - Left particles (0..N-2): full trajectory line
  - Top-right particle (N-1): shadow dot trail every SHADOW_EVERY steps
  - Final particle positions (full opacity)
  - Orientation arrow at each final position

Run from project root:
    python fig3b/render.py

Output:
    fig3b/fig3.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

SHADOW_EVERY_PARTICLE = 400
SHADOW_EVERY_PAYLOAD  = 2000

# ── Load data ─────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(HERE, 'data.npz'))

positions         = data['positions']           # (T, 8, 2)
orientations      = data['orientations']        # (T, 8, 2)
payload_positions = data['payload_positions']   # (T, 2)
PAYLOAD_R         = float(data['payload_radius'])
BOX_SIZE          = float(data['box_size'])
PARTICLE_R        = float(data['particle_radius'])
DT                = float(data['dt'])
CURVITY_VALS      = data['curvity_values']      # (8,)

N_PARTICLES = positions.shape[1]
n_frames    = positions.shape[0]

ORANGE = (0.924, 0.390, 0.124)
COLORS = [ORANGE] * N_PARTICLES

# ── Figure / axis ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
ax.axis('off')
ax.set_position([0, 0, 1, 1])

# ── Payload ───────────────────────────────────────────────────────────────────

ax.plot(payload_positions[:, 0], payload_positions[:, 1],
        color='#737BA8', alpha=0.45, linewidth=1.5, linestyle='--', zorder=1)

# Shadow circles along payload trajectory (same cadence as particle shadows)
_payload_shadow_steps = list(range(0, n_frames, SHADOW_EVERY_PAYLOAD))
_n_payload_shadows    = len(_payload_shadow_steps)
for _idx, _step in enumerate(_payload_shadow_steps[:-1]):   # skip last; final circle drawn below
    _alpha = 0.08 + (_idx / max(_n_payload_shadows - 1, 1)) * (0.10 - 0.04)
    ax.add_patch(Circle(payload_positions[_step], PAYLOAD_R,
                        facecolor='#737BA8', edgecolor='none', alpha=_alpha, zorder=2))

ax.add_patch(Circle(payload_positions[-1], PAYLOAD_R,
                    facecolor='#737BA8', edgecolor='none', alpha=1.0, zorder=3))

# s in scatter is marker area in pt²; figure is 10" wide over BOX_SIZE data units
_pts_per_unit = 10.0 * 72.0 / BOX_SIZE
PARTICLE_S    = np.pi * (PARTICLE_R * _pts_per_unit) ** 2

# ── Left particles: full trajectory line (commented out) ──────────────────────

# for i in range(N_PARTICLES - 1):
#     ax.plot(positions[:, i, 0], positions[:, i, 1],
#             color=COLORS[i], alpha=0.5, linewidth=1.0, zorder=4)

# ── All particles: shadow dots every SHADOW_EVERY_PARTICLE steps ──────────────

ALPHA_MIN  = 0.16
ALPHA_MAX  = 0.50

shadow_steps = list(range(0, n_frames, SHADOW_EVERY_PARTICLE))
n_shadows    = len(shadow_steps)

for idx, step in enumerate(shadow_steps):
    alpha = ALPHA_MIN + (idx / max(n_shadows - 1, 1)) * (ALPHA_MAX - ALPHA_MIN)
    for i in range(N_PARTICLES):
        ax.scatter(positions[step, i, 0], positions[step, i, 1],
                   s=PARTICLE_S, color=COLORS[i], alpha=alpha,
                   linewidths=0, zorder=4)

# ── Final particle positions ───────────────────────────────────────────────────

ax.scatter(positions[-1, :, 0], positions[-1, :, 1],
           s=PARTICLE_S, c=COLORS, alpha=0.92,
           linewidths=0.8, edgecolors='white', zorder=6)

# ── Orientation arrows at final position ──────────────────────────────────────

ARROW_SCALE = 1.0
for i in range(N_PARTICLES):
    px, py = positions[-1, i]
    ox, oy = orientations[-1, i]
    ax.quiver(px, py, ox * ARROW_SCALE, oy * ARROW_SCALE,
              angles='xy', scale_units='xy', scale=1,
              color='black', alpha=0.85,
              width=0.0015, headwidth=3, headlength=4,
              zorder=7)

# ── Label ─────────────────────────────────────────────────────────────────────

ax.text(4, 142,
        rf'$\kappa = {float(CURVITY_VALS[0]):+.1f}$',
        color=ORANGE, fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='white', edgecolor=ORANGE, alpha=0.75),
        zorder=8)

# ── Save ──────────────────────────────────────────────────────────────────────

out = os.path.join(HERE, 'fig3b.png')
fig.savefig(out, dpi=150)
plt.close()
print(f"Saved → {out}")
