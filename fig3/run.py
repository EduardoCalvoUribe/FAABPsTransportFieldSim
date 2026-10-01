"""
fig3/run.py
===========
Static-curvity two-particle FAABP simulation.

Setup:
  Box 200×200, periodic BCs.
  Payload: static at (100, 100), radius 20.
  No walls.  No polarity/score mechanic.
  Particle 0: κ = -0.5
  Particle 1: κ = +0.5
  Both start to the left of the payload, heading right (+x).

Equations of motion from src/simulation.py (Numba JIT).
Hyperparameters match main.py where relevant.

Run from project root:
    python fig3/run.py

Output:
    fig3/data.npz  (full trajectory, every step)
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import compute_all_forces, update_orientation_vectors
from src.forces import build_wall_spatial_index

# ── Hyperparameters (from main.py where relevant) ─────────────────────────────

BOX_SIZE        = 100.0
PARTICLE_RADIUS = 1.0
V0              = 2.0    
MOBILITY        = 1.0   
STIFFNESS       = 25.0   
ROT_DIFFUSION   = 0.01    
DT              = 0.01  
N_STEPS         = 2000

PAYLOAD_RADIUS  = 25.0   
PAYLOAD_CENTER  = np.array([90.0, 50.0])

N_PARTICLES     = 2
CURVITY         = np.array([-0.1, 0.4])   # static, one value per particle

# Both start to the left of the payload, heading right (+x).
# Offset in y (±9) so each hits the payload on a different arc —
# this makes the opposite curvity signs produce visibly opposite deflections.
INIT_POS = np.array([
    [50.0,  45.0],   # particle 0  (κ = -0.5)
    [50.0, 55.0],   # particle 1  (κ = +0.5)
])
INIT_ORIENT = np.array([
    [1.0, 0.0],
    [1.0, 0.0],
])

# ── Wall setup (no walls) ──────────────────────────────────────────────────────

walls = np.zeros((0, 5), dtype=np.float64)
wall_cell_size = max(BOX_SIZE / 20.0, 2.0)
wall_grid_offsets, wall_grid_indices, n_wall_cells = build_wall_spatial_index(
    walls, BOX_SIZE, wall_cell_size, PAYLOAD_RADIUS
)

# ── Per-particle arrays ────────────────────────────────────────────────────────

radii     = np.full(N_PARTICLES, PARTICLE_RADIUS)
v0s       = np.full(N_PARTICLES, V0)
mobilities = np.full(N_PARTICLES, MOBILITY)
rot_diff  = np.full(N_PARTICLES, ROT_DIFFUSION)

# ── Numba warmup ───────────────────────────────────────────────────────────────

print("Warming up Numba JIT…")
_n  = 10
_p  = np.random.uniform(0, BOX_SIZE, (_n, 2))
_o  = np.zeros((_n, 2)); _o[:, 0] = 1.0
_r  = np.full(_n, PARTICLE_RADIUS)
_c  = np.zeros(_n)
_d  = np.full(_n, ROT_DIFFUSION)
_f  = np.zeros((_n, 2))
_pp = np.array([50.0, 50.0])
compute_all_forces(_p, _pp, _r, PAYLOAD_RADIUS, STIFFNESS, _n, BOX_SIZE,
                   walls, wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size)
update_orientation_vectors(_o, _f, _c, DT, _d, _n)
print("Warmup done.")

# ── Initialise state ───────────────────────────────────────────────────────────

positions    = INIT_POS.copy()
orientations = INIT_ORIENT.copy()
velocities   = np.zeros((N_PARTICLES, 2))
payload_pos  = PAYLOAD_CENTER.copy()   # never updated — static payload

# Pre-allocate output arrays (full data, every step)
saved_positions    = np.zeros((N_STEPS + 1, N_PARTICLES, 2))
saved_orientations = np.zeros((N_STEPS + 1, N_PARTICLES, 2))
saved_curvity      = np.zeros((N_STEPS + 1, N_PARTICLES))

saved_positions[0]    = positions
saved_orientations[0] = orientations
saved_curvity[0]      = CURVITY

# ── Main loop ─────────────────────────────────────────────────────────────────

print(f"Simulating {N_STEPS} steps  (t_end = {N_STEPS * DT:.1f})…")
t0 = time.time()

for step in range(1, N_STEPS + 1):
    # Forces on particles from payload (walls array is empty → wall forces = 0)
    particle_forces, _ = compute_all_forces(
        positions, payload_pos, radii, PAYLOAD_RADIUS, STIFFNESS, N_PARTICLES, BOX_SIZE,
        walls, wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size,
    )

    # Orientation update: torque from curvity + rotational diffusion noise
    orientations = update_orientation_vectors(
        orientations, particle_forces, CURVITY, DT, rot_diff, N_PARTICLES,
    )

    # Position update: self-propulsion + mobility × force
    for i in range(N_PARTICLES):
        velocities[i] = v0s[i] * orientations[i] + mobilities[i] * particle_forces[i]
        positions[i] += velocities[i] * DT

    positions %= BOX_SIZE   # periodic BCs; payload stays fixed

    saved_positions[step]    = positions
    saved_orientations[step] = orientations
    saved_curvity[step]      = CURVITY

print(f"Simulation complete in {time.time() - t0:.2f}s")

# ── Save NPZ (full data, not light) ───────────────────────────────────────────

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.npz')
np.savez(
    out_path,
    positions=saved_positions,        # (N_STEPS+1, 2, 2)
    orientations=saved_orientations,  # (N_STEPS+1, 2, 2)
    curvity=saved_curvity,            # (N_STEPS+1, 2)
    payload_center=PAYLOAD_CENTER,
    payload_radius=np.float64(PAYLOAD_RADIUS),
    box_size=np.float64(BOX_SIZE),
    particle_radius=np.float64(PARTICLE_RADIUS),
    v0=np.float64(V0),
    mobility=np.float64(MOBILITY),
    stiffness=np.float64(STIFFNESS),
    rot_diffusion=np.float64(ROT_DIFFUSION),
    dt=np.float64(DT),
    n_steps=np.int64(N_STEPS),
    curvity_values=CURVITY,
    init_positions=INIT_POS,
)
print(f"Saved → {out_path}")
