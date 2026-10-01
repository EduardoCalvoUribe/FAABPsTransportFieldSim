"""
fig3b/run.py
============
Static-curvity 8-particle FAABP simulation with a passive payload.

Setup:
  Box 150×150, periodic BCs.
  Payload: passive (pushed by particle forces), starts at (75, 75), radius 20.
  No walls.  No polarity/score mechanic.
  All 4 particles: κ = -0.2
  Particles 0–2: start left of payload, oriented toward payload center.
  Particle 3: starts top-right of payload, facing bottom-left (−x, −y).

Equations of motion from src/simulation.py (Numba JIT).
Hyperparameters match main.py where relevant.

Run from project root:
    python fig3b/run.py

Output:
    fig3b/data.npz  (full trajectory, every step)
"""

import math
import os
import sys
import time
import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import compute_all_forces, update_orientation_vectors
from src.forces import build_wall_spatial_index


@njit(fastmath=True)
def compute_particle_particle_forces(positions, radii, stiffness, n_particles, box_size):
    """Brute-force O(N²) particle-particle repulsion. Fine for small N."""
    forces = np.zeros((n_particles, 2))
    half_box = 0.5 * box_size
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            if dx > half_box:
                dx -= box_size
            elif dx < -half_box:
                dx += box_size
            if dy > half_box:
                dy -= box_size
            elif dy < -half_box:
                dy += box_size
            sum_r = radii[i] + radii[j]
            dist2 = dx * dx + dy * dy
            if dist2 < sum_r * sum_r:
                if dist2 < 1e-20:
                    dx = 1e-5
                    dy = 1e-5
                    dist2 = dx * dx + dy * dy
                dist = math.sqrt(dist2)
                overlap = sum_r - dist
                mag = stiffness * overlap / dist
                fx = -mag * dx
                fy = -mag * dy
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
    return forces

# ── Hyperparameters ────────────────────────────────────────────────────────────

BOX_SIZE         = 150.0
PARTICLE_RADIUS  = 1.0
V0               = 1.0
MOBILITY         = 1.0
STIFFNESS        = 100.0
ROT_DIFFUSION    = 0.005
DT               = 0.01
N_STEPS          = 10000

PAYLOAD_RADIUS   = 30.0
PAYLOAD_CENTER   = np.array([75.0, 75.0])
PAYLOAD_MOBILITY = 0.2

PARTICLE_COLLISIONS = True   # toggle particle-particle repulsion on/off

N_PARTICLES = 4
CURVITY     = np.full(N_PARTICLES, -0.1)

# Particles 0–2: left of payload, pointed toward payload center
# Particle  3:   top-right of payload, facing bottom-left (−x, −y)
_left_ys = [70.0, 75.0, 80.0]
_left_pos = [[35.0, y] for y in _left_ys]
_diag = 1.0 / np.sqrt(2.0)

INIT_POS = np.array(_left_pos + [[105.0, 105.0]])

# Orientations: left particles aim at payload center; last one faces bottom-left
_left_orients = []
for _x, _y in _left_pos:
    _dx, _dy = PAYLOAD_CENTER[0] - _x, PAYLOAD_CENTER[1] - _y
    _norm = np.sqrt(_dx**2 + _dy**2)
    _left_orients.append([_dx / _norm, _dy / _norm])

INIT_ORIENT = np.array(_left_orients + [[-_diag, -_diag]])

# ── Wall setup (no walls) ──────────────────────────────────────────────────────

walls = np.zeros((0, 5), dtype=np.float64)
wall_cell_size = max(BOX_SIZE / 20.0, 2.0)
wall_grid_offsets, wall_grid_indices, n_wall_cells = build_wall_spatial_index(
    walls, BOX_SIZE, wall_cell_size, PAYLOAD_RADIUS
)

# ── Per-particle arrays ────────────────────────────────────────────────────────

radii      = np.full(N_PARTICLES, PARTICLE_RADIUS)
v0s        = np.full(N_PARTICLES, V0)
mobilities = np.full(N_PARTICLES, MOBILITY)
rot_diff   = np.full(N_PARTICLES, ROT_DIFFUSION)

# ── Numba warmup ───────────────────────────────────────────────────────────────

print("Warming up Numba JIT…")
_n  = 10
_p  = np.random.uniform(0, BOX_SIZE, (_n, 2))
_o  = np.zeros((_n, 2)); _o[:, 0] = 1.0
_r  = np.full(_n, PARTICLE_RADIUS)
_c  = np.zeros(_n)
_d  = np.full(_n, ROT_DIFFUSION)
_f  = np.zeros((_n, 2))
_pp = np.array([75.0, 75.0])
compute_all_forces(_p, _pp, _r, PAYLOAD_RADIUS, STIFFNESS, _n, BOX_SIZE,
                   walls, wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size)
update_orientation_vectors(_o, _f, _c, DT, _d, _n)
compute_particle_particle_forces(_p, _r, STIFFNESS, _n, BOX_SIZE)
print("Warmup done.")

# ── Initialise state ───────────────────────────────────────────────────────────

positions    = INIT_POS.copy()
orientations = INIT_ORIENT.copy()
velocities   = np.zeros((N_PARTICLES, 2))
payload_pos  = PAYLOAD_CENTER.copy()

# Pre-allocate output arrays (full data, every step)
saved_positions         = np.zeros((N_STEPS + 1, N_PARTICLES, 2))
saved_orientations      = np.zeros((N_STEPS + 1, N_PARTICLES, 2))
saved_curvity           = np.zeros((N_STEPS + 1, N_PARTICLES))
saved_payload_positions = np.zeros((N_STEPS + 1, 2))

saved_positions[0]          = positions
saved_orientations[0]       = orientations
saved_curvity[0]            = CURVITY
saved_payload_positions[0]  = payload_pos

# ── Main loop ─────────────────────────────────────────────────────────────────

print(f"Simulating {N_STEPS} steps  (t_end = {N_STEPS * DT:.1f})…")
t0 = time.time()

for step in range(1, N_STEPS + 1):
    particle_forces, payload_force = compute_all_forces(
        positions, payload_pos, radii, PAYLOAD_RADIUS, STIFFNESS, N_PARTICLES, BOX_SIZE,
        walls, wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size,
    )

    if PARTICLE_COLLISIONS:
        particle_forces += compute_particle_particle_forces(
            positions, radii, STIFFNESS, N_PARTICLES, BOX_SIZE
        )

    orientations = update_orientation_vectors(
        orientations, particle_forces, CURVITY, DT, rot_diff, N_PARTICLES,
    )

    for i in range(N_PARTICLES):
        velocities[i] = v0s[i] * orientations[i] + mobilities[i] * particle_forces[i]
        positions[i] += velocities[i] * DT

    payload_pos += PAYLOAD_MOBILITY * payload_force * DT

    positions   %= BOX_SIZE
    payload_pos %= BOX_SIZE

    saved_positions[step]         = positions
    saved_orientations[step]      = orientations
    saved_curvity[step]           = CURVITY
    saved_payload_positions[step] = payload_pos

print(f"Simulation complete in {time.time() - t0:.2f}s")

# ── Save NPZ ──────────────────────────────────────────────────────────────────

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.npz')
np.savez(
    out_path,
    positions=saved_positions,              # (N_STEPS+1, 8, 2)
    orientations=saved_orientations,        # (N_STEPS+1, 8, 2)
    curvity=saved_curvity,                  # (N_STEPS+1, 8)
    payload_positions=saved_payload_positions,  # (N_STEPS+1, 2)
    payload_center=PAYLOAD_CENTER,
    payload_radius=np.float64(PAYLOAD_RADIUS),
    box_size=np.float64(BOX_SIZE),
    particle_radius=np.float64(PARTICLE_RADIUS),
    v0=np.float64(V0),
    mobility=np.float64(MOBILITY),
    payload_mobility=np.float64(PAYLOAD_MOBILITY),
    particle_collisions=np.bool_(PARTICLE_COLLISIONS),
    stiffness=np.float64(STIFFNESS),
    rot_diffusion=np.float64(ROT_DIFFUSION),
    dt=np.float64(DT),
    n_steps=np.int64(N_STEPS),
    curvity_values=CURVITY,
    init_positions=INIT_POS,
)
print(f"Saved → {out_path}")
