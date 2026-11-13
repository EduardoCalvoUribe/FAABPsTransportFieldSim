"""Load a simulation NPZ file and render it as an MP4 animation.

Usage:
    python load_and_render.py                          # uses SOURCE_FILE below
    python load_and_render.py path/to/file.npz
    python load_and_render.py path/to/file.npz out.mp4
"""

from __future__ import annotations

import sys
import os
import time
import numpy as np

from src.visualization import create_payload_animation

###############################################################################
# Configuration — edit these or pass CLI args                                 #
###############################################################################

SOURCE_FILE   = "D:/PostThesis/data/snell_4000_20_1m_short2.npz"   # default input
OUTPUT_FILE   = "D:/PostThesis/visualizations/conference/snell_4000_20_1m_short2_final.mp4"  # None → auto-derive from source name
SHOW_VECTORS  = False                   # overlay polarity arrows
COLOR_BY_SCORE = False                   # True = rainbow by score, False = curvity

###############################################################################

def _derive_output(source: str) -> str:
    base = os.path.splitext(os.path.basename(source))[0]
    os.makedirs("visualizations", exist_ok=True)
    return f"visualizations/{base}.mp4"


def load_and_render(source_file: str, output_file: str | None = None,
                    show_vectors: bool = SHOW_VECTORS,
                    color_by_score: bool = COLOR_BY_SCORE) -> None:

    if output_file is None:
        output_file = _derive_output(source_file)

    # Guard against accidentally passing a .npz path as the output
    if not output_file.lower().endswith('.mp4'):
        output_file = os.path.splitext(output_file)[0] + '.mp4'
        print(f"Output extension corrected to .mp4: {output_file}")

    print(f"Loading {source_file} ...")
    t0 = time.time()
    data = np.load(source_file, mmap_mode='r')
    print(f"  keys: {list(data.keys())}")

    positions        = data['positions']           # (T, N, 2)
    payload_positions = data['payload_positions']  # (T, 2)
    curvity_values   = data['curvity_values']      # (T, N)
    n_particles      = positions.shape[1]

    print(f"  frames: {positions.shape[0]},  particles: {n_particles}")
    print(f"  payload frames: {payload_positions.shape[0]}")
    print(f"Data mapped in {time.time() - t0:.1f}s")

    # Optional arrays — present in full NPZ, absent in light NPZ
    polarity        = data['polarity']        if 'polarity'        in data else None
    particle_scores = data['particle_scores'] if 'particle_scores' in data else None

    # Per-particle display params: full NPZ stores arrays, light NPZ stores scalars
    def _to_array(key, fallback, n):
        if key in data:
            v = data[key]
            return v if v.ndim >= 1 else np.full(n, float(v))
        return np.full(n, fallback)

    particle_radius = _to_array('particle_radius', 1.0, n_particles)
    rot_diffusion   = _to_array('rot_diffusion',   float('nan'), n_particles)
    mobility        = _to_array('mobility',         float('nan'), n_particles)

    payload_mobility = float(data['payload_mobility']) if 'payload_mobility' in data else float('nan')

    params = {
        'n_particles':    n_particles,
        'box_size':       float(data['box_size']),
        'payload_radius': float(data['payload_radius']),
        'goal_position':  data['goal_position'],
        'particle_radius': particle_radius,
        'rot_diffusion':  rot_diffusion,
        'mobility':       mobility,
        'payload_mobility': payload_mobility,
        'walls':          data['walls'] if 'walls' in data else np.zeros((0, 5), dtype=np.float64),
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    create_payload_animation(
        positions, None, None,
        payload_positions, params, curvity_values,
        output_file=output_file,
        show_vectors=show_vectors and polarity is not None,
        polarity=polarity,
        particle_scores=particle_scores if color_by_score else None,
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    src = args[0] if len(args) >= 1 else SOURCE_FILE
    out = args[1] if len(args) >= 2 else OUTPUT_FILE
    load_and_render(src, out)
