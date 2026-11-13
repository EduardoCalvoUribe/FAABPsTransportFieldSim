"""Split a simulation NPZ into N equal pieces.

Output files are written alongside the source file with _part1 … _partN suffixes.
"""

import io
import os
import zipfile

import numpy as np

SOURCE  = "D:/PostThesis/data/snell_4000_20_1m.npz"
N_PARTS = 4

FRAME_ARRAYS = {
    'positions', 'orientations', 'velocities',
    'payload_positions', 'payload_velocities',
    'curvity_values', 'polarity', 'particle_scores',
}


def split_npz(source_file, n_parts=N_PARTS):
    base, ext = os.path.splitext(source_file)

    # Find total frame count
    with zipfile.ZipFile(source_file, 'r') as zf:
        for name in zf.namelist():
            key = name[:-4] if name.endswith('.npy') else name
            if key in ('positions', 'payload_positions'):
                with zf.open(name) as f:
                    arr = np.load(io.BytesIO(f.read()))
                n_frames = arr.shape[0]
                break

    boundaries = [round(n_frames * i / n_parts) for i in range(n_parts + 1)]
    out_paths  = [f"{base}_part{i+1}{ext}" for i in range(n_parts)]

    print(f"Total frames: {n_frames}  →  splitting into {n_parts} parts")
    for i, (s, e) in enumerate(zip(boundaries, boundaries[1:])):
        print(f"  part{i+1}: frames {s}–{e-1}  ({e - s} frames)")

    out_zips = [zipfile.ZipFile(p, 'w', compression=zipfile.ZIP_STORED) for p in out_paths]
    try:
        with zipfile.ZipFile(source_file, 'r') as src_zip:
            for name in src_zip.namelist():
                key = name[:-4] if name.endswith('.npy') else name
                with src_zip.open(name) as f:
                    arr = np.load(io.BytesIO(f.read()))

                is_frame_array = key in FRAME_ARRAYS and arr.ndim >= 1 and arr.shape[0] == n_frames

                for i, (z, s, e) in enumerate(zip(out_zips, boundaries, boundaries[1:])):
                    chunk = arr[s:e] if is_frame_array else arr
                    buf = io.BytesIO()
                    np.save(buf, chunk)
                    z.writestr(name, buf.getvalue())

                del arr
    finally:
        for z in out_zips:
            z.close()

    for path in out_paths:
        print(f"  {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")

    return out_paths


if __name__ == "__main__":
    split_npz(SOURCE, N_PARTS)
