"""Project-relative locations for data, videos and figures (all inside bachelor_thesis/)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')                 # .npz simulation output
VIS_DIR = os.path.join(ROOT, 'visualizations')        # mp4 / gif animations
FIG_DIR = os.path.join(ROOT, 'figures')               # analysis plots (png)


def out_path(path):
    """Create the parent directory of an output file and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
