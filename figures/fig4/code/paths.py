"""Project-relative locations for data, videos and figures (data and videos on the external drive, figures inside bachelor_thesis/)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DIR = 'E:/BachelorThesisData'                # external drive holding the original data
DATA_DIR = EXTERNAL_DIR + '/data'                     # .npz simulation output
VIS_DIR = EXTERNAL_DIR + '/visualizations'            # mp4 / gif animations
FIG_DIR = ROOT                                        # analysis plots (png), written into figures/fig4/


def out_path(path):
    """Create the parent directory of an output file and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
