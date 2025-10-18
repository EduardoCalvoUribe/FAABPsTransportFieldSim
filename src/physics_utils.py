import numpy as np
import math
from numba import njit


##############################
# Physics utility functions  #
##############################

@njit(fastmath=True)
def normalize(v):
    """Normalize a vector to unit length."""
    norm = np.sqrt(np.sum(v**2))
    if norm > 0:
        return v / norm
    return v

@njit(fastmath=True)
def line_segments_intersect(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y):
    """Check if line segment (p1, p2) intersects with line segment (p3, p4).

    Uses the cross-product method to determine intersection.
    Returns True if segments intersect, False otherwise.

    Args:
        p1_x, p1_y: First point of segment 1
        p2_x, p2_y: Second point of segment 1
        p3_x, p3_y: First point of segment 2
        p4_x, p4_y: Second point of segment 2
    """
    # Direction vectors
    d1_x = p2_x - p1_x
    d1_y = p2_y - p1_y
    d2_x = p4_x - p3_x
    d2_y = p4_y - p3_y

    # Cross product of direction vectors
    cross = d1_x * d2_y - d1_y * d2_x

    # If cross product is zero, lines are parallel or coincident
    if abs(cross) < 1e-10:
        return False

    # Vector from p1 to p3
    v_x = p3_x - p1_x
    v_y = p3_y - p1_y

    # Calculate parameters for intersection point
    t1 = (v_x * d2_y - v_y * d2_x) / cross
    t2 = (v_x * d1_y - v_y * d1_x) / cross

    # Check if intersection point lies on both segments (0 <= t <= 1)
    if 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0:
        return True

    return False

@njit(fastmath=True)
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Calculate the minimum distance from a point (px, py) to a line segment (x1,y1)-(x2,y2).

    Returns:
        distance: minimum distance from point to segment
        closest_x, closest_y: coordinates of closest point on segment
    """
    # Vector from segment start to point
    dx = px - x1
    dy = py - y1

    # Segment vector
    sx = x2 - x1
    sy = y2 - y1

    # Segment length squared
    seg_len_sq = sx*sx + sy*sy

    if seg_len_sq < 1e-10:
        # Degenerate segment (both endpoints same)
        dist = np.sqrt(dx*dx + dy*dy)
        return dist, x1, y1

    # Project point onto line (parameter t)
    t = (dx*sx + dy*sy) / seg_len_sq

    # Clamp t to [0, 1] to stay on segment
    t = max(0.0, min(1.0, t))

    # Closest point on segment
    closest_x = x1 + t * sx
    closest_y = y1 + t * sy

    # Distance from point to closest point
    dist_x = px - closest_x
    dist_y = py - closest_y
    dist = np.sqrt(dist_x*dist_x + dist_y*dist_y)

    return dist, closest_x, closest_y

@njit(fastmath=True)
def line_intersects_any_wall(p1_x, p1_y, p2_x, p2_y, walls):
    """Check if line segment (p1, p2) intersects any wall.

    Args:
        p1_x, p1_y: Start point coordinates
        p2_x, p2_y: End point coordinates
        walls: np.ndarray of shape (n_walls, 4) with [x1, y1, x2, y2] per wall

    Returns:
        bool: True if line intersects any wall, False otherwise
    """
    n_walls = walls.shape[0]
    for i in range(n_walls):
        if line_segments_intersect(p1_x, p1_y, p2_x, p2_y,
                                   walls[i, 0], walls[i, 1],
                                   walls[i, 2], walls[i, 3]):
            return True
    return False

@njit(fastmath=True)
def particles_separated_by_wall(pos_i, pos_j, walls):
    """Check if a wall blocks the line segment between two particles.

    Args:
        pos_i: np.ndarray [x, y], position of particle i
        pos_j: np.ndarray [x, y], position of particle j
        walls: np.ndarray of shape (n_walls, 4)

    Returns:
        bool: True if any wall separates the particles, False otherwise
    """
    return line_intersects_any_wall(pos_i[0], pos_i[1], pos_j[0], pos_j[1], walls)

@njit(fastmath=True)
def particles_separated_by_wall_periodic(pos_i, pos_j, walls, box_size):
    """Check if a wall blocks the periodic shortest path between two particles.

    With periodic boundaries, there are multiple paths between particles (wrapping around edges).
    This checks if a wall blocks the shortest periodic path.

    Args:
        pos_i: np.ndarray [x, y], position of particle i
        pos_j: np.ndarray [x, y], position of particle j
        walls: np.ndarray of shape (n_walls, 4)
        box_size: float, size of the simulation box

    Returns:
        bool: True if any wall separates the particles along shortest periodic path
    """
    # Compute the periodic shortest displacement vector
    r_ij = compute_minimum_distance(pos_i, pos_j, box_size)

    # The actual endpoint following periodic shortest path
    pos_j_periodic = pos_i + r_ij

    # Check if any wall intersects this shortest path
    return line_intersects_any_wall(pos_i[0], pos_i[1], pos_j_periodic[0], pos_j_periodic[1], walls)

@njit(fastmath=True)
def compute_minimum_distance(pos_i, pos_j, box_size):
    """Compute minimum distance vector considering periodic boundaries."""
    r_ij = pos_j - pos_i

    r_ij = r_ij - box_size * np.round(r_ij / box_size)

    return r_ij
