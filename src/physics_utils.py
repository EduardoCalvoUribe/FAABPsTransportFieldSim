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


##############################
# Arc geometry helpers       #
##############################

@njit(fastmath=True)
def _point_on_arc(qx, qy, cx, cy, p1x, p1y, p2x, p2y):
    """Check if Q (on the circle) lies on the minor arc from P1 to P2 with center C.

    Convention: the arc is the shorter one (subtending ≤ 180° at center C).
    Returns True if Q is on the arc (including endpoints), False otherwise.
    """
    v1x, v1y = p1x - cx, p1y - cy
    v2x, v2y = p2x - cx, p2y - cy
    vqx, vqy = qx - cx, qy - cy
    
    # cross(v1, v2) ≥ 0 → minor arc goes CCW from P1 to P2; < 0 → CW
    cross12 = v1x * v2y - v1y * v2x
    cross1q = v1x * vqy - v1y * vqx
    crossq2 = vqx * v2y - vqy * v2x

    if cross12 >= 0.0:
        # Minor arc goes CCW: Q must be CCW of P1 and CW of P2
        return cross1q >= -1e-9 and crossq2 >= -1e-9
    else:
        # Minor arc goes CW: Q must be CW of P1 and CCW of P2
        return cross1q <= 1e-9 and crossq2 <= 1e-9


@njit(fastmath=True)
def _arc_center(x1, y1, x2, y2, c):
    """Compute the center of the circular arc defined by endpoints and curvature c.

    c = chord_length / (2 * radius), with sign determining which side the arc bulges:
      c > 0: arc bulges left of p1→p2 direction
      c < 0: arc bulges right

    Returns (cx, cy, R): arc center coordinates and radius.
    """
    chx, chy = x2 - x1, y2 - y1
    chord_len = math.sqrt(chx * chx + chy * chy)
    R = chord_len / (2.0 * abs(c))
    half_chord = chord_len / 2.0
    h_sq = R * R - half_chord * half_chord
    h = math.sqrt(h_sq) if h_sq > 0.0 else 0.0

    # CW perpendicular to chord direction: rotate chord 90° clockwise = (chy, -chx)/chord_len
    cw_perp_x = chy / chord_len
    cw_perp_y = -chx / chord_len

    mid_x = (x1 + x2) * 0.5
    mid_y = (y1 + y2) * 0.5

    sc = 1.0 if c > 0.0 else -1.0
    cx = mid_x + sc * h * cw_perp_x
    cy = mid_y + sc * h * cw_perp_y
    return cx, cy, R


@njit(fastmath=True)
def point_to_curve_distance(px, py, x1, y1, x2, y2, c):
    """Minimum distance from point (px,py) to a wall defined by two endpoints and curvature c.

    When c ≈ 0: the wall is a straight segment (delegates to point_to_segment_distance).
    When c ≠ 0: the wall is a circular arc.

    Returns (distance, closest_x, closest_y).
    """
    if abs(c) < 1e-9:
        return point_to_segment_distance(px, py, x1, y1, x2, y2)

    chx, chy = x2 - x1, y2 - y1
    chord_len = math.sqrt(chx * chx + chy * chy)
    if chord_len < 1e-10:
        return point_to_segment_distance(px, py, x1, y1, x2, y2)

    cx, cy, R = _arc_center(x1, y1, x2, y2, c)

    # Project particle radially onto the circle
    dx, dy = px - cx, py - cy
    dist_to_center = math.sqrt(dx * dx + dy * dy)

    if dist_to_center < 1e-10:
        # Particle at center; fall back to nearer endpoint
        d1 = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        d2 = math.sqrt((px - x2) ** 2 + (py - y2) ** 2)
        if d1 <= d2:
            return d1, x1, y1
        return d2, x2, y2

    # Projection on circle surface (outward from center)
    proj_x = cx + R * dx / dist_to_center
    proj_y = cy + R * dy / dist_to_center

    if _point_on_arc(proj_x, proj_y, cx, cy, x1, y1, x2, y2):
        distance = abs(dist_to_center - R)
        # proj_x/proj_y is the actual closest point on the arc.
        # (pos - proj)/distance is outward from center when outside the circle,
        # and inward (toward center) when inside — correctly repelling from both sides.
        return distance, proj_x, proj_y

    # Projection not on arc: closest is one of the endpoints
    d1 = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    d2 = math.sqrt((px - x2) ** 2 + (py - y2) ** 2)
    if d1 <= d2:
        return d1, x1, y1
    return d2, x2, y2


@njit(fastmath=True)
def line_intersects_arc(ax, ay, bx, by, x1, y1, x2, y2, c):
    """Check if line segment A→B intersects a wall arc defined by endpoints and curvature c.

    When c ≈ 0: delegates to line_segments_intersect.
    When c ≠ 0: finds line-circle intersections and checks if they fall on the arc.
    """
    if abs(c) < 1e-9:
        return line_segments_intersect(ax, ay, bx, by, x1, y1, x2, y2)

    chx, chy = x2 - x1, y2 - y1
    chord_len = math.sqrt(chx * chx + chy * chy)
    if chord_len < 1e-10:
        return False

    cx, cy, R = _arc_center(x1, y1, x2, y2, c)

    # Quadratic coefficients for line-circle intersection
    # L(t) = (ax,ay) + t*(dx,dy), |L(t) - (cx,cy)|^2 = R^2
    dx, dy = bx - ax, by - ay
    fx, fy = ax - cx, ay - cy

    a_q = dx * dx + dy * dy
    if a_q < 1e-20:
        return False  # Degenerate segment (point)

    b_q = 2.0 * (fx * dx + fy * dy)
    c_q = fx * fx + fy * fy - R * R

    disc = b_q * b_q - 4.0 * a_q * c_q
    if disc < 0.0:
        return False

    sqrt_disc = math.sqrt(disc)

    t1 = (-b_q - sqrt_disc) / (2.0 * a_q)
    if 0.0 <= t1 <= 1.0:
        ix = ax + t1 * dx
        iy = ay + t1 * dy
        if _point_on_arc(ix, iy, cx, cy, x1, y1, x2, y2):
            return True

    t2 = (-b_q + sqrt_disc) / (2.0 * a_q)
    if 0.0 <= t2 <= 1.0:
        ix = ax + t2 * dx
        iy = ay + t2 * dy
        if _point_on_arc(ix, iy, cx, cy, x1, y1, x2, y2):
            return True

    return False


##############################
# Straight-wall helpers      #
##############################

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
def line_intersects_any_wall_indexed(p1_x, p1_y, p2_x, p2_y, walls,
                                     wall_grid_offsets, wall_grid_indices,
                                     n_wall_cells, wall_cell_size):
    """Check wall intersection using the spatial index.

    Iterates only over cells whose bounding boxes overlap the segment's axis-
    aligned bounding box.  The same wall may appear in more than one cell and
    be tested twice; this is safe because the test is idempotent and we exit
    immediately on the first intersection found.

    Segment coordinates may lie outside [0, box_size]; out-of-range cell
    indices are clamped to [0, n_wall_cells-1], which is correct because walls
    exist only inside the box.
    """
    if walls.shape[0] == 0:
        return False

    min_x = p1_x if p1_x < p2_x else p2_x
    max_x = p1_x if p1_x > p2_x else p2_x
    min_y = p1_y if p1_y < p2_y else p2_y
    max_y = p1_y if p1_y > p2_y else p2_y

    gx_min = int(min_x / wall_cell_size)
    gx_max = int(max_x / wall_cell_size)
    gy_min = int(min_y / wall_cell_size)
    gy_max = int(max_y / wall_cell_size)

    if gx_min < 0:
        gx_min = 0
    if gx_max >= n_wall_cells:
        gx_max = n_wall_cells - 1
    if gy_min < 0:
        gy_min = 0
    if gy_max >= n_wall_cells:
        gy_max = n_wall_cells - 1

    for gx in range(gx_min, gx_max + 1):
        for gy in range(gy_min, gy_max + 1):
            cell_id = gy * n_wall_cells + gx
            start = wall_grid_offsets[cell_id]
            end = wall_grid_offsets[cell_id + 1]
            for k in range(start, end):
                w = wall_grid_indices[k]
                if line_intersects_arc(p1_x, p1_y, p2_x, p2_y,
                                       walls[w, 0], walls[w, 1],
                                       walls[w, 2], walls[w, 3],
                                       walls[w, 4]):
                    return True
    return False


@njit(fastmath=True)
def particles_separated_by_wall_periodic_indexed(pos_i, pos_j, walls, box_size,
                                                  wall_grid_offsets, wall_grid_indices,
                                                  n_wall_cells, wall_cell_size):
    """Check wall separation along periodic shortest path using the spatial index."""
    r_ij = compute_minimum_distance(pos_i, pos_j, box_size)
    pos_j_periodic = pos_i + r_ij
    return line_intersects_any_wall_indexed(
        pos_i[0], pos_i[1], pos_j_periodic[0], pos_j_periodic[1],
        walls, wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size
    )


@njit(fastmath=True)
def line_intersects_any_wall(p1_x, p1_y, p2_x, p2_y, walls):
    """Check if line segment (p1, p2) intersects any wall.

    Args:
        p1_x, p1_y: Start point coordinates
        p2_x, p2_y: End point coordinates
        walls: np.ndarray of shape (n_walls, 5) with [x1, y1, x2, y2, c] per wall,
               where c is the arc curvature (0 = straight segment).

    Returns:
        bool: True if line intersects any wall, False otherwise
    """
    n_walls = walls.shape[0]
    for i in range(n_walls):
        if line_intersects_arc(p1_x, p1_y, p2_x, p2_y,
                               walls[i, 0], walls[i, 1],
                               walls[i, 2], walls[i, 3],
                               walls[i, 4]):
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
