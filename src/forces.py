import numpy as np
from numba import njit, int64

from .physics_utils import compute_minimum_distance, point_to_curve_distance


##########################
# Wall spatial index     #
##########################

def build_wall_spatial_index(walls, box_size, cell_size, margin):
    """Build a Compressed Sparse Row (CSR) spatial index mapping grid cells to nearby wall IDs.

    Each wall's bounding box is expanded by 'margin' before insertion into the
    grid.  Setting margin >= the largest expected interaction distance (e.g.
    payload_radius) means that a force lookup only needs to query the single
    cell that contains the querying position — no neighbourhood needed.

    Args:
        walls:     np.ndarray shape (n_walls, 5), columns [x1,y1,x2,y2,c]
        box_size:  float, simulation box side length
        cell_size: float, spatial-index grid cell size
        margin:    float, expand each wall bbox by this amount on every side

    Returns:
        offsets:  int64 array (n_cells**2 + 1,), CSR row offsets
        indices:  int64 array, wall IDs stored per cell (CSR values)
        n_cells:  int, cells per side (square grid)
    """
    n_walls = walls.shape[0]
    n_cells = max(1, int(np.floor(box_size / cell_size)))

    pairs = []  # (flat_cell_id, wall_id)
    for w in range(n_walls):
        x1, y1, x2, y2 = walls[w, 0], walls[w, 1], walls[w, 2], walls[w, 3]
        c = walls[w, 4]

        if abs(c) < 1e-9:
            # Straight segment: axis-aligned bbox
            bx_min, bx_max = min(x1, x2), max(x1, x2)
            by_min, by_max = min(y1, y2), max(y1, y2)
        else:
            # Circular arc: compute center and radius, use full circle bbox
            chx, chy = x2 - x1, y2 - y1
            chord_len = np.sqrt(chx ** 2 + chy ** 2)
            R = chord_len / (2.0 * abs(c))
            half_chord = chord_len / 2.0
            h_sq = max(0.0, R ** 2 - half_chord ** 2)
            h = np.sqrt(h_sq)
            cw_perp_x = chy / chord_len
            cw_perp_y = -chx / chord_len
            sc = 1.0 if c > 0.0 else -1.0
            cx_arc = (x1 + x2) * 0.5 + sc * h * cw_perp_x
            cy_arc = (y1 + y2) * 0.5 + sc * h * cw_perp_y
            bx_min, bx_max = cx_arc - R, cx_arc + R
            by_min, by_max = cy_arc - R, cy_arc + R

        bx_min -= margin
        bx_max += margin
        by_min -= margin
        by_max += margin

        gx_min = max(0, int(np.floor(bx_min / cell_size)))
        gx_max = min(n_cells - 1, int(np.floor(bx_max / cell_size)))
        gy_min = max(0, int(np.floor(by_min / cell_size)))
        gy_max = min(n_cells - 1, int(np.floor(by_max / cell_size)))

        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                pairs.append((gy * n_cells + gx, w))

    pairs.sort()

    n_total = n_cells * n_cells
    offsets = np.zeros(n_total + 1, dtype=np.int64)
    for cell_id, _ in pairs:
        offsets[cell_id + 1] += 1
    np.cumsum(offsets, out=offsets)

    indices_arr = np.empty(len(pairs), dtype=np.int64)
    write_pos = offsets[:-1].copy()
    for cell_id, w_id in pairs:
        indices_arr[write_pos[cell_id]] = w_id
        write_pos[cell_id] += 1

    return offsets, indices_arr, n_cells


##########################
# Force computation      #
##########################

@njit(fastmath=True)
def compute_repulsive_force(pos_i, pos_j, radius_i, radius_j, stiffness, box_size):
    """Compute repulsive force between two particles.

    Implements the equation:
    f_ij = { S_0 * (a+b-r_ij) * r_hat_ij, if r_ij <= a+b
           { 0,                           otherwise

    where:
    - S_0 is the stiffness
    - a, b are the particle radii
    - r_ij is the distance between particles
    - r_hat_ij is the unit vector from particle i to j
    """
    # r_ij = pos_j - pos_i
    r_ij = compute_minimum_distance(pos_i, pos_j, box_size)

    dist = np.sqrt(np.sum(r_ij**2))

    if dist < 1e-10:
        r_ij = np.array([1e-5, 1e-5])
        dist = np.sqrt(np.sum(r_ij**2))

    r_hat = r_ij / dist

    sum_radii = radius_i + radius_j

    if dist < sum_radii:
        # Force magnitude: S_0 * (a+b-r_ij)
        force_magnitude = stiffness * (sum_radii - dist)

        # Force direction: -r_hat
        return -force_magnitude * r_hat

    # No force if particles don't overlap
    return np.zeros(2)

@njit(fastmath=True)
def compute_wall_forces_indexed(pos, radius, walls,
                                wall_grid_offsets, wall_grid_indices,
                                n_wall_cells, wall_cell_size, stiffness):
    """Compute wall forces using the spatial index.

    Queries only the single cell that contains `pos`.  This is correct because
    the index was built with margin >= radius, so all walls that could be within
    `radius` of `pos` are guaranteed to appear in that cell.
    """
    force = np.zeros(2)
    if walls.shape[0] == 0:
        return force

    cell_x = min(int(pos[0] / wall_cell_size), n_wall_cells - 1)
    cell_y = min(int(pos[1] / wall_cell_size), n_wall_cells - 1)
    if cell_x < 0:
        cell_x = 0
    if cell_y < 0:
        cell_y = 0

    cell_id = cell_y * n_wall_cells + cell_x
    start = wall_grid_offsets[cell_id]
    end = wall_grid_offsets[cell_id + 1]

    for k in range(start, end):
        w = wall_grid_indices[k]
        x1 = walls[w, 0]
        y1 = walls[w, 1]
        x2 = walls[w, 2]
        y2 = walls[w, 3]
        c = walls[w, 4]

        distance, closest_x, closest_y = point_to_curve_distance(pos[0], pos[1], x1, y1, x2, y2, c)

        if distance < radius:
            overlap = radius - distance
            if distance > 1e-10:
                normal_x = (pos[0] - closest_x) / distance
                normal_y = (pos[1] - closest_y) / distance
            else:
                wall_dx = x2 - x1
                wall_dy = y2 - y1
                wall_len = np.sqrt(wall_dx * wall_dx + wall_dy * wall_dy)
                if wall_len > 1e-10:
                    normal_x = -wall_dy / wall_len
                    normal_y = wall_dx / wall_len
                else:
                    normal_x = 1.0
                    normal_y = 0.0
            force_magnitude = stiffness * overlap
            force[0] += force_magnitude * normal_x
            force[1] += force_magnitude * normal_y

    return force


@njit(fastmath=True)
def compute_wall_forces(pos, radius, walls, stiffness):
    """Compute repulsive forces from all walls on a particle/payload.

    For each wall:
    1. Calculate distance from particle center to wall (segment or arc)
    2. If distance < radius: particle is colliding with wall
    3. Apply force: F = stiffness * (radius - distance) * normal_direction

    Args:
        pos: np.ndarray [x, y], particle/payload position
        radius: float, particle/payload radius
        walls: np.ndarray of shape (n_walls, 5) with [x1, y1, x2, y2, c] per wall,
               where c is the arc curvature (0 = straight segment).
        stiffness: float, wall stiffness (same as particle stiffness)

    Returns:
        force: np.ndarray [fx, fy], total force from all walls
    """
    force = np.zeros(2)
    n_walls = walls.shape[0]

    for w in range(n_walls):
        x1, y1, x2, y2, c = walls[w, 0], walls[w, 1], walls[w, 2], walls[w, 3], walls[w, 4]

        distance, closest_x, closest_y = point_to_curve_distance(pos[0], pos[1], x1, y1, x2, y2, c)

        if distance < radius:
            overlap = radius - distance

            if distance > 1e-10:
                normal_x = (pos[0] - closest_x) / distance
                normal_y = (pos[1] - closest_y) / distance
            else:
                # Particle exactly on wall — push perpendicular to wall chord
                wall_dx = x2 - x1
                wall_dy = y2 - y1
                wall_len = np.sqrt(wall_dx * wall_dx + wall_dy * wall_dy)
                if wall_len > 1e-10:
                    normal_x = -wall_dy / wall_len
                    normal_y = wall_dx / wall_len
                else:
                    normal_x = 1.0
                    normal_y = 0.0

            force_magnitude = stiffness * overlap
            force[0] += force_magnitude * normal_x
            force[1] += force_magnitude * normal_y

    return force

@njit(fastmath=True)
def create_cell_list(positions, box_size, cell_size, n_particles):
    """Create a cell list for efficient neighbor searching. Uses a linked list implementation"""
    n_cells = int(np.floor(box_size / cell_size)) # cell_size is at least 2*max_radius (particle-particle max interaction)

    # Initialize cell lists with -1 (empty indicator)
    head = np.ones((n_cells, n_cells), dtype=int64) * -1  # First particle in each cell # n_cells * n_cells
    list_next = np.ones(n_particles, dtype=int64) * -1   # Next particle in same cell
    # fails to work without int64 for some reason

    for i in range(n_particles):
        cell_x = min(int(positions[i, 0] / cell_size), n_cells - 1)
        cell_y = min(int(positions[i, 1] / cell_size), n_cells - 1)

        list_next[i] = head[cell_x, cell_y]
        head[cell_x, cell_y] = i

    return head, list_next, n_cells
