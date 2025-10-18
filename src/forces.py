import numpy as np
from numba import njit, int64

from .physics_utils import compute_minimum_distance, point_to_segment_distance


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
def compute_wall_forces(pos, radius, walls, stiffness):
    """Compute repulsive forces from all walls on a particle/payload.

    For each wall:
    1. Calculate distance from particle center to wall segment
    2. If distance < radius: particle is colliding with wall
    3. Apply force: F = stiffness * (radius - distance) * normal_direction

    Args:
        pos: np.ndarray [x, y], particle/payload position
        radius: float, particle/payload radius
        walls: np.ndarray of shape (n_walls, 4) with [x1, y1, x2, y2] per wall
        stiffness: float, wall stiffness (same as particle stiffness)

    Returns:
        force: np.ndarray [fx, fy], total force from all walls
    """
    force = np.zeros(2)
    n_walls = walls.shape[0]

    for w in range(n_walls):
        # Get wall segment endpoints
        x1, y1, x2, y2 = walls[w, 0], walls[w, 1], walls[w, 2], walls[w, 3]

        # Calculate distance from particle to wall segment
        distance, closest_x, closest_y = point_to_segment_distance(pos[0], pos[1], x1, y1, x2, y2)

        # Check for collision
        if distance < radius:
            overlap = radius - distance

            # Calculate normal vector (from wall toward particle)
            if distance > 1e-10:
                # Normal direction: from closest point on wall toward particle center
                normal_x = (pos[0] - closest_x) / distance
                normal_y = (pos[1] - closest_y) / distance
            else:
                # Particle exactly on wall - use perpendicular to wall direction
                wall_dx = x2 - x1
                wall_dy = y2 - y1
                wall_len = np.sqrt(wall_dx*wall_dx + wall_dy*wall_dy)

                if wall_len > 1e-10:
                    # Perpendicular vector (rotate 90 degrees)
                    normal_x = -wall_dy / wall_len
                    normal_y = wall_dx / wall_len
                else:
                    # Degenerate wall, push in arbitrary direction
                    normal_x = 1.0
                    normal_y = 0.0

            # Apply repulsive force
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
        cell_x = int(positions[i, 0] / cell_size) # , n_cells - 1
        cell_y = int(positions[i, 1] / cell_size) # , n_cells - 1

        list_next[i] = head[cell_x, cell_y]
        head[cell_x, cell_y] = i

    return head, list_next, n_cells
