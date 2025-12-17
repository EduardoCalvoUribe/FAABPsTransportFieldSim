import numpy as np
from numba import njit, int64

from .physics_utils import compute_minimum_distance, point_to_segment_distance, point_to_curve_distance


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
def compute_hollow_payload_force(pos_particle, pos_payload, particle_radius, inner_radius, outer_radius, inner_offset, stiffness, box_size):
    """Compute repulsive force between a particle and a hollow (ring-shaped) payload.

    The hollow payload is a ring with inner_radius and outer_radius.
    The inner circle can be offset from the outer circle center.
    - Particles inside the hollow center can push outward against the inner surface
    - Particles outside can push inward against the outer surface

    Args:
        pos_particle: np.ndarray [x, y], particle position
        pos_payload: np.ndarray [x, y], hollow payload center position (outer circle center)
        particle_radius: float, particle radius
        inner_radius: float, inner radius of the hollow payload (the hole)
        outer_radius: float, outer radius of the hollow payload
        inner_offset: np.ndarray [x, y], offset of inner circle center from outer circle center
        stiffness: float, force stiffness
        box_size: float, simulation box size for periodic boundaries

    Returns:
        force: np.ndarray [fx, fy], force on the particle (negate for payload)
    """
    # Compute inner circle center position
    pos_inner_center = pos_payload + inner_offset

    # Vector from outer payload center to particle (with periodic boundaries)
    r_vec_outer = compute_minimum_distance(pos_payload, pos_particle, box_size)
    dist_outer = np.sqrt(np.sum(r_vec_outer**2))

    # Vector from inner circle center to particle (with periodic boundaries)
    r_vec_inner = compute_minimum_distance(pos_inner_center, pos_particle, box_size)
    dist_inner = np.sqrt(np.sum(r_vec_inner**2))

    force = np.zeros(2)

    # Check collision with inner surface (based on distance to inner circle center)
    if dist_inner < 1e-10:
        # Particle at inner center - only push if it overlaps with inner wall
        if particle_radius > inner_radius:
            overlap = particle_radius - inner_radius
            force += np.array([-stiffness * overlap, 0.0])
    elif dist_inner < inner_radius:
        # Particle center is inside the hollow region
        if dist_inner + particle_radius > inner_radius:
            # Collision with inner surface
            overlap = (dist_inner + particle_radius) - inner_radius
            r_hat_inner = r_vec_inner / dist_inner
            force_magnitude = stiffness * overlap
            force += -force_magnitude * r_hat_inner  # Push toward inner center (away from inner wall)
    elif dist_inner <= inner_radius + particle_radius:
        # Particle center is outside inner circle but particle edge might overlap
        # This case: particle is in the ring material or touching from outside
        if dist_inner - particle_radius < inner_radius:
            # Particle overlaps with inner wall from outside (inside ring material)
            overlap = inner_radius - (dist_inner - particle_radius)
            r_hat_inner = r_vec_inner / dist_inner
            force_magnitude = stiffness * overlap
            force += -force_magnitude * r_hat_inner  # Push toward inner center

    # Check collision with outer surface (based on distance to outer circle center)
    if dist_outer > 1e-10:
        r_hat_outer = r_vec_outer / dist_outer

        if dist_outer > outer_radius:
            # Particle center is outside the ring
            if dist_outer - particle_radius < outer_radius:
                # Collision with outer surface
                overlap = outer_radius - (dist_outer - particle_radius)
                force_magnitude = stiffness * overlap
                force += force_magnitude * r_hat_outer  # Push away from outer center
        elif dist_outer >= outer_radius - particle_radius:
            # Particle center is inside but particle edge might overlap outer wall
            if dist_outer + particle_radius > outer_radius:
                # Particle overlaps with outer wall from inside (inside ring material)
                overlap = (dist_outer + particle_radius) - outer_radius
                force_magnitude = stiffness * overlap
                force += -force_magnitude * r_hat_outer  # Push toward outer center

    return force


@njit(fastmath=True)
def compute_wall_forces(pos, radius, walls, stiffness):
    """Compute repulsive forces from all walls on a particle/payload.

    For each wall:
    1. Calculate distance from particle center to wall segment/curve
    2. If distance < radius: particle is colliding with wall
    3. Apply force: F = stiffness * (radius - distance) * normal_direction

    Args:
        pos: np.ndarray [x, y], particle/payload position
        radius: float, particle/payload radius
        walls: np.ndarray of shape (n_walls, 5) with [x1, y1, x2, y2, c] per wall
        stiffness: float, wall stiffness (same as particle stiffness)

    Returns:
        force: np.ndarray [fx, fy], total force from all walls
    """
    force = np.zeros(2)
    n_walls = walls.shape[0]

    for w in range(n_walls):
        # Get wall segment endpoints and curvature
        x1, y1, x2, y2, c = walls[w, 0], walls[w, 1], walls[w, 2], walls[w, 3], walls[w, 4]

        # Calculate distance from particle to wall curve
        distance, closest_x, closest_y = point_to_curve_distance(pos[0], pos[1], x1, y1, x2, y2, c)

        # Check for collision
        if distance < radius:
            overlap = radius - distance

            # Calculate normal vector (from wall toward particle)
            if distance > 1e-10:
                # Normal direction: from closest point on wall toward particle center
                normal_x = (pos[0] - closest_x) / distance
                normal_y = (pos[1] - closest_y) / distance
            else:
                # Particle exactly on wall
                # For curved walls, we need to calculate the tangent at the closest point
                if abs(c) < 1e-10:
                    # Straight wall - use perpendicular to wall direction
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
                else:
                    # Curved wall - calculate normal from arc center
                    dx = x2 - x1
                    dy = y2 - y1
                    distance_endpoints = np.sqrt(dx*dx + dy*dy)
                    mid_x = (x1 + x2) / 2.0
                    mid_y = (y1 + y2) / 2.0
                    perp_x = -dy / distance_endpoints
                    perp_y = dx / distance_endpoints
                    radius_arc = distance_endpoints / (2.0 * abs(c))
                    h = np.sqrt(radius_arc*radius_arc - (distance_endpoints/2.0)*(distance_endpoints/2.0))

                    if c > 0:
                        center_x = mid_x + h * perp_x
                        center_y = mid_y + h * perp_y
                    else:
                        center_x = mid_x - h * perp_x
                        center_y = mid_y - h * perp_y

                    # Normal points away from center
                    to_particle_x = pos[0] - center_x
                    to_particle_y = pos[1] - center_y
                    norm_len = np.sqrt(to_particle_x*to_particle_x + to_particle_y*to_particle_y)

                    if norm_len > 1e-10:
                        normal_x = to_particle_x / norm_len
                        normal_y = to_particle_y / norm_len
                    else:
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
