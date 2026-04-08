import numpy as np
import math
from numba import njit, prange

from .physics_utils import (normalize, line_intersects_any_wall,
                            line_intersects_any_wall_indexed,
                            particles_separated_by_wall_periodic_indexed,
                            compute_minimum_distance,
                            particles_separated_by_wall, particles_separated_by_wall_periodic)
from .forces import compute_wall_forces, compute_wall_forces_indexed, create_cell_list


##########################
# Main physics functions #
##########################

@njit(fastmath=True, parallel=True)
def compute_all_forces(positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls,
                       wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size):
    """Compute all forces acting on particles and the payload"""
    particle_forces = np.zeros((n_particles, 2))  # Initialize force array for particles
    payload_force = np.zeros(2)  # Initialize force array for payload

    # Determine maximum interaction distance (for cell size) - for particle-particle forces
    # max_radius = np.max(radii)  # Takes maximum radius of all particles. (Because radius of particles is possibly heterogeneous)
    # cell_size = 2 * max_radius  # For particle-particle interactions (not payload-particle)

    # Create cell list (O(N)) - for particle-particle forces
    # head, list_next, n_cells = create_cell_list(positions, box_size, cell_size, n_particles)

    # Compute forces between particles and payload (O(N))
    for i in range(n_particles):
        pix = positions[i, 0]
        piy = positions[i, 1]
        pjx = payload_pos[0]
        pjy = payload_pos[1]

        dx = pjx - pix
        dy = pjy - piy

        half_box = 0.5 * box_size

        # Minimum-image convention
        if dx > half_box:
            dx -= box_size
        elif dx < -half_box:
            dx += box_size

        if dy > half_box:
            dy -= box_size
        elif dy < -half_box:
            dy += box_size

        dist2 = dx * dx + dy * dy
        sum_radii = radii[i] + payload_radius
        sum_radii2 = sum_radii * sum_radii

        # Early reject without sqrt
        if dist2 < sum_radii2:
            pos_j_periodic_x = pix + dx
            pos_j_periodic_y = piy + dy

            # Particle is close enough to potentially interact — check wall separation
            if not line_intersects_any_wall_indexed(
                pix, piy,
                pos_j_periodic_x, pos_j_periodic_y,
                walls, wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size
            ):
                if dist2 < 1e-20:
                    dx_force = 1e-5
                    dy_force = 1e-5
                    dist2_force = dx_force * dx_force + dy_force * dy_force
                else:
                    dx_force = dx
                    dy_force = dy
                    dist2_force = dist2

                dist = math.sqrt(dist2_force)
                overlap = sum_radii - dist
                force_magnitude = stiffness * overlap
                inv_dist = 1.0 / dist

                fx = -force_magnitude * dx_force * inv_dist
                fy = -force_magnitude * dy_force * inv_dist

                particle_forces[i, 0] += fx
                particle_forces[i, 1] += fy
                payload_force[0] -= fx
                payload_force[1] -= fy

    # Compute forces between particles and walls (indexed: single-cell lookup per particle)
    for i in prange(n_particles):
        fx, fy = compute_wall_forces_indexed(
            positions[i], radii[i], walls,
            wall_grid_offsets, wall_grid_indices,
            n_wall_cells, wall_cell_size, stiffness
        )
        particle_forces[i, 0] += fx
        particle_forces[i, 1] += fy

    # Compute force between payload and walls (indexed: single-cell lookup)
    fx, fy = compute_wall_forces_indexed(
        payload_pos, payload_radius, walls,
        wall_grid_offsets, wall_grid_indices,
        n_wall_cells, wall_cell_size, stiffness
    )
    payload_force[0] += fx
    payload_force[1] += fy

    # Compute pairwise particle-particle interactions (O(N)) using cell list
    # NOTE:
    # If you uncomment this, do NOT use prange directly over particle pairs while
    # updating particle_forces[i] and particle_forces[j] in place, because that
    # creates write conflicts. Keep this serial unless you redesign it with
    # thread-local accumulation.
    #
    # for cell_x in range(n_cells):
    #     for cell_y in range(n_cells):
    #         i = head[cell_x, cell_y]
    #         while i != -1:
    #             pix = positions[i, 0]
    #             piy = positions[i, 1]
    #             ri = radii[i]
    #
    #             # Check this cell and neighboring cells
    #             for dcell_x in range(-1, 2):
    #                 neigh_x = (cell_x + dcell_x) % n_cells
    #                 for dcell_y in range(-1, 2):
    #                     neigh_y = (cell_y + dcell_y) % n_cells
    #
    #                     j = head[neigh_x, neigh_y]
    #                     while j != -1:
    #                         if j > i:
    #                             pjx = positions[j, 0]
    #                             pjy = positions[j, 1]
    #                             rj = radii[j]
    #
    #                             dx = pjx - pix
    #                             dy = pjy - piy
    #
    #                             half_box = 0.5 * box_size
    #
    #                             if dx > half_box:
    #                                 dx -= box_size
    #                             elif dx < -half_box:
    #                                 dx += box_size
    #
    #                             if dy > half_box:
    #                                 dy -= box_size
    #                             elif dy < -half_box:
    #                                 dy += box_size
    #
    #                             sum_radii = ri + rj
    #                             dist2 = dx * dx + dy * dy
    #
    #                             # Early reject before sqrt
    #                             if dist2 < sum_radii * sum_radii:
    #                                 if dist2 < 1e-20:
    #                                     dx_force = 1e-5
    #                                     dy_force = 1e-5
    #                                     dist2_force = dx_force * dx_force + dy_force * dy_force
    #                                 else:
    #                                     dx_force = dx
    #                                     dy_force = dy
    #                                     dist2_force = dist2
    #
    #                                 dist = math.sqrt(dist2_force)
    #                                 overlap = sum_radii - dist
    #                                 force_magnitude = stiffness * overlap
    #                                 inv_dist = 1.0 / dist
    #
    #                                 fx = -force_magnitude * dx_force * inv_dist
    #                                 fy = -force_magnitude * dy_force * inv_dist
    #
    #                                 particle_forces[i, 0] += fx
    #                                 particle_forces[i, 1] += fy
    #                                 particle_forces[j, 0] -= fx
    #                                 particle_forces[j, 1] -= fy
    #
    #                         j = list_next[j]
    #
    #             i = list_next[i]

    return particle_forces, payload_force


@njit(fastmath=True, parallel=True)
def update_orientation_vectors(orientations, forces, curvity, dt, rot_diffusion, n_particles):
    new_orientations = np.empty_like(orientations)
    """
    The torque is calculated as:
    torque = k * (n × F)
    (this is equivalent to k(e x (v x e)) from paper)

    Orientation update is:
    dn/dt = torque * (n × z) + noise

    Where:
    - n is the orientation vector
    - F is the net force
    - k is curvity
    - z is the unit vector pointing out of the 2D plane (implicitly used in the cross product calculation)
    """
    
    for i in prange(n_particles):
        ox = orientations[i, 0]
        oy = orientations[i, 1]
        fx = forces[i, 0]
        fy = forces[i, 1]

        # torque = k * (n x F) = k * (ox*fy - oy*fx)
        torque = curvity[i] * (ox * fy - oy * fx)

        # (n x z) = (-oy, ox)
        dx = torque * (-oy) * dt
        dy = torque * ox * dt

        if rot_diffusion[i] > 0.0:
            noise_mag = math.sqrt(2.0 * rot_diffusion[i] * dt)
            nx = np.random.normal(0.0, noise_mag)
            ny = np.random.normal(0.0, noise_mag)

            # project noise perpendicular to orientation
            dot = nx * ox + ny * oy
            dx += nx - dot * ox
            dy += ny - dot * oy

        newx = ox + dx
        newy = oy + dy

        norm = math.sqrt(newx * newx + newy * newy)
        if norm > 0.0:
            inv = 1.0 / norm
            new_orientations[i, 0] = newx * inv
            new_orientations[i, 1] = newy * inv
        else:
            new_orientations[i, 0] = ox
            new_orientations[i, 1] = oy

    return new_orientations



@njit(fastmath=True, parallel=True)
def nudge_orientations_toward_polarity(orientations, polarity, nudge_strength, n_particles):
    """Angularly nudge each particle's orientation toward its polarity vector.

    If angle between orientation and polarity <= 90° (dot >= 0): nudge toward polarity.
    If angle > 90° (dot < 0): nudge toward -polarity instead.

    The nudge rotates the orientation by exactly nudge_strength radians toward the target.
    """
    new_orientations = np.zeros_like(orientations)

    for i in prange(n_particles):
        e_x = orientations[i, 0]
        e_y = orientations[i, 1]

        dot = e_x * polarity[i, 0] + e_y * polarity[i, 1]

        # Choose target direction based on alignment
        if dot >= 0:
            t_x = polarity[i, 0]
            t_y = polarity[i, 1]
        else:
            t_x = -polarity[i, 0]
            t_y = -polarity[i, 1]

        # Perpendicular component of target relative to orientation: t - (t·e)*e
        dot_te = t_x * e_x + t_y * e_y
        perp_x = t_x - dot_te * e_x
        perp_y = t_y - dot_te * e_y

        perp_norm = math.sqrt(perp_x * perp_x + perp_y * perp_y)

        if perp_norm > 1e-10:
            perp_hat_x = perp_x / perp_norm
            perp_hat_y = perp_y / perp_norm

            # Rotate orientation by nudge_strength radians toward target
            cos_a = math.cos(nudge_strength)
            sin_a = math.sin(nudge_strength)
            new_orientations[i, 0] = cos_a * e_x + sin_a * perp_hat_x
            new_orientations[i, 1] = cos_a * e_y + sin_a * perp_hat_y
        else:
            new_orientations[i, 0] = e_x
            new_orientations[i, 1] = e_y

    return new_orientations


@njit(fastmath=True, parallel=True)
def compute_curvity_from_polarity(orientations, polarity, n_particles, max_curvity = 1, min_curvity = -1, mid_curvity = 0.5):
    """Compute curvity for all particles based on their polarity vectors.

    Curvity = -(e · p), where:
    - e is the particle's heading direction (orientation)
    - p is the particle's polarity vector
    """
    assert -1 <= min_curvity <= 1, "min_curvity must be in range [-1, 1]"
    assert -1 <= max_curvity <= 1, "max_curvity must be in range [-1, 1]"
    assert -1 <= mid_curvity <= 1, "mid_curvity must be in range [-1, 1]"
    if not min_curvity < mid_curvity < max_curvity:
        # Nudge values to ensure strict ordering
        epsilon = 1e-6
        if min_curvity >= mid_curvity:
            mid_curvity = min_curvity + epsilon
        if mid_curvity >= max_curvity:
            mid_curvity = mid_curvity - epsilon
    
    curvity = np.zeros(n_particles)

    for i in prange(n_particles):
        # Compute dot product: e · p
        dot_product = orientations[i, 0] * polarity[i, 0] + orientations[i, 1] * polarity[i, 1]

        # Piecewise linear mapping:
        # dot_product = 1 (aligned) → min_curvity
        # dot_product = 0 (perpendicular) → mid_curvity
        # dot_product = -1 (disaligned) → max_curvity
        if dot_product >= 0:
            # Linear interpolation from mid_curvity (at 0) to min_curvity (at 1)
            curvity[i] = mid_curvity + (min_curvity - mid_curvity) * dot_product
        else:
            # Linear interpolation from max_curvity (at -1) to mid_curvity (at 0)
            curvity[i] = max_curvity + (mid_curvity - max_curvity) * (dot_product + 1)

    return curvity

@njit(fastmath=True)
def has_line_of_sight(pos_i, goal_position, payload_pos, payload_radius, walls,
                      wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size):
    """Check if particle i has line of sight to goal (no walls or payload blocking).

    Returns True if line from particle to goal doesn't intersect with walls or payload circle.
    Does not use periodic boundaries.
    """
    x_i, y_i = pos_i
    x_goal, y_goal = goal_position
    x_p, y_p = payload_pos

    # Check if any wall blocks line of sight (indexed: only nearby walls checked)
    if line_intersects_any_wall_indexed(x_i, y_i, x_goal, y_goal, walls,
                                        wall_grid_offsets, wall_grid_indices,
                                        n_wall_cells, wall_cell_size):
        return False  # Wall blocks line of sight

    # Quick bounding box check
    min_x = min(x_i, x_goal) - 0.001
    max_x = max(x_i, x_goal) + 0.001
    min_y = min(y_i, y_goal) - 0.001
    max_y = max(y_i, y_goal) + 0.001

    # If payload is outside bounding box (plus radius), it can't intersect
    if (x_p - payload_radius > max_x or
        x_p + payload_radius < min_x or
        y_p - payload_radius > max_y or
        y_p + payload_radius < min_y):
        return True

    # Direction vector: particle to goal
    dx, dy = x_goal - x_i, y_goal - y_i

    # Vector from particle to payload center
    fx, fy = x_p - x_i, y_p - y_i

    # Coefficients of quadratic equation for line-circle intersection
    # Line: P(t) = (x_i, y_i) + t*(dx, dy), we want |P(t) - payload_center|^2 = radius^2
    a = dx**2 + dy**2
    b = -2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - payload_radius**2

    # Discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return True  # No intersection
    else:
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)

        # Check if intersection is between particle and goal (t in [0, 1])
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return False  # Intersection blocks line of sight
        else:
            return True



@njit(fastmath=True)
def compute_polarity_toward_minscore_pos(pos_i, neighbor_scores, neighbor_positions, box_size):
    """Component 2: Direction toward particle with lowest score.

    Finds particles with minimum score and computes direction toward their
    average position using periodic boundaries.

    Args:
        pos_i: current particle position
        neighbor_scores: scores of neighbor particles
        neighbor_positions: positions of neighbor particles
        box_size: simulation box size for periodic boundaries

    Returns:
        gradient_polarity: normalized direction toward min-score particles
    """
    min_score = min(neighbor_scores)

    # Find the particle(s) with minimum score
    min_score_indices = []
    for idx in range(len(neighbor_scores)):
        if neighbor_scores[idx] == min_score:
            min_score_indices.append(idx)

    # Point toward the average position of min-score particles (using periodic relative positions)
    avg_relative_pos = np.zeros(2)
    for idx in min_score_indices:
        # Get relative position accounting for periodic boundaries
        r_ij = compute_minimum_distance(pos_i, neighbor_positions[idx], box_size)
        avg_relative_pos += r_ij
    avg_relative_pos = avg_relative_pos / len(min_score_indices)

    # Direction from current particle to target
    gradient_polarity = avg_relative_pos
    norm_gradient = np.sqrt(np.sum(gradient_polarity**2))
    if norm_gradient > 0:
        gradient_polarity = gradient_polarity / norm_gradient
    else:
        gradient_polarity = np.array([0.0, 0.0])

    return gradient_polarity


@njit(fastmath=True)
def compute_polarity_toward_minscore_ang(pos_i, neighbor_scores, neighbor_positions, box_size):
    """ALTERNATIVE Component 2: Average angle toward min-score neighbors.

    Instead of pointing to average position, this computes the average of unit
    vectors pointing toward each min-score neighbor separately.

    Args:
        pos_i: current particle position
        neighbor_scores: scores of neighbor particles
        neighbor_positions: positions of neighbor particles
        box_size: simulation box size for periodic boundaries

    Returns:
        gradient_polarity: normalized average direction toward min-score particles
    """
    min_score = min(neighbor_scores)

    # Find the particle(s) with minimum score
    min_score_indices = []
    for idx in range(len(neighbor_scores)):
        if neighbor_scores[idx] == min_score:
            min_score_indices.append(idx)

    # Compute average angle from current particle to min-score neighbors
    total_x = 0.0
    total_y = 0.0
    for idx in min_score_indices:
        # Direction from current particle to this neighbor (PERIODIC)
        r_ij = compute_minimum_distance(pos_i, neighbor_positions[idx], box_size)
        dist = np.sqrt(np.sum(r_ij**2))
        if dist > 0:
            # Unit vector toward this neighbor
            unit_x = r_ij[0] / dist
            unit_y = r_ij[1] / dist
            total_x += unit_x
            total_y += unit_y

    # Average the unit vectors (this gives average angle)
    gradient_polarity = np.array([total_x, total_y])
    norm_gradient = np.sqrt(gradient_polarity[0]**2 + gradient_polarity[1]**2)
    if norm_gradient > 0:
        gradient_polarity = gradient_polarity / norm_gradient
    else:
        gradient_polarity = np.array([0.0, 0.0])

    return gradient_polarity


@njit(fastmath=True)
def point_polarity_to_goal(pos_i, goal_position, positions, particle_scores, i,
                                r, box_size, head, list_next, n_cells,
                                payload_pos, payload_radius, walls,
                                wall_grid_offsets, wall_grid_indices,
                                n_wall_cells, wall_cell_size):
    """
    Compute polarity vector pointing toward lowest-score neighbor.

    Score calculation:
    - If goal is within range r: point to goal, return score 0
    - Otherwise: score = min(neighbor scores within r) + 1
    - If no neighbors in range: score = 9999

    Polarity calculation:
    - Points toward the position of the lowest-score neighbor(s)

    Uses bounding box optimization for neighbor search.


    (Memory-lean version of point_polarity_to_goal().)

    Differences vs current version:
    - no neighbor_scores temporary array
    - no neighbor_positions temporary array
    - two-pass reduction:
        pass 1 -> find min neighbor score
        pass 2 -> average relative positions of only min-score neighbors
    - returns (px, py, score) as scalars to avoid tiny array allocations
    """
    x_i = pos_i[0]
    y_i = pos_i[1]
    x_goal = goal_position[0]
    y_goal = goal_position[1]
    r2 = r * r

    # Cheap bounding-box rejection before periodic distance
    goal_in_bbox = (x_goal >= x_i - r and x_goal <= x_i + r and
                    y_goal >= y_i - r and y_goal <= y_i + r)

    if goal_in_bbox:
        r_goal = compute_minimum_distance(pos_i, goal_position, box_size)
        dx_goal = r_goal[0]
        dy_goal = r_goal[1]
        dist2_goal = dx_goal * dx_goal + dy_goal * dy_goal

        if dist2_goal <= r2:
            if has_line_of_sight(pos_i, goal_position, payload_pos, payload_radius,
                                 walls, wall_grid_offsets, wall_grid_indices,
                                 n_wall_cells, wall_cell_size):
                if dist2_goal > 0.0:
                    inv_dist = 1.0 / math.sqrt(dist2_goal)
                    return dx_goal * inv_dist, dy_goal * inv_dist, 0
                return 0.0, 0.0, 0

    cell_size = box_size / n_cells
    cell_x = int(x_i / cell_size)
    cell_y = int(y_i / cell_size)

    # ----------------------------
    # Pass 1: find minimum score
    # ----------------------------
    found_neighbor = False
    min_score = 2147483647  # large int

    for dcell_x in range(-1, 2):
        neigh_x = (cell_x + dcell_x) % n_cells
        for dcell_y in range(-1, 2):
            neigh_y = (cell_y + dcell_y) % n_cells
            j = head[neigh_x, neigh_y]

            while j != -1:
                if j != i:
                    r_ij = compute_minimum_distance(pos_i, positions[j], box_size)
                    dx = r_ij[0]
                    dy = r_ij[1]
                    dist2 = dx * dx + dy * dy

                    if dist2 <= r2:
                        pos_j_periodic_x = x_i + dx
                        pos_j_periodic_y = y_i + dy

                        blocked = line_intersects_any_wall_indexed(
                            x_i, y_i,
                            pos_j_periodic_x, pos_j_periodic_y,
                            walls, wall_grid_offsets, wall_grid_indices,
                            n_wall_cells, wall_cell_size
                        )

                        if not blocked:
                            score_j = particle_scores[j]
                            if score_j < min_score:
                                min_score = score_j
                            found_neighbor = True

                j = list_next[j]

    if not found_neighbor:
        return 0.0, 0.0, 9999

    # ---------------------------------------------------
    # Pass 2: average relative positions of min-score neighbors
    # ---------------------------------------------------
    sum_dx = 0.0
    sum_dy = 0.0
    count = 0

    for dcell_x in range(-1, 2):
        neigh_x = (cell_x + dcell_x) % n_cells
        for dcell_y in range(-1, 2):
            neigh_y = (cell_y + dcell_y) % n_cells
            j = head[neigh_x, neigh_y]

            while j != -1:
                if j != i and particle_scores[j] == min_score:
                    r_ij = compute_minimum_distance(pos_i, positions[j], box_size)
                    dx = r_ij[0]
                    dy = r_ij[1]
                    dist2 = dx * dx + dy * dy

                    if dist2 <= r2:
                        pos_j_periodic_x = x_i + dx
                        pos_j_periodic_y = y_i + dy

                        blocked = line_intersects_any_wall_indexed(
                            x_i, y_i,
                            pos_j_periodic_x, pos_j_periodic_y,
                            walls, wall_grid_offsets, wall_grid_indices,
                            n_wall_cells, wall_cell_size
                        )

                        if not blocked:
                            sum_dx += dx
                            sum_dy += dy
                            count += 1

                j = list_next[j]

    if count == 0:
        return 0.0, 0.0, 9999

    avg_dx = sum_dx / count
    avg_dy = sum_dy / count
    norm2 = avg_dx * avg_dx + avg_dy * avg_dy

    if norm2 > 0.0:
        inv_norm = 1.0 / math.sqrt(norm2)
        return avg_dx * inv_norm, avg_dy * inv_norm, min_score + 1

    return 0.0, 0.0, min_score + 1


@njit(fastmath=True, parallel=True)
def simulate_single_step(positions, orientations, velocities, payload_pos, payload_vel,
                         radii, v0s, mobilities, payload_mobility, polarity, particle_scores,
                         stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
                         step, goal_position, particle_view_range, score_and_polarity_update_interval, walls,
                         max_curvity, min_curvity, mid_curvity,
                         polarity_nudge_interval, polarity_nudge_strength,
                         wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size):
    """Simulate a single time step"""
    # Compute forces on particles and payload
    particle_forces, payload_force = compute_all_forces(
        positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls,
        wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size
    )

    # Update polarity vectors and scores based on goal (at update interval)
    if step % score_and_polarity_update_interval == 0:
        # Create cell list for efficient neighbor search
        cell_size = particle_view_range  # Use particle_view_range as cell size for this search
        head_goal, list_next_goal, n_cells_goal = create_cell_list(positions, box_size, cell_size, n_particles)

        old_scores = particle_scores.copy()

        for i in prange(n_particles):
            px, py, new_score = point_polarity_to_goal(
                positions[i], goal_position, positions, old_scores, i,
                particle_view_range, box_size,
                head_goal, list_next_goal, n_cells_goal,
                payload_pos, payload_radius, walls,
                wall_grid_offsets, wall_grid_indices, n_wall_cells, wall_cell_size
            )
            polarity[i, 0] = px
            polarity[i, 1] = py
            particle_scores[i] = new_score

    curvity = compute_curvity_from_polarity(orientations, polarity, n_particles, max_curvity, min_curvity, mid_curvity)

    # Update particle orientations
    orientations = update_orientation_vectors(
        orientations, particle_forces, curvity, dt, rot_diffusion, n_particles
    )

    # Polarity nudge: angularly push heading toward polarity every nudge interval
    if step % polarity_nudge_interval == 0:
        orientations = nudge_orientations_toward_polarity(
            orientations, polarity, polarity_nudge_strength, n_particles
        )

    # Update particle positions and apply goal-based modulation if enabled
    for i in prange(n_particles):

        # Self-propulsion velocity with particle-specific v0
        self_propulsion = v0s[i] * orientations[i]

        # Force-induced velocity with particle-specific mobility
        force_velocity = mobilities[i] * particle_forces[i]

        # Total velocity
        velocities[i] = self_propulsion + force_velocity

        # Update position
        positions[i] += velocities[i] * dt

    # Update payload
    payload_vel = payload_mobility * payload_force

    payload_pos += payload_vel * dt

    # Apply periodic boundary conditions
    positions = positions % box_size
    payload_pos = payload_pos % box_size

    return positions, orientations, velocities, payload_pos, payload_vel, curvity
