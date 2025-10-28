import numpy as np
import math
from numba import njit

from .physics_utils import normalize, line_intersects_any_wall, compute_minimum_distance, particles_separated_by_wall, particles_separated_by_wall_periodic
from .forces import compute_repulsive_force, compute_wall_forces, create_cell_list


##########################
# Main physics functions #
##########################

@njit(fastmath=True)
def compute_all_forces(positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls):
    """Compute all forces acting on particles and the payload"""
    particle_forces = np.zeros((n_particles, 2)) # Initialize force array for particles
    payload_force = np.zeros(2) # Initialize force array for payload

    # Determine maximum interaction distance (for cell size)
    max_radius = np.max(radii) # Takes maximum radius of all particles. (Because radius of particles is possibly heterogeneous)
    cell_size = 2 * max_radius  # For particle-particle interactions (not payload-particle)

    # Create cell list (O(N))
    head, list_next, n_cells = create_cell_list(positions, box_size, cell_size, n_particles)

    # Compute forces between particles and payload (O(N))
    for i in range(n_particles):
        # Only compute forces if particle and payload are not separated by a wall (periodic shortest path)
        if not particles_separated_by_wall_periodic(positions[i], payload_pos, walls, box_size):
            force_particle_payload = compute_repulsive_force( # Computes force between particle and payload
                positions[i], payload_pos, radii[i], payload_radius, stiffness, box_size
            )
            particle_forces[i] += force_particle_payload # Applies force to particle
            payload_force -= force_particle_payload  # Applies opposite force to payload

    # Compute forces between particles and walls (O(N * n_walls))
    for i in range(n_particles):
        wall_force = compute_wall_forces(positions[i], radii[i], walls, stiffness)
        particle_forces[i] += wall_force

    # Compute force between payload and walls
    payload_wall_force = compute_wall_forces(payload_pos, payload_radius, walls, stiffness)
    payload_force += payload_wall_force

    # Compute forces between particles using cell list (now O(N))
    # For each particle
    for i in range(n_particles):
        # Find which cell it belongs to
        cell_x = int(positions[i, 0] / cell_size)
        cell_y = int(positions[i, 1] / cell_size)

        # Check neighboring cells (including own cell)
        for dx in range(-1, 2):  # -1, 0, 1
            for dy in range(-1, 2):  # -1, 0, 1
                # Get neighboring cell (periodic boundaries)
                neigh_x = (cell_x + dx) % n_cells
                neigh_y = (cell_y + dy) % n_cells
                # neigh_cell_id = neigh_y * n_cells + neigh_x  #in case you use row-major ordering

                # Get the first particle in the neighboring cell
                j = head[neigh_x, neigh_y] # head[neigh_cell_id]

                # Looping through all particles in this cell
                while j != -1:
                    if i != j:
                        # Only compute forces if particles are not separated by a wall (periodic shortest path)
                        if not particles_separated_by_wall_periodic(positions[i], positions[j], walls, box_size):
                            particle_forces[i] += compute_repulsive_force(
                                positions[i], positions[j], radii[i], radii[j], stiffness, box_size
                            )
                    j = list_next[j] # Check create_cell_list() for more details

    return particle_forces, payload_force

@njit(fastmath=True)
def update_orientation_vectors(orientations, forces, curvity, dt, rot_diffusion, n_particles):
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

    new_orientations = np.zeros_like(orientations)

    for i in range(n_particles):
        # Calculate torque: τ = curvity * (n × F)
        # n × F = n_x*F_y - n_y*F_x
        cross_product = orientations[i, 0] * forces[i, 1] - orientations[i, 1] * forces[i, 0]
        torque = curvity[i] * cross_product

        # Calculate orientation change: dn/dt = torque * (n × z)
        # n × z = (-n_y, n_x)
        n_cross_z = np.array([-orientations[i, 1], orientations[i, 0]])
        orientation_change = torque * n_cross_z * dt

        # Add rotational diffusion as a random perpendicular vector
        if rot_diffusion[i] > 0:
            # Generate noise using normal distribution
            noise_magnitude = np.sqrt(2 * rot_diffusion[i] * dt)
            noise_x = np.random.normal(0, noise_magnitude)
            noise_y = np.random.normal(0, noise_magnitude)
            noise_vector = np.array([noise_x, noise_y])

            # Project noise to be perpendicular to orientation using cross product
            # (n × (noise × n)) = noise - (noise·n)n
            noise_dot_n = noise_vector[0] * orientations[i, 0] + noise_vector[1] * orientations[i, 1]
            noise_perp = np.array([
                noise_vector[0] - noise_dot_n * orientations[i, 0],
                noise_vector[1] - noise_dot_n * orientations[i, 1]
            ])

            orientation_change += noise_perp

        # Update orientation and normalize
        new_orientations[i] = normalize(orientations[i] + orientation_change)

    return new_orientations



@njit(fastmath=True)
def compute_curvity_from_polarity(orientations, polarity, n_particles):
    """Compute curvity for all particles based on their polarity vectors.

    Curvity = -(e · p), where:
    - e is the particle's heading direction (orientation)
    - p is the particle's polarity vector
    """
    curvity = np.zeros(n_particles)

    for i in range(n_particles):
        # Compute dot product: e · p
        dot_product = orientations[i, 0] * polarity[i, 0] + orientations[i, 1] * polarity[i, 1]

        # Curvity is negative dot product
        curvity[i] = -dot_product

    return curvity

@njit(fastmath=True)
def has_line_of_sight(pos_i, goal_position, payload_pos, payload_radius, box_size, walls):
    """Check if particle i has line of sight to goal (no walls or payload blocking).

    Returns True if line from particle to goal doesn't intersect with walls or payload circle.
    Does not use periodic boundaries.
    """
    x_i, y_i = pos_i
    x_goal, y_goal = goal_position
    x_p, y_p = payload_pos

    # Check if any wall blocks line of sight
    if line_intersects_any_wall(x_i, y_i, x_goal, y_goal, walls):
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

    # Direction vector: particle to payload
    fx, fy = x_p - x_i, y_p - y_i

    # Coefficients of quadratic equation for line-circle intersection
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
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
def compute_polarity_weighted_vicsek(neighbor_indices, neighbor_scores, all_polarity):
    """Component 1: Score-weighted alignment with neighbors.

    Lower scores = higher weight (stronger influence).
    Uses exponential weighting: weight = exp(-(score - min_score)).

    Args:
        neighbor_indices: list of neighbor particle indices
        neighbor_scores: scores of neighbor particles
        all_polarity: polarity vectors of all particles

    Returns:
        weighted_polarity: normalized score-weighted average polarity vector
    """
    min_score = min(neighbor_scores)
    weighted_polarity = np.zeros(2)
    total_weight = 0.0

    for idx in range(len(neighbor_indices)):
        j = neighbor_indices[idx]
        score_j = neighbor_scores[idx]

        # Weight based on relative score difference from minimum
        # Particles with min_score get weight=1.0, others get exponentially smaller weights
        score_diff = score_j - min_score
        weight = np.exp(-score_diff)  # Exponential decay based on score difference

        weighted_polarity += weight * all_polarity[j]
        total_weight += weight

    # Normalize the weighted average
    if total_weight > 0:
        weighted_polarity = weighted_polarity / total_weight

    # Normalize to unit vector
    norm_weighted = np.sqrt(np.sum(weighted_polarity**2))
    if norm_weighted > 0:
        weighted_polarity = weighted_polarity / norm_weighted
    else:
        weighted_polarity = np.array([0.0, 0.0])

    return weighted_polarity


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
def point_polarity_to_goal(pos_i, goal_position, positions, particle_scores, i, n_particles, r, box_size, current_score, head, list_next, n_cells, all_polarity, payload_pos, payload_radius, walls, directedness):
    """Compute polarity vector by balancing score-weighted alignment and gradient following.

    Score calculation:
    - If goal is within range r: point to goal, return score 0
    - Otherwise: score = min(neighbor scores within r) + 1
    - If no neighbors in range: score = 9999

    Polarity calculation:
    - Balances two components based on directedness parameter:
      1. Score-weighted alignment: neighbors' polarity vectors weighted by exp(-(s_j - s_min)) (weight: 1 - directedness)
      2. Gradient following: direction toward lowest-score neighbor position (weight: directedness)
    - directedness ∈ [0, 1]: 0 = pure score-weighted Vicsek, 1 = pure gradient descent

    Uses bounding box optimization for neighbor search.

    Returns:
        polarity: aligned unit polarity vector
        score: new score for particle i
    """
    x_i, y_i = pos_i
    x_goal, y_goal = goal_position

    # Quick bounding box check for goal (cheap early rejection)
    # If goal is outside the bounding box, it's definitely out of range
    goal_in_bbox = (x_goal >= x_i - r and x_goal <= x_i + r and
                    y_goal >= y_i - r and y_goal <= y_i + r)

    # Only compute expensive distance if goal is in bounding box
    if goal_in_bbox:
        # Check distance to goal (PERIODIC)
        r_goal = compute_minimum_distance(pos_i, goal_position, box_size)
        dist_to_goal = np.sqrt(np.sum(r_goal**2))
        dx_goal, dy_goal = r_goal[0], r_goal[1]

        # If goal is within range, check line of sight
        if dist_to_goal <= r:
            # Check if line of sight is clear (no walls or payload blocking)
            if has_line_of_sight(pos_i, goal_position, payload_pos, payload_radius, box_size, walls):
                if dist_to_goal > 0:
                    return np.array([dx_goal / dist_to_goal, dy_goal / dist_to_goal]), 0
                return np.array([0.0, 0.0]), 0 # payload is exactly on the goal

            # If blocked, fall through to gradient-following behavior

    # Goal out of range - find all neighbors within r and collect their info
    neighbor_indices = []
    neighbor_scores = []
    neighbor_positions = []

    # Determine cell size from the cell list
    cell_size = box_size / n_cells

    # Find which cell particle i belongs to
    cell_x = int(pos_i[0] / cell_size)
    cell_y = int(pos_i[1] / cell_size)

    # Check neighboring cells (including own cell)
    for dx in range(-1, 2):  # -1, 0, 1
        for dy in range(-1, 2):  # -1, 0, 1

            # Get neighboring cell (PERIODIC)
            neigh_x = (cell_x + dx) % n_cells
            neigh_y = (cell_y + dy) % n_cells

            # Get the first particle in the neighboring cell
            j = head[neigh_x, neigh_y]

            # Loop through all particles in this cell
            while j != -1:
                if i != j:
                    # First check if wall separates particles along periodic shortest path
                    if not particles_separated_by_wall_periodic(pos_i, positions[j], walls, box_size):
                        # Compute distance (PERIODIC)
                        r_ij = compute_minimum_distance(pos_i, positions[j], box_size)

                        dist_j = np.sqrt(np.sum(r_ij**2))
                        # Only include neighbor if within range
                        if dist_j <= r:
                            neighbor_indices.append(j)
                            neighbor_scores.append(particle_scores[j])
                            neighbor_positions.append(positions[j].copy())

                j = list_next[j]

    # If no neighbors found, return score 9999 and zero vector
    if len(neighbor_indices) == 0:
        return np.array([0.0, 0.0]), 9999

    # Calculate new score: min(neighbor scores) + 1
    min_score = min(neighbor_scores)
    new_score = min_score + 1

    # Component 1: Score-weighted alignment with neighbors
    weighted_polarity = compute_polarity_weighted_vicsek(neighbor_indices, neighbor_scores, all_polarity)

    # Component 2: Direction toward particle with lowest score
    gradient_polarity = compute_polarity_toward_minscore_pos(pos_i, neighbor_scores, neighbor_positions, box_size)

    # ALTERNATIVE Component 2: Uncomment to use average angle method instead
    # gradient_polarity = compute_polarity_toward_minscore_ang(pos_i, neighbor_scores, neighbor_positions, box_size)

    # Combine components based on directedness parameter
    # (1-d): score-weighted alignment, (d): direct gradient following
    combined_polarity = (1.0 - directedness) * weighted_polarity + directedness * gradient_polarity

    # Normalize final vector
    norm = np.sqrt(np.sum(combined_polarity**2))
    if norm > 0:
        combined_polarity = combined_polarity / norm
    else:
        combined_polarity = np.array([0.0, 0.0])

    return combined_polarity, new_score


@njit(fastmath=True)
def simulate_single_step(positions, orientations, velocities, payload_pos, payload_vel,
                         radii, v0s, mobilities, payload_mobility, polarity, particle_scores,
                         stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
                         step, goal_position, particle_view_range, score_and_polarity_update_interval, walls, directedness):
    """Simulate a single time step"""
    # Compute forces on particles and payload
    particle_forces, payload_force = compute_all_forces(
        positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls
    )

    # Initialize curvity (will be updated if at update interval)
    curvity = compute_curvity_from_polarity(orientations, polarity, n_particles)

    # Update polarity vectors and scores based on goal (at update interval)
    if step % score_and_polarity_update_interval == 0:
        # Create cell list for efficient neighbor search
        cell_size = particle_view_range  # Use particle_view_range as cell size for this search
        head_goal, list_next_goal, n_cells_goal = create_cell_list(positions, box_size, cell_size, n_particles)

        for i in range(n_particles):
            polarity[i], particle_scores[i] = point_polarity_to_goal(
                positions[i], goal_position, positions, particle_scores, i, n_particles,
                particle_view_range, box_size, particle_scores[i], head_goal, list_next_goal, n_cells_goal,
                polarity, payload_pos, payload_radius, walls, directedness
            )

        # Recompute curvity from updated polarity vectors
        curvity = compute_curvity_from_polarity(orientations, polarity, n_particles)

    # Update particle orientations
    orientations = update_orientation_vectors(
        orientations, particle_forces, curvity, dt, rot_diffusion, n_particles
    )

    # Update particle positions and apply goal-based modulation if enabled
    for i in range(n_particles):

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
