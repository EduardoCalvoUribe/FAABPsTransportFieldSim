import numpy as np
from numba import njit

from .physics_utils import normalize, particles_separated_by_wall_periodic
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
def fetch_polarity_from_field(positions, polarity_fields, box_size, n_particles):
    """Fetch polarity vectors from each particle's field based on current position.

    Args:
        positions: Current particle positions (n_particles, 2)
        polarity_fields: Polarity vector fields (n_particles, box_size, box_size, 2)
        box_size: Size of simulation box (determines grid size)
        n_particles: Number of particles

    Returns:
        active_polarity: Array of polarity vectors fetched from fields (n_particles, 2)
    """
    active_polarity = np.zeros((n_particles, 2))

    for i in range(n_particles):
        # Convert position to grid indices (floor to get cell)
        grid_x = int(positions[i, 0]) % box_size
        grid_y = int(positions[i, 1]) % box_size

        # Fetch polarity vector from field
        active_polarity[i] = polarity_fields[i, grid_x, grid_y]

    return active_polarity


@njit(fastmath=True)
def update_polarity_fields_from_forces(positions, forces, polarity_fields, box_size, n_particles, force_scaling):
    """Accumulate external forces in polarity vector fields based on particle positions.

    When a particle experiences force F at position (x, y), we add scaled F to the
    polarity field at grid location (int(x), int(y)), then apply tanh() to prevent
    unbounded growth. This accumulates force history over time at each spatial location.

    Args:
        positions: Current particle positions (n_particles, 2)
        forces: External forces on particles (n_particles, 2)
        polarity_fields: Polarity vector fields to update (n_particles, box_size, box_size, 2)
        box_size: Size of simulation box
        n_particles: Number of particles
        force_scaling: Scaling factor applied to forces before accumulation
    """
    for i in range(n_particles):
        # Convert position to grid indices
        grid_x = int(positions[i, 0]) % box_size
        grid_y = int(positions[i, 1]) % box_size

        # Accumulate scaled force vector and apply tanh to bound growth
        polarity_fields[i, grid_x, grid_y, 0] = np.tanh(polarity_fields[i, grid_x, grid_y, 0] + force_scaling * forces[i, 0])
        polarity_fields[i, grid_x, grid_y, 1] = np.tanh(polarity_fields[i, grid_x, grid_y, 1] + force_scaling * forces[i, 1])


@njit(fastmath=True)
def share_polarity_fields_on_collision(positions, radii, polarity_fields, box_size, n_particles,
                                        last_collision_partner):
    """Share polarity field knowledge when particles collide.

    When two particles collide, they sum their polarity fields and apply tanh() to prevent
    unbounded growth. Only counts new collisions (not continuous contact with same particle).

    Args:
        positions: Particle positions (n_particles, 2)
        radii: Particle radii (n_particles,)
        polarity_fields: Polarity vector fields (n_particles, box_size, box_size, 2)
        box_size: Size of simulation box
        n_particles: Number of particles
        last_collision_partner: Track last collision partner for each particle (n_particles,)
                               -1 means no recent collision

    Returns:
        Updated last_collision_partner array
    """
    # Create cell list for efficient collision detection
    max_radius = np.max(radii)
    cell_size = 2 * max_radius
    head, list_next, n_cells = create_cell_list(positions, box_size, cell_size, n_particles)

    # Track which particles have collided this step
    new_collision_partner = np.full(n_particles, -1, dtype=np.int64)

    # Check for collisions using cell list
    for i in range(n_particles):
        cell_x = int(positions[i, 0] / cell_size)
        cell_y = int(positions[i, 1] / cell_size)

        # Check neighboring cells
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neigh_x = (cell_x + dx) % n_cells
                neigh_y = (cell_y + dy) % n_cells

                j = head[neigh_x, neigh_y]

                while j != -1:
                    if i < j:  # Only process each pair once (i < j ensures no duplicates)
                        # Check if particles are colliding
                        dx_ij = positions[j, 0] - positions[i, 0]
                        dy_ij = positions[j, 1] - positions[i, 1]

                        # Periodic boundary handling
                        if dx_ij > box_size / 2:
                            dx_ij -= box_size
                        elif dx_ij < -box_size / 2:
                            dx_ij += box_size
                        if dy_ij > box_size / 2:
                            dy_ij -= box_size
                        elif dy_ij < -box_size / 2:
                            dy_ij += box_size

                        dist = np.sqrt(dx_ij**2 + dy_ij**2)
                        contact_dist = radii[i] + radii[j]

                        if dist < contact_dist:
                            # Collision detected! Check if this is a NEW collision
                            # (not the same partner as last time for either particle)
                            if last_collision_partner[i] != j and last_collision_partner[j] != i:
                                # New collision - share polarity fields
                                # Sum the fields and apply tanh to each component
                                for gx in range(box_size):
                                    for gy in range(box_size):
                                        # Sum fields
                                        sum_x = polarity_fields[i, gx, gy, 0] + polarity_fields[j, gx, gy, 0]
                                        sum_y = polarity_fields[i, gx, gy, 1] + polarity_fields[j, gx, gy, 1]

                                        # Apply tanh and update both particles
                                        polarity_fields[i, gx, gy, 0] = np.tanh(sum_x)
                                        polarity_fields[i, gx, gy, 1] = np.tanh(sum_y)
                                        polarity_fields[j, gx, gy, 0] = np.tanh(sum_x)
                                        polarity_fields[j, gx, gy, 1] = np.tanh(sum_y)

                            # Record collision partner
                            new_collision_partner[i] = j
                            new_collision_partner[j] = i

                    j = list_next[j]

    return new_collision_partner


@njit(fastmath=True)
def compute_curvity_from_polarity(orientations, active_polarity, n_particles):
    """Compute curvity for all particles based on their active polarity vectors.

    Curvity = -(e · p), where:
    - e is the particle's heading direction (orientation)
    - p is the particle's active polarity vector (fetched from field)
    """
    curvity = np.zeros(n_particles)

    for i in range(n_particles):
        # Compute dot product: e · p
        dot_product = orientations[i, 0] * active_polarity[i, 0] + orientations[i, 1] * active_polarity[i, 1]

        # Curvity is negative dot product
        curvity[i] = -dot_product

    return curvity



@njit(fastmath=True)
def simulate_single_step(positions, orientations, velocities, payload_pos, payload_vel,
                         radii, v0s, mobilities, payload_mobility, polarity_fields,
                         stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
                         step, force_update_interval, walls, n_training_steps,
                         payload_circular_motion, payload_circle_center, payload_circle_radius,
                         payload_n_rotations, last_collision_partner, collision_share_interval,
                         polarity_force_scaling, test_phase_learning):
    """Simulate a single time step with force-based polarity fields.

    Args:
        positions: Particle positions
        orientations: Particle orientations
        velocities: Particle velocities
        payload_pos: Payload position
        payload_vel: Payload velocity
        radii: Particle radii
        v0s: Self-propulsion speeds
        mobilities: Particle mobilities
        payload_mobility: Payload mobility
        polarity_fields: Polarity vector fields for each particle (n_particles, box_size, box_size, 2)
        stiffness: Force stiffness
        box_size: Simulation box size
        payload_radius: Payload radius
        dt: Time step
        rot_diffusion: Rotational diffusion coefficients
        n_particles: Number of particles
        step: Current time step
        force_update_interval: How often to update polarity fields with forces
        walls: Wall segments array
        n_training_steps: Number of training steps (circular motion phase)
        payload_circular_motion: If True, payload follows circular path during training phase
        payload_circle_center: Center of circular path (x, y)
        payload_circle_radius: Radius of circular path
        payload_n_rotations: Number of full rotations to complete during training phase
        last_collision_partner: Track collision partners (n_particles,)
        collision_share_interval: How often to share polarity fields on collision (timesteps)
        polarity_force_scaling: Scaling factor for force accumulation in polarity fields
        test_phase_learning: Learning control during test phase (0=none, 1=collision only, 2=both)

    Returns:
        Updated positions, orientations, velocities, payload_pos, payload_vel, curvity, active_polarity, last_collision_partner
    """
    # Compute forces on particles and payload
    particle_forces, payload_force = compute_all_forces(
        positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls
    )

    # Fetch active polarity vectors from fields based on current positions
    active_polarity = fetch_polarity_from_field(positions, polarity_fields, box_size, n_particles)

    # Compute curvity from active polarity
    curvity = compute_curvity_from_polarity(orientations, active_polarity, n_particles)

    # Determine if we're in test phase
    in_test_phase = step > n_training_steps

    # Update polarity fields with current forces (at update interval)
    # Skip if in test phase and learning is disabled (test_phase_learning < 2)
    if step % force_update_interval == 0:
        if not in_test_phase or test_phase_learning >= 2:
            update_polarity_fields_from_forces(positions, particle_forces, polarity_fields, box_size, n_particles, polarity_force_scaling)

    # Share polarity fields when particles collide (at sharing interval)
    # Skip if in test phase and collision learning is disabled (test_phase_learning < 1)
    if step % collision_share_interval == 0:
        if not in_test_phase or test_phase_learning >= 1:
            last_collision_partner = share_polarity_fields_on_collision(
                positions, radii, polarity_fields, box_size, n_particles, last_collision_partner
            )

    # Update particle orientations
    orientations = update_orientation_vectors(
        orientations, particle_forces, curvity, dt, rot_diffusion, n_particles
    )

    # Update particle positions
    for i in range(n_particles):
        # Self-propulsion velocity with particle-specific v0
        self_propulsion = v0s[i] * orientations[i]

        # Force-induced velocity with particle-specific mobility
        force_velocity = mobilities[i] * particle_forces[i]

        # Total velocity
        velocities[i] = self_propulsion + force_velocity

        # Update position
        positions[i] += velocities[i] * dt

    # Update payload - either follow circular path or respond to forces
    if payload_circular_motion and n_training_steps > 0 and step <= n_training_steps:
        # Training phase: payload follows circular path, ignores forces
        # Angular position: complete n_rotations full circles during training phase
        # (2*pi * n_rotations) radians over n_training_steps steps
        angular_velocity = (2.0 * np.pi * payload_n_rotations) / n_training_steps
        angle = step * angular_velocity

        # Position on circle
        payload_pos[0] = payload_circle_center[0] + payload_circle_radius * np.cos(angle)
        payload_pos[1] = payload_circle_center[1] + payload_circle_radius * np.sin(angle)

        # Velocity (tangent to circle)
        payload_vel[0] = -payload_circle_radius * angular_velocity * np.sin(angle)
        payload_vel[1] = payload_circle_radius * angular_velocity * np.cos(angle)
    else:
        # Test phase (or always if circular motion disabled): passive payload
        payload_vel = payload_mobility * payload_force
        payload_pos += payload_vel * dt

    # Apply periodic boundary conditions
    positions = positions % box_size
    payload_pos = payload_pos % box_size

    return positions, orientations, velocities, payload_pos, payload_vel, curvity, active_polarity, last_collision_partner
