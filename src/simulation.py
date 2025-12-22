import numpy as np
from numba import njit

from .physics_utils import normalize, particles_separated_by_wall_periodic
from .forces import compute_repulsive_force, compute_wall_forces, create_cell_list, compute_hollow_payload_force


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
def compute_all_forces_hollow_payload(positions, payload_pos, radii, payload_radius,
                                       stiffness, n_particles, box_size, walls):
    """Compute all forces acting on particles and a hollow payload.

    The hollow payload can be pushed from inside (particles pushing outward)
    or from outside (particles pushing inward).
    """
    particle_forces = np.zeros((n_particles, 2))
    payload_force = np.zeros(2)

    # Determine maximum interaction distance (for cell size)
    max_radius = np.max(radii)
    cell_size = 2 * max_radius

    # Create cell list (O(N))
    head, list_next, n_cells = create_cell_list(positions, box_size, cell_size, n_particles)

    # Compute forces between particles and hollow payload (O(N))
    for i in range(n_particles):
        # Only compute forces if particle and payload are not separated by a wall
        if not particles_separated_by_wall_periodic(positions[i], payload_pos, walls, box_size):
            force_particle_payload = compute_hollow_payload_force(
                positions[i], payload_pos, radii[i], payload_radius, stiffness, box_size
            )
            particle_forces[i] += force_particle_payload
            payload_force -= force_particle_payload  # Newton's third law

    # Compute forces between particles and walls (O(N * n_walls))
    for i in range(n_particles):
        wall_force = compute_wall_forces(positions[i], radii[i], walls, stiffness)
        particle_forces[i] += wall_force

    # Compute force between hollow payload and walls
    payload_wall_force = compute_wall_forces(payload_pos, payload_radius, walls, stiffness)
    payload_force += payload_wall_force

    # Compute forces between particles using cell list (O(N))
    for i in range(n_particles):
        cell_x = int(positions[i, 0] / cell_size)
        cell_y = int(positions[i, 1] / cell_size)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neigh_x = (cell_x + dx) % n_cells
                neigh_y = (cell_y + dy) % n_cells

                j = head[neigh_x, neigh_y]

                while j != -1:
                    if i != j:
                        if not particles_separated_by_wall_periodic(positions[i], positions[j], walls, box_size):
                            particle_forces[i] += compute_repulsive_force(
                                positions[i], positions[j], radii[i], radii[j], stiffness, box_size
                            )
                    j = list_next[j]

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
def simulate_single_step(positions, orientations, velocities, payload_pos, payload_vel,
                         radii, v0s, mobilities, payload_mobility, curvity,
                         stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles, walls):
    """Simulate a single time step with fixed curvity values"""
    # Compute forces on particles and payload
    particle_forces, payload_force = compute_all_forces(
        positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls
    )

    # Update particle orientations using fixed curvity
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

    # Update payload
    payload_vel = payload_mobility * payload_force
    payload_pos += payload_vel * dt

    # Apply periodic boundary conditions
    positions = positions % box_size
    payload_pos = payload_pos % box_size

    return positions, orientations, velocities, payload_pos, payload_vel


@njit(fastmath=True)
def simulate_single_step_hollow_payload(positions, orientations, velocities, payload_pos, payload_vel,
                                         radii, v0s, mobilities, payload_mobility, curvity,
                                         stiffness, box_size, payload_radius,
                                         dt, rot_diffusion, n_particles, walls):
    """Simulate a single time step with a hollow payload.

    Particles inside push outward, particles outside push inward.
    """
    # Compute forces on particles and hollow payload
    particle_forces, payload_force = compute_all_forces_hollow_payload(
        positions, payload_pos, radii, payload_radius,
        stiffness, n_particles, box_size, walls
    )

    # Update particle orientations using fixed curvity
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

    # Update payload
    payload_vel = payload_mobility * payload_force
    payload_pos += payload_vel * dt

    # Apply periodic boundary conditions
    positions = positions % box_size
    payload_pos = payload_pos % box_size

    return positions, orientations, velocities, payload_pos, payload_vel
