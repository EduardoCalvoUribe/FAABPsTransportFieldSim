
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle
import time
import os
import math
from numba import njit, float64, int64
from vector_fields import uniform_vector_field, radial_vector_field, limit_cycle_vector_field



np.random.seed(42)

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
def compute_minimum_distance(pos_i, pos_j, box_size):
    """Compute minimum distance vector considering periodic boundaries."""
    r_ij = pos_j - pos_i
    
    r_ij = r_ij - box_size * np.round(r_ij / box_size)
    
    return r_ij

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

##########################
# Main physics functions #
##########################

@njit(fastmath=True)
def compute_all_forces(positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size):
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
        force_particle_payload = compute_repulsive_force( # Computes force between particle and payload
            positions[i], payload_pos, radii[i], payload_radius, stiffness, box_size
        )
        particle_forces[i] += force_particle_payload # Applies force to particle
        payload_force -= force_particle_payload  # Applies opposite force to payload
    
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
def discretize_position(pos, box_size, grid_size):
    """Convert continuous position to discrete grid indices."""
    # Map position [0, box_size) to grid index [0, grid_size)
    x_idx = np.int64(pos[0] * grid_size / box_size)
    y_idx = np.int64(pos[1] * grid_size / box_size)

    # Clamp to valid range (in case of rounding errors at boundaries)
    x_idx = min(max(x_idx, 0), grid_size - 1)
    y_idx = min(max(y_idx, 0), grid_size - 1)

    return x_idx, y_idx

@njit(fastmath=True)
def compute_curvity_from_vector_field(positions, orientations, vector_fields, box_size, grid_size, n_particles):
    """Compute curvity for all particles based on their vector field memory.

    Curvity = -(e · v), where:
    - e is the particle's heading direction (orientation)
    - v is the vector at the particle's current location in its vector field
    """
    curvity = np.zeros(n_particles)

    for i in range(n_particles):
        # Get discrete grid indices for particle's position
        x_idx, y_idx = discretize_position(positions[i], box_size, grid_size)

        # Get vector from particle's memory at this location
        v = vector_fields[i, x_idx, y_idx]

        # Compute dot product: e · v
        dot_product = orientations[i, 0] * v[0] + orientations[i, 1] * v[1]

        # Curvity is negative dot product
        curvity[i] = -dot_product

    return curvity
    
@njit(fastmath=True)
def simulate_single_step(positions, orientations, velocities, payload_pos, payload_vel,
                         radii, v0s, mobilities, payload_mobility, vector_fields, grid_size,
                         stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles, step):
    """Simulate a single time step"""
    # Compute forces on particles and payload
    particle_forces, payload_force = compute_all_forces(
        positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size
    )

    # Compute curvity from vector fields
    curvity = compute_curvity_from_vector_field(
        positions, orientations, vector_fields, box_size, grid_size, n_particles
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
        velocities[i] = self_propulsion + force_velocity # + velocities[i] / 10 # added velocities[i] to test inertia!!

        # Update position
        positions[i] += velocities[i] * dt

    # Update payload
    payload_vel = payload_mobility * payload_force

    payload_pos += payload_vel * dt

    # Apply periodic boundary conditions
    positions = positions % box_size
    payload_pos = payload_pos % box_size

    return positions, orientations, velocities, payload_pos, payload_vel, curvity

#####################################################
# Main simulation runner functions                  #
#####################################################

def run_payload_simulation(params):
    """Run the complete payload transport simulation."""
    print(f"Running payload transport simulation with {params['n_particles']} particles for {params['n_steps']} steps...")

    # Initialize arrays
    n_particles = params['n_particles']
    box_size = params['box_size']
    n_steps = params['n_steps']
    save_interval = params['save_interval']
    grid_size = params['grid_size']

    # Initialize particle positions, orientations, and velocities
    positions = np.random.uniform(0, box_size, (n_particles, 2))
    orientations = np.zeros((n_particles, 2))
    velocities = np.zeros((n_particles, 2))

    # Initialize random orientations
    for i in range(n_particles):
        angle = np.random.uniform(0, 2*np.pi)
        orientations[i] = np.array([np.cos(angle), np.sin(angle)])

    # Initialize vector fields for all particles (all pointing upward: (0, 1))
    # vector_fields = np.zeros((n_particles, grid_size, grid_size, 2), dtype=np.float64)
    # vector_fields[:, :, :, 1] = 1.0  # All vectors point upward (0, 1)

    #single_field = limit_cycle_vector_field(grid_size, cycle_radius=grid_size / 4, rad_mult=6.0)
    #single_field = uniform_vector_field(grid_size, np.pi / 4)
    single_field = radial_vector_field(grid_size, np.array([grid_size / 2, grid_size / 2]))
    vector_fields = np.tile(single_field, (n_particles, 1, 1, 1))

    # Initialize payload location. Bottom left corner for now
    payload_pos = np.array([box_size/4, box_size/4])
    payload_vel = np.zeros(2)

    # Pre-allocate arrays for storing simulation data
    n_saves = n_steps // save_interval + 1
    saved_positions = np.zeros((n_saves, n_particles, 2))
    saved_orientations = np.zeros((n_saves, n_particles, 2))
    saved_velocities = np.zeros((n_saves, n_particles, 2))
    saved_payload_positions = np.zeros((n_saves, 2))
    saved_payload_velocities = np.zeros((n_saves, 2))
    saved_curvity = np.zeros((n_saves, n_particles))
    
    # Compute initial curvity from vector fields
    initial_curvity = compute_curvity_from_vector_field(
        positions, orientations, vector_fields, box_size, grid_size, n_particles
    )

    # Store initial state
    saved_positions[0] = positions.copy()
    saved_orientations[0] = orientations.copy()
    saved_velocities[0] = velocities.copy()
    saved_payload_positions[0] = payload_pos.copy()
    saved_payload_velocities[0] = payload_vel.copy()
    saved_curvity[0] = initial_curvity.copy()
    
    # Run simulation
    start_time = time.time()
    save_idx = 1
    
    for step in range(1, n_steps + 1):
        # Unified simulation step
        positions, orientations, velocities, payload_pos, payload_vel, curvity = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            params['particle_radius'], params['v0'], params['mobility'], params['payload_mobility'],
            vector_fields, grid_size, params['stiffness'],
            params['box_size'], params['payload_radius'], params['dt'], params['rot_diffusion'], n_particles, step
        )

        # Save data at specified intervals
        if step % save_interval == 0:
            saved_positions[save_idx] = positions
            saved_orientations[save_idx] = orientations
            saved_velocities[save_idx] = velocities
            saved_payload_positions[save_idx] = payload_pos
            saved_payload_velocities[save_idx] = payload_vel
            saved_curvity[save_idx] = curvity.copy()
            save_idx += 1
            
            # Report progress periodically
            if step % (save_interval * 10) == 0:
                print(f"Step {step}:")
                payload_displacement = np.sqrt(np.sum((saved_payload_positions[save_idx-1] - saved_payload_positions[0])**2))
                print(f"  Payload displacement from start: {payload_displacement:.3f}")
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Calculate payload displacement
    total_payload_displacement = np.sqrt(np.sum((saved_payload_positions[-1] - saved_payload_positions[0])**2))
    print(f"Total payload displacement: {total_payload_displacement:.3f}")
    
    return (
        saved_positions,
        saved_orientations,
        saved_velocities,
        saved_payload_positions,
        saved_payload_velocities,
        saved_curvity,
        vector_fields,
        end_time - start_time
    )

def create_payload_animation(positions, orientations, velocities, payload_positions, params,
                            curvity_values, output_file='visualizations/payload_animation_00.mp4',
                            show_avg_vector_field=False, vector_field_step=20, vector_fields=None):
    """Create an animation of the payload transport simulation."""
    
    print("Creating animation...")

    start_time = time.time()

    # Extract parameters
    box_size = params['box_size']
    payload_radius = params['payload_radius']
    n_particles = params['n_particles']
    grid_size = params['grid_size']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set axis limits
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title('FAABP Cooperative Transport Simulation')
    ax.grid(True, alpha=0.3)
    
    # Color mapping: curvity -1 (dark blue) -> 0 (gray) -> +1 (red)
    def get_particle_color(curvity_value):
        """Map curvity value to RGB color with smooth gradient.
        -1: dark blue, 0: gray, +1: red"""
        # Clamp curvity to [-1, 1] range
        c = np.clip(curvity_value, -1, 1)

        if c < 0:
            # Interpolate from dark blue (0, 0, 0.5) to gray (0.5, 0.5, 0.5)
            t = (c + 1)  # Map [-1, 0] to [0, 1]
            r = 0.0 + t * 0.5
            g = 0.0 + t * 0.5
            b = 0.5 + t * 0.0
        else:
            # Interpolate from gray (0.5, 0.5, 0.5) to red (1, 0, 0)
            t = c  # Map [0, 1] to [0, 1]
            r = 0.5 + t * 0.5
            g = 0.5 - t * 0.5
            b = 0.5 - t * 0.5

        return (r, g, b)

    # Initialize particle colors based on initial curvity values
    particle_colors = [get_particle_color(curvity_values[0, i]) for i in range(n_particles)]

    scatter = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        s=np.pi * (params['particle_radius'] * 2)**2,  # Area of circle
        c=particle_colors,
        alpha=0.7
    )
    
    # Create payload
    payload = Circle(
        (payload_positions[0, 0], payload_positions[0, 1]),
        radius=payload_radius,
        color='gray',
        alpha=0.7
    )
    ax.add_patch(payload)

    # Create payload trajectory
    trajectory, = ax.plot(
        payload_positions[0:1, 0], 
        payload_positions[0:1, 1], 
        'k--', 
        alpha=0.5, 
        linewidth=1.0
    )
    
    # Add parameters text
    params_text = ax.text(-0.02, -0.065, f'n_particles: {n_particles}, particle radius: {params["particle_radius"][0]}, payload radius: {payload_radius}', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    params_text_2 = ax.text(-0.02, -0.093, f'orientational noise: {params["rot_diffusion"][0]}, particle mobility: {params["mobility"][0]}, payload mobility: {params["payload_mobility"]}', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    
    # Add time counter
    time_text = ax.text(0.02, 0.98, 'Frame: 0', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')

    # Initialize vector field visualization
    quiver = None
    if show_avg_vector_field:
        if vector_fields is None:
            raise ValueError("vector_fields must be provided when show_avg_vector_field is True")

        # Compute average vector field across all particles
        avg_field = np.mean(vector_fields, axis=0)

        # Create coordinate grids with specified step
        y_coords, x_coords = np.meshgrid(
            np.arange(0, grid_size, vector_field_step),
            np.arange(0, grid_size, vector_field_step),
            indexing='ij'
        )

        # Scale coordinates to box_size
        x_scaled = x_coords * (box_size / grid_size)
        y_scaled = y_coords * (box_size / grid_size)

        # Sample the average vector field at the specified step
        u = avg_field[::vector_field_step, ::vector_field_step, 0]
        v = avg_field[::vector_field_step, ::vector_field_step, 1]

        # Create quiver plot with semi-transparent arrows
        quiver = ax.quiver(x_scaled, y_scaled, u, v,
                          angles='xy', scale_units='xy', scale=0.15,
                          alpha=0.4, color='black', width=0.003)

    def init():
        """Initialize the animation."""
        artists = [scatter, payload, trajectory, time_text, params_text, params_text_2]
        if quiver is not None:
            artists.append(quiver)
        return artists
    
    def update(frame):
        """Update the animation for each frame."""
        # Update time counter
        time_text.set_text(f'Frame: {frame}')
        
        # Report progress periodically
        if frame % 50 == 0:
            print(f"Progress: Frame {frame}")
        
        # Update payload
        payload.center = (payload_positions[frame, 0], payload_positions[frame, 1])
        
        # Update payload trajectory
        trajectory_end = min(frame + 1, len(payload_positions))
        trajectory.set_data(
            payload_positions[:trajectory_end, 0],
            payload_positions[:trajectory_end, 1]
        )
        
        # Particle positions & colors update
        scatter.set_offsets(positions[frame])
        scatter.set_color([get_particle_color(cv) for cv in curvity_values[frame]])

        artists = [scatter, payload, trajectory, time_text]
        if quiver is not None:
            artists.append(quiver)
        return artists
    
    # Create animation
    n_frames = positions.shape[0]
    
    sim_seconds_per_real_second = 75 # Increase frame skip for fewer frames to render if its too slow
    target_fps = 15
    
    # Calculate frame skip to maintain consistent sim-time to real-time ratio
    skip = max(1, int(sim_seconds_per_real_second / target_fps))
    
    # Create sequence of frames to include
    frames = range(0, n_frames, skip)
    print(f"Number of frames: {n_frames}")
    
    plt.rcParams['savefig.dpi'] = 170  # Lower dpi for faster rendering
    
    anim = FuncAnimation(
        fig, 
        update, 
        frames=frames, 
        init_func=init, 
        blit=True, 
        interval=120  # Increased from 50
    )
    
    #writer = PillowWriter(fps=target_fps) # for gifs, but its slower
    writer = FFMpegWriter(fps=target_fps, bitrate=1000) # mp4
    
    anim.save(output_file, writer=writer)
    plt.close()
    
    end_time = time.time()
    
    print(f"Animation saved as '{output_file}'")
    print(f"Animation creation time: {end_time - start_time:.2f} seconds")

#####################################################
# Simulation configuration functions                #
#####################################################

def default_payload_params(n_particles=1000, curvity=0, payload_radius=20):
    """Return default parameters for payload transport simulation"""
    return {
        # Global parameters
        'n_particles': n_particles,
        'box_size': 350,
        'grid_size': 350,  # Vector field grid size (matches box_size for 1:1 mapping)
        'dt': 0.01,
        'n_steps': 40000,
        'save_interval': 10,            # Interval for saving data
        'payload_radius': payload_radius,
        'payload_mobility': 1 / payload_radius,
        'stiffness': 25.0,
        # Particle-specific parameters
        'v0': np.ones(n_particles) * 3.75,
        'curvity': np.ones(n_particles) * curvity,     # Curvity array (now computed dynamically from vector fields)
        'particle_radius': np.ones(n_particles) * 1,
        'mobility': np.ones(n_particles) * 1,    # 1/r
        'rot_diffusion': np.ones(n_particles) * 0.05 # 0.05
    }
    

# Currently unused
def heterogeneous_curvity(params):
    """Make a random half of the particles have a positive curvity (default is currently all of them negative)."""
    n_particles = params['n_particles']
    half_particles = n_particles // 2
    curvity = np.ones(n_particles) * params['curvity'][0]  # Use first value as baseline
    curvity[:half_particles] = 0.4 # Make half of the particles have opposite curvity
    params['curvity'] = curvity
    return params

def save_simulation_data(filename, positions, orientations, velocities, payload_positions, 
                        payload_velocities, params, curvity_values):
    """Save simulation data including individual particle parameters."""
    np.savez(
        filename,
        # Frame-specific data
        positions=positions,
        orientations=orientations,
        velocities=velocities,
        payload_positions=payload_positions,
        payload_velocities=payload_velocities,
        curvity_values=curvity_values, # Curvity values over time, for each particle
        # Parameters
        # params['curvity'] accessible through curvity_values[-1]
        v0=params['v0'],
        mobility=params['mobility'],
        particle_radius=params['particle_radius'],
        payload_mobility=params['payload_mobility'],
        payload_radius=params['payload_radius'],
        box_size=params['box_size'],
        dt=params['dt'],
        stiffness=params['stiffness']
    )

def extract_simulation_data(filename):
    """Extract simulation data from a file."""
    data = np.load(filename)
    return data

#####################
# Main execution    #
#####################

if __name__ == "__main__":
    
    # Create directories if they don't exist
    # os.makedirs('./data', exist_ok=True)
    # os.makedirs('./visualizations', exist_ok=True)
    # os.makedirs('./logs', exist_ok=True)
    
    # Run short simulation to trigger numba compilation
    params = default_payload_params(n_particles=10)    
    params['n_steps'] = 10
    run_payload_simulation(params)


    params = default_payload_params(n_particles=1000)
    positions, orientations, velocities, payload_positions, payload_velocities, curvity_values, vector_fields, runtime = run_payload_simulation(params)

    # Timestamp
    T = int(time.time())

    # Save simulation data
    # save_simulation_data(
    #     f'./data/sim_data_T_{T}.npz',
    #     positions, orientations, velocities, payload_positions, payload_velocities, params, curvity_values
    # )

    # Create animation
    create_payload_animation(positions, orientations, velocities, payload_positions, params,
                                curvity_values, f'./visualizations/sim_animation_T_{T}.mp4', 
                                show_avg_vector_field=True, vector_field_step=20, vector_fields=vector_fields)

    

    
    # Create log file
    # log_file = f'./logs/log_{T}.txt'
    # with open(log_file, 'a') as f:
        # f.write("DEFAULT SETTINGS\n")
        # f.write(f"Timestamp: {T}\n")
        # f.write(f"Params: {params}\n")
        # f.write(f"All runtimes: {runtimes}\n")
        # f.write(f"Average runtime: {np.mean(runtimes)}\n")
    
    # Save simulation data
    # save_simulation_data(
    #     # Include timestamp in filename
    #     'data/sim_data_{timestamp}.npz'.format(timestamp=T),
    #     positions, orientations, velocities, payload_positions, payload_velocities, params,
    #     curvity_values
    # )
    
    # Create animation with frame-specific curvity values
    # create_payload_animation(positions, orientations, velocities, payload_positions, params, 
    #                          curvity_values, 'visualizations/sim_animation_{timestamp}.mp4'.format(timestamp=T))
    
    print("Payload simulation and animation completed successfully!")