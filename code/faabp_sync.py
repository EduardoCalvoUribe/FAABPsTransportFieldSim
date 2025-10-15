
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle
import time
import os
import math
from numba import njit, float64, int64



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
        
        # cell_id = cell_y * n_cells + cell_x 
        # Optionally, use row-major ordering; converts 2D grid coords into a 1D index.
        # Every x, y pair maps to a different number, and it's reversible
        # x = cell_id % n_cells
        # y = cell_id // n_cells
        
        list_next[i] = head[cell_x, cell_y] # head[cell_id] if row-major ordering
        head[cell_x, cell_y] = i # head[cell_id]
        
        # Example:
        
        # Initial state: Empty cell (cell coords: = 1, 1)
        # head[1, 1] = -1
        # list_next = [-1, -1, -1, ...]

        # Add particle 10 to cell (1, 1):
        # head[1, 1] = 10
        # list_next[10] = -1

        # Add particle 7 to cell (1, 1):
        # list_next[7] = 10  (7 points to 10)
        # head[1, 1] = 7        (7 is new head)

        # Add particle 3 to cell (1, 1):
        # list_next[3] = 7   (3 points to 7)
        # head[1, 1] = 3        (3 is new head)
        
        # Later, when we want to find all particles in cell (1, 1), we can start at head[1, 1] and follow the links:
        # j = head[1, 1]
        # while j != -1:
        #     # Do something with j
        #     j = list_next[j]
    
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
def has_line_of_sight(positions, i, goal_position, payload_pos, payload_radius):
    """Returns True if particle has line of sight with the goal, False otherwise. Does not use periodic boundaries for obvious reasons"""
    # I think I can vectorize this implementation to make it faster w/ Numba
    # but I dont know if the gains are actually worth the effort.
    # Would be less readable too
    
    # Points
    x_i, y_i = positions[i]
    x_goal, y_goal = goal_position
    x_p, y_p = payload_pos
    
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
    
    # Coefficients of quadratic equations
    a = dx**2 + dy**2 # Length of particle-goal line
    b = 2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - payload_radius**2
    
    # Discriminant
    discriminant = b**2 - 4 * a * c
    
    if discriminant < 0:
        return True
    else:
        sqrt_discriminant = math.sqrt(discriminant)
        # if a == 0:
        #     a = 0.00000001 # was occuring when goal was attached to particle
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        
        if (-1 <= t1 <= 0) or (-1 <= t2 <= 0):
            return False
        else:
            return True

# Unused
@njit
def update_curvity_half_box(positions, i, box_size):
    """Updates curvity based on position. For testing purposes"""
    # If the particle is in the top half of the box, it has positive curvity. Otherwise, negative
    if positions[i, 1] > box_size / 2:
        return True
    else:
        return False
    
@njit(fastmath=True)
def simulate_single_step(positions, orientations, velocities, payload_pos, payload_vel, 
                         radii, v0s, mobilities, payload_mobility, curvity, curvity_on, curvity_off,
                         stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles, step):
    """Simulate a single time step"""
    # Compute forces on particles and payload
    particle_forces, payload_force = compute_all_forces(
        positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size
    )
    
    # Update particle orientations
    orientations = update_orientation_vectors(
        orientations, particle_forces, curvity, dt, rot_diffusion, n_particles
    )
    
    # Update particle positions
    for i in range(n_particles):
        
        # Update curvity
        goal_position = [4*(box_size / 5), 4*(box_size / 5)] # static location for now, top right corner.
        # goal_position = positions[0]
        
        
        if has_line_of_sight(positions, i, goal_position, payload_pos, payload_radius):
            # In the light
                   
            # curvity[i] = curvity_on
        
            v0s[i] = curvity_on
        else:
            # In the shadow
        
            # curvity[i] = curvity_off
        
            v0s[i] = curvity_off
        
        # Self-propulsion velocity with particle-specific v0
        self_propulsion = v0s[i] * orientations[i]
        
        # Force-induced velocity with particle-specific mobility
        force_velocity = mobilities[i] * particle_forces[i]
        
        # Total velocity
        velocities[i] = self_propulsion + force_velocity # + velocities[i] / 10 # added velocities[i] to test inertia!!
        
        # Update position
        positions[i] += velocities[i] * dt
    
    # INERTIA TEST
    # payload_accel = payload_mobility * payload_force / 10 # 10 is mass. picked arbitrarily... could be scaled properly
    # payload_vel = payload_vel + payload_accel * dt
    
    # Update payload
    payload_vel = payload_mobility * payload_force
    
    # if step >= 0 and step <= 9: # BIG PUSH!!! -----------------------------------
    #     payload_vel = payload_vel + np.array([10, 10])
    #     print(f"Step {step}: PUSH!")
    # if step < 10000:
    #     payload_vel = np.array([0, 0], dtype=np.float64)
    # if step == 30000: # BIG PUSH!!! -----------------------------------
    #     payload_vel = payload_vel + np.array([5, 5], dtype=np.float64)
    #     print(f"Step {step}: PUSH!")
        
    payload_pos += payload_vel * dt
    
    # Apply periodic boundary conditions
    positions = positions % box_size
    payload_pos = payload_pos % box_size
    
    return positions, orientations, velocities, payload_pos, payload_vel

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
    
    # Initialize particle positions, orientations, and velocities
    positions = np.random.uniform(0, box_size, (n_particles, 2))
    orientations = np.zeros((n_particles, 2))
    velocities = np.zeros((n_particles, 2))
    
    # Initialize random orientations
    for i in range(n_particles):
        angle = np.random.uniform(0, 2*np.pi)
        orientations[i] = np.array([np.cos(angle), np.sin(angle)])
    
    # Initialize payload location. Bottom left corner for now
    payload_pos = np.array([box_size/4, box_size/4])
    payload_vel = np.zeros(2)
    
    # Define goal position (currently hardcoded)
    goal_position = np.array([4*(box_size / 5), 4*(box_size / 5)])
    
    # Pre-allocate arrays for storing simulation data
    n_saves = n_steps // save_interval + 1
    saved_positions = np.zeros((n_saves, n_particles, 2))
    saved_orientations = np.zeros((n_saves, n_particles, 2))
    saved_velocities = np.zeros((n_saves, n_particles, 2))
    saved_payload_positions = np.zeros((n_saves, 2))
    saved_payload_velocities = np.zeros((n_saves, 2))
    saved_curvity = np.zeros((n_saves, n_particles))
    
    # Store initial state
    saved_positions[0] = positions.copy()
    saved_orientations[0] = orientations.copy()
    saved_velocities[0] = velocities.copy()
    saved_payload_positions[0] = payload_pos.copy()
    saved_payload_velocities[0] = payload_vel.copy()
    saved_curvity[0] = params['curvity'].copy()
    
    # Run simulation
    start_time = time.time()
    save_idx = 1
    
    for step in range(1, n_steps + 1):
        # Unified simulation step
        positions, orientations, velocities, payload_pos, payload_vel = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel, 
            params['particle_radius'], params['v0'], params['mobility'], params['payload_mobility'], 
            params['curvity'], params['curvity_on'], params['curvity_off'], params['stiffness'], 
            params['box_size'], params['payload_radius'], params['dt'], params['rot_diffusion'], n_particles, step
        )
        
        # Check if payload has reached goal for early stopping
        distance_to_goal = np.sqrt(np.sum((payload_pos - goal_position)**2))
        if distance_to_goal <= params['payload_radius']:
            print(f"Goal reached at step {step}!")
            # Save final state
            saved_positions[save_idx] = positions
            saved_orientations[save_idx] = orientations
            saved_velocities[save_idx] = velocities
            saved_payload_positions[save_idx] = payload_pos
            saved_payload_velocities[save_idx] = payload_vel
            saved_curvity[save_idx] = params['curvity'].copy()
            save_idx += 1
            # Trim arrays to actual size
            saved_positions = saved_positions[:save_idx]
            saved_orientations = saved_orientations[:save_idx]
            saved_velocities = saved_velocities[:save_idx]
            saved_payload_positions = saved_payload_positions[:save_idx]
            saved_payload_velocities = saved_payload_velocities[:save_idx]
            saved_curvity = saved_curvity[:save_idx]
            break
        
        # Save data at specified intervals
        if step % save_interval == 0:
            saved_positions[save_idx] = positions
            saved_orientations[save_idx] = orientations
            saved_velocities[save_idx] = velocities
            saved_payload_positions[save_idx] = payload_pos
            saved_payload_velocities[save_idx] = payload_vel
            saved_curvity[save_idx] = params['curvity'].copy()
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
        end_time - start_time
    )

def create_payload_animation(positions, orientations, velocities, payload_positions, params, 
                            curvity_values, output_file='visualizations/payload_animation_00.mp4'):
    """Create an animation of the payload transport simulation."""
    print("Creating animation...")
    
    start_time = time.time()
    
    # Extract parameters
    box_size = params['box_size']
    payload_radius = params['payload_radius']
    n_particles = params['n_particles']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set axis limits
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title('FAABP Cooperative Transport Simulation')
    ax.grid(True, alpha=0.3)
    
    # Color of particle based on curvity value (or is leader)
    def get_particle_color(curvity_value, particle_index):
        # if particle_index == 0:
        #     return 'green'
        return 'red' if curvity_value > 0 else 'darkblue'
    
    default_curvity = params['curvity'][0]
    particle_colors = [get_particle_color(default_curvity, i) for i in range(n_particles)] # -------------- for v0_2
    # Using scatter for particles instead of individual circles
    # particle_colors = [get_particle_color(curvity_values[0, i], i) for i in range(n_particles)]
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
    
    # TEMPORARY: Static goal indication
    goal_position = [4*(box_size / 5), 4*(box_size / 5)] 
    goal = Circle(
        (goal_position[0], goal_position[1]),
        radius=2,
        color='green'
    )
    ax.add_patch(goal)
    
    # Create payload trajectory
    trajectory, = ax.plot(
        payload_positions[0:1, 0], 
        payload_positions[0:1, 1], 
        'k--', 
        alpha=0.5, 
        linewidth=1.0
    )
    
    # Add parameters text
    params_text = ax.text(-0.02, -0.065, f'n_particles: {n_particles}, curvity with LOS: {params["curvity_on"]}, curvity without LOS: {params["curvity_off"]}, particle radius: {params["particle_radius"][0]}, payload radius: {payload_radius}', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    params_text_2 = ax.text(-0.02, -0.093, f'orientational noise: {params["rot_diffusion"][0]}, particle mobility: {params["mobility"][0]}, payload mobility: {params["payload_mobility"]}', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    
    # Add time counter
    time_text = ax.text(0.02, 0.98, 'Frame: 0', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    
    def init():
        """Initialize the animation."""
        return [scatter, payload, trajectory, time_text, params_text, params_text_2]
    
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
        scatter.set_color([get_particle_color(cv, i) for i, cv in enumerate(curvity_values[frame])])

        return [scatter, payload, trajectory, time_text]
    
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

def default_payload_params(n_particles=1000, curvity_on=-0.3, curvity_off=-0.3, curvity=0, payload_radius=20):
    """Return default parameters for payload transport simulation"""
    return {
        # Global parameters
        'n_particles': n_particles,    
        'box_size': 350,               
        'dt': 0.01,                  
        'n_steps': 30000,               
        'save_interval': 10,            # Interval for saving data
        'payload_radius': payload_radius,        
        'payload_mobility': 0.05,        # Manually kept to 1/r
        'stiffness': 25.0,              
        # Particle-specific parameters
        'v0': np.ones(n_particles) * 3.75,           
        'mobility': np.ones(n_particles) * 1,    # Manually kept to 1/r
        'curvity': np.ones(n_particles) * curvity,     # Curvity array for all particles. Default is all 0 (but doesnt matter since it gets updated every step)
        'curvity_on': curvity_on, # When there is line of sight
        'curvity_off': curvity_off, # When there is no line of sight
        'particle_radius': np.ones(n_particles) * 1, 
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


    # params = default_payload_params(n_particles=1000)
    # positions, orientations, velocities, payload_positions, payload_velocities, curvity_values, runtime = run_payload_simulation(params)
    
    # Save simulation data
    # save_simulation_data(
    #     f'D:/ThesisData/data/tests/sim_data_vOn_{3.75}_vOff_{10}_curvity_{0}_payloadinertia.npz',
    #     positions, orientations, velocities, payload_positions, payload_velocities, params, curvity_values
    # )
    
    # # Create animation
    # create_payload_animation(positions, orientations, velocities, payload_positions, params, 
    #                             curvity_values, f'D:/ThesisData/visualizations/tests/sim_animation_vOn_{3.75}_vOff_{10}_curvity_{0}_pinertia.mp4')

    # Run simulations
   
    vels = [
        [5, 10],
        [5, 5],
        [10, 10],
        [10, 5],
    ]
    for curvity in [1, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8, -1]: 
        for pair in vels: 
            print(f"Running simulation with vOn={pair[0]}, vOff={pair[1]}, curvity={curvity}")
            params = default_payload_params(curvity_on=pair[0], curvity_off=pair[1], curvity=curvity)
            positions, orientations, velocities, payload_positions, payload_velocities, curvity_values, runtime = run_payload_simulation(params)
            
            # Save simulation data
            save_simulation_data(
                f'D:/ThesisData/data/dynamic_v0_2/sim_data_vOn_{pair[0]}_vOff_{pair[1]}_curvity_{curvity}.npz',
                positions, orientations, velocities, payload_positions, payload_velocities, params, curvity_values
            )
            # Create animation
            # create_payload_animation(positions, orientations, velocities, payload_positions, params, 
            #                             curvity_values, f'D:/ThesisData/visualizations/dynamic_v0_2/sim_animation_vOn_{pair[0]}_vOff_{pair[1]}_curvity_{curvity}.mp4')

                
    
    # Timestamp
    # T = int(time.time())
    
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