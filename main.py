import numpy as np
import time
import os

from src.runner import run_payload_simulation
from src.visualization import create_payload_animation
from src.genes import crossover_and_mutate


# Hyperparameters

# Set random seed
RANDOM_SEED = None

# Simulation parameters
N_PARTICLES = 200 #800 # 200
BOX_SIZE = 100 #300 # 100
N_STEPS = 100000
SAVE_INTERVAL = 10
DT = 0.01

# Particle parameters
PARTICLE_RADIUS = 1.0
PARTICLE_V0 = 5.0
PARTICLE_MOBILITY = 1.0
ROTATIONAL_DIFFUSION = 0.1650 #0.05

MAX_CURVITY = 0.625 #1.0
MIN_CURVITY = -0.747 #-1.0
MID_CURVITY = -0.640 #0.0

# Payload parameters
PAYLOAD_RADIUS = 10 #20 #10 for opt
PAYLOAD_MOBILITY = 1 / PAYLOAD_RADIUS
PAYLOAD_START_POSITION = np.array([20.0, 20.0])

# Force parameters
STIFFNESS = 25.0

# Goal parameters
GOAL_POSITION = np.array([83.3, 83.3])
PARTICLE_VIEW_RANGE = 20 #30 #20 #0.2 * BOX_SIZE  # Range for goal detection
SCORE_AND_POLARITY_UPDATE_INTERVAL = 20  # How often to update scores & polarity (timesteps)
END_WHEN_GOAL_REACHED = True        # If True, simulation ends when payload reaches goal

# Wall configuration
# Example walls:
WALLS = np.array([
    # Boundary walls
    [0, 0, 0, BOX_SIZE],
    [0, 0, BOX_SIZE, 0],
    [BOX_SIZE, BOX_SIZE, 0, BOX_SIZE],
    [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0],
    # Maze walls
    # [BOX_SIZE*0.33, BOX_SIZE*0.66, BOX_SIZE, BOX_SIZE*0.66],
    # [0, BOX_SIZE*0.25, BOX_SIZE*0.55, BOX_SIZE*0.25], # bottom wall
    # [BOX_SIZE*0.375, BOX_SIZE, BOX_SIZE*0.375, BOX_SIZE*0.45], # top left wall
    # [BOX_SIZE*0.75, BOX_SIZE, BOX_SIZE*0.75, BOX_SIZE*0.45], # top right wall
    [0, BOX_SIZE*0.33, BOX_SIZE*0.66, BOX_SIZE*0.33],
    [BOX_SIZE*0.33, BOX_SIZE*0.66, BOX_SIZE, BOX_SIZE*0.66],
], dtype=np.float64)
# WALLS = None
# WALLS = np.array([
#     # Boundary walls
#     [0, 0, 0, BOX_SIZE],
#     [0, 0, BOX_SIZE, 0],
#     [BOX_SIZE, BOX_SIZE, 0, BOX_SIZE],
#     [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0],
#     # Inverted Y shape walls
#     # [2 * BOX_SIZE/6, BOX_SIZE, 4 * BOX_SIZE/6, BOX_SIZE], #top wall
#     [2.2 * BOX_SIZE/6, 4 * BOX_SIZE/7, 2.2 * BOX_SIZE/6, BOX_SIZE], #top left
#     [3.8 * BOX_SIZE/6, 4 * BOX_SIZE/7, 3.8 * BOX_SIZE/6, BOX_SIZE], #top right
#     [2.2 * BOX_SIZE/6, 4 * BOX_SIZE/7, 0, 4 * BOX_SIZE/7], # left shoulder
#     [3.8 * BOX_SIZE/6, 4 * BOX_SIZE/7, BOX_SIZE, 4 * BOX_SIZE/7], # right shoulder
#     # [0, 4 * BOX_SIZE/7, 0, 0], #bot left
#     # [BOX_SIZE, 4*BOX_SIZE/7, BOX_SIZE, 0], #bot right
#     # [0, 0, BOX_SIZE, 0], #bot
#     [2 * BOX_SIZE/7, 2.5 * BOX_SIZE/7, 2 * BOX_SIZE/7, 0], #inner left
#     [2 * BOX_SIZE/7, 2.5 * BOX_SIZE/7, 5 * BOX_SIZE/7, 2.5 * BOX_SIZE/7], #inner top
#     [5 * BOX_SIZE/7, 2.5 * BOX_SIZE/7, 5 * BOX_SIZE/7, 0], #inner right
# ], dtype=np.float64)
WALLS = np.array([
    # Boundary walls
    [0, 0, 0, BOX_SIZE],
    [0, 0, BOX_SIZE, 0],
    [BOX_SIZE, BOX_SIZE, 0, BOX_SIZE],
    [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0],
    # walls inside
    # [0, 33.3, 66.6, 33.3],
    # [33.3, 66.6, 100.0, 66.3]
], dtype=np.float64)


# Visualization parameters
SHOW_VECTORS = False              # Display polarity vectors as arrows
COLOR_BY_SCORE = False           # If True: color by score, if False: color by curvity
OUTPUT_FILENAME = "D:/PostThesis/visualizations/CAS_test_small_optimized.mp4"           # If None, uses timestamp. Otherwise specify path.

# Data saving
SAVE_DATA = False
DATA_OUTPUT_PATH = None                   # If None, uses timestamp. Otherwise specify path.

# Genetic algorithm parameters
N_GENERATIONS = 16
POPULATION_SIZE = 12
MUTATION_PROBABILITY = 0.5
OPTIMIZATION_RESULTS_FILE = "optimization_results_3.txt"


# Genetic Algorithm Helper Functions                

def create_random_gene():
    """
    Create a valid random gene for hyperparameter optimization.
    Gene format: [max_curvity, min_curvity, mid_curvity, rot_diffusion]
    """
    # Generate 3 random curvity values in [-1, 1], sort them
    curvity_vals = sorted(np.random.uniform(-1, 1, 3))
    max_c = curvity_vals[2]  # largest
    min_c = curvity_vals[0]  # smallest
    mid_c = curvity_vals[1]  # middle

    # Log-uniform rot_diffusion in [0.001, 0.3]
    rot_diff = np.exp(np.random.uniform(np.log(0.001), np.log(0.3)))

    return np.array([max_c, min_c, mid_c, rot_diff])


def run_single_simulation(base_params, gene, verbose=True):
    """
    Run simulation with gene parameters, return steps taken.

    Args:
        base_params: Base simulation parameters dict
        gene: [max_curvity, min_curvity, mid_curvity, rot_diffusion]
        verbose: If True, print simulation output

    Returns:
        steps_taken: Number of steps until goal reached (or max steps)
    """
    # Copy params to avoid modifying original
    params = base_params.copy()

    # Apply gene parameters
    params['max_curvity'] = gene[0]
    params['min_curvity'] = gene[1]
    params['mid_curvity'] = gene[2]
    params['rot_diffusion'] = np.ones(params['n_particles']) * gene[3]

    # Reset payload position for each run
    params['payload_position'] = PAYLOAD_START_POSITION.copy()

    # Run simulation
    result = run_payload_simulation(params)
    steps_taken = result[11]  # final_step is at index 11

    return steps_taken


def run_genetic_optimization(base_params, n_generations=5, population_size=10,
                            mutation_probability=0.3, output_file="optimization_results.txt"):
    """
    Run genetic algorithm optimization for hyperparameters.

    Args:
        base_params: Base simulation parameters dict
        n_generations: Number of generations to evolve
        population_size: Number of individuals per generation (should be 10)
        mutation_probability: Probability of mutation (constant across generations)
        output_file: Path to save results text file

    Returns:
        best_gene: Best performing gene found
        best_steps: Steps taken by best gene
    """
    # Calculate exponential decay constant for sigma (mutation magnitude)
    # sigma_scale decays from 1.0 to ~0.01 by the last generation
    # Formula: sigma_scale(gen) = exp(-decay_constant * gen)
    if n_generations > 1:
        sigma_decay_constant = 4.6 / (n_generations - 1)  # Ensures ~1% of initial by last gen
    else:
        sigma_decay_constant = 0.0

    # Open results file for writing
    results_log = []

    def log(msg):
        """Print and log message."""
        print(msg)
        results_log.append(msg)

    log(f"\n{'='*60}")
    log("GENETIC ALGORITHM OPTIMIZATION")
    log(f"{'='*60}")
    log(f"Generations: {n_generations}")
    log(f"Population size: {population_size}")
    log(f"Mutation probability: {mutation_probability} (constant)")
    log(f"Sigma decay: exponential (1.0 -> ~1% by final generation)")
    log(f"Total simulations: {n_generations * population_size}")
    log(f"{'='*60}\n")

    # Initialize population: gene 0 = defaults, rest = random
    default_gene = np.array([MAX_CURVITY, MIN_CURVITY, MID_CURVITY, ROTATIONAL_DIFFUSION])
    population = [default_gene]
    population += [create_random_gene() for _ in range(population_size - 1)]

    best_overall_gene = None
    best_overall_steps = float('inf')

    for gen in range(n_generations):
        # Calculate decaying sigma scale for this generation
        # gen=0: sigma_scale=1.0 (large mutations), gen=last: sigma_scale~0.01 (tiny nudges)
        current_sigma_scale = np.exp(-sigma_decay_constant * gen)

        log(f"\n--- Generation {gen + 1}/{n_generations} ---")
        log(f"Sigma scale: {current_sigma_scale:.4f}")

        # Run all simulations and collect results
        results = []
        for i, gene in enumerate(population):
            log(f"\n[Gen {gen+1}, Individual {i+1}/{population_size}]")
            log(f"  Gene: max_c={gene[0]:.3f}, min_c={gene[1]:.3f}, "
                f"mid_c={gene[2]:.3f}, rot_diff={gene[3]:.4f}")

            steps = run_single_simulation(base_params, gene)
            results.append((gene.copy(), steps))

            log(f"  Steps taken: {steps}")

            # Track overall best
            if steps < best_overall_steps:
                best_overall_steps = steps
                best_overall_gene = gene.copy()

        # Sort by steps (ascending = fewer steps is better)
        results.sort(key=lambda x: x[1])

        # Log generation summary
        gen_best = results[0][1]
        gen_worst = results[-1][1]
        log(f"\n[Generation {gen+1} Summary]")
        log(f"  Best: {gen_best} steps")
        log(f"  Worst: {gen_worst} steps")
        log(f"  Best gene: max_c={results[0][0][0]:.3f}, min_c={results[0][0][1]:.3f}, "
            f"mid_c={results[0][0][2]:.3f}, rot_diff={results[0][0][3]:.4f}")

        # Select top 2 parents
        parent1 = results[0][0]
        parent2 = results[1][0]

        # Generate new population via crossover + mutation (except for last generation)
        if gen < n_generations - 1:
            # Use next generation's sigma scale for offspring
            next_gen_sigma_scale = np.exp(-sigma_decay_constant * (gen + 1))
            population = crossover_and_mutate(
                parent1, parent2,
                mutation_probability,  # constant probability
                population_size=population_size,
                use_hyperparam_mutation=True,
                sigma_scale=next_gen_sigma_scale  # decaying sigma
            )

    log(f"\n{'='*60}")
    log("OPTIMIZATION COMPLETE")
    log(f"{'='*60}")
    log(f"Best gene found:")
    log(f"  max_curvity:          {best_overall_gene[0]:.6f}")
    log(f"  min_curvity:          {best_overall_gene[1]:.6f}")
    log(f"  mid_curvity:          {best_overall_gene[2]:.6f}")
    log(f"  rotational_diffusion: {best_overall_gene[3]:.6f}")
    log(f"Steps taken: {best_overall_steps}")
    log(f"{'='*60}\n")

    # Write results to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(results_log))
    print(f"Results saved to: {output_file}")

    return best_overall_gene, best_overall_steps


# Main execution

if __name__ == "__main__":

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Create directories if they don't exist
    if SAVE_DATA:
        os.makedirs('./data', exist_ok=True)
    os.makedirs('./visualizations', exist_ok=True)

    # JIT COMPILATION

    print("Compiling JIT functions...")

    # Build parameter dictionary for compilation run
    compile_n_particles = 10
    compile_params = {
        'n_particles': compile_n_particles,
        'box_size': BOX_SIZE,
        'dt': DT,
        'n_steps': 10,
        'save_interval': SAVE_INTERVAL,
        'payload_radius': PAYLOAD_RADIUS,
        'payload_mobility': PAYLOAD_MOBILITY,
        'payload_position': PAYLOAD_START_POSITION,
        'stiffness': STIFFNESS,
        'goal_position': GOAL_POSITION,
        'particle_view_range': PARTICLE_VIEW_RANGE,
        'score_and_polarity_update_interval': SCORE_AND_POLARITY_UPDATE_INTERVAL,
        'end_when_goal_reached': END_WHEN_GOAL_REACHED,
        'walls': WALLS if WALLS is not None else np.zeros((0, 4), dtype=np.float64),
        'v0': np.ones(compile_n_particles) * PARTICLE_V0,
        'curvity': np.zeros(compile_n_particles),
        'particle_radius': np.ones(compile_n_particles) * PARTICLE_RADIUS,
        'mobility': np.ones(compile_n_particles) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(compile_n_particles) * ROTATIONAL_DIFFUSION,
        'max_curvity': MAX_CURVITY,
        'min_curvity': MIN_CURVITY,
        'mid_curvity': MID_CURVITY
    }

    run_payload_simulation(compile_params)
    print("JIT compilation complete.\n")

    # BUILD SIMULATION PARAMETERS

    params = {
        # Global parameters
        'n_particles': N_PARTICLES,
        'box_size': BOX_SIZE,
        'dt': DT,
        'n_steps': N_STEPS,
        'save_interval': SAVE_INTERVAL,
        'payload_radius': PAYLOAD_RADIUS,
        'payload_mobility': PAYLOAD_MOBILITY,
        'payload_position': PAYLOAD_START_POSITION,
        'stiffness': STIFFNESS,

        # Goal parameters
        'goal_position': GOAL_POSITION,
        'particle_view_range': PARTICLE_VIEW_RANGE,
        'score_and_polarity_update_interval': SCORE_AND_POLARITY_UPDATE_INTERVAL,
        'end_when_goal_reached': END_WHEN_GOAL_REACHED,

        # Wall parameters
        'walls': WALLS if WALLS is not None else np.zeros((0, 4), dtype=np.float64),

        # Particle-specific parameters (arrays)
        'v0': np.ones(N_PARTICLES) * PARTICLE_V0,
        'curvity': np.zeros(N_PARTICLES),  # Computed dynamically from score & polarity
        'particle_radius': np.ones(N_PARTICLES) * PARTICLE_RADIUS,
        'mobility': np.ones(N_PARTICLES) * PARTICLE_MOBILITY,
        'rot_diffusion': np.ones(N_PARTICLES) * ROTATIONAL_DIFFUSION,
        
        'max_curvity': MAX_CURVITY,
        'min_curvity': MIN_CURVITY,
        'mid_curvity': MID_CURVITY
    }

    # RUN GENETIC OPTIMIZATION

    # best_gene, best_steps = run_genetic_optimization(
    #     params,
    #     n_generations=N_GENERATIONS,
    #     population_size=POPULATION_SIZE,
    #     mutation_probability=MUTATION_PROBABILITY,
    #     output_file=OPTIMIZATION_RESULTS_FILE
    # )

    # RUN SIMULATION

    # Run the simulation
    result = run_payload_simulation(params)
    
    # Unpack results
    (saved_positions, saved_orientations, saved_velocities,
     saved_payload_positions, saved_payload_velocities, saved_curvity,
     saved_polarity, saved_particle_scores, particle_scores, polarity,
     simulation_time, final_step) = result

    print(f"\nSimulation completed in {simulation_time:.2f} seconds")
    print(f"Final step: {final_step}")

    # Save data if enabled
    if SAVE_DATA:
        from src.runner import save_simulation_data
        save_simulation_data(
            DATA_OUTPUT_PATH,
            saved_positions, saved_orientations, saved_velocities,
            saved_payload_positions, saved_payload_velocities,
            params, saved_curvity, saved_polarity, saved_particle_scores
        )
        print(f"Data saved to: {DATA_OUTPUT_PATH}")

    # Create visualization
    create_payload_animation(
        saved_positions, saved_orientations, saved_velocities,
        saved_payload_positions, params, saved_curvity,
        output_file=OUTPUT_FILENAME,
        show_vectors=SHOW_VECTORS,
        polarity=saved_polarity,
        particle_scores=saved_particle_scores if COLOR_BY_SCORE else None
    )

    print("Simulation and visualization completed successfully!")
