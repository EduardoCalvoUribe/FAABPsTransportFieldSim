"""
Genetic algorithm functions for hyperparameter optimization.
"""
import numpy as np
from typing import Tuple, List


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform crossover between two parent genes by swapping 50% of their values.

    Args:
        parent1: First parent gene (numpy array)
        parent2: Second parent gene (numpy array)

    Returns:
        Tuple of two offspring genes
    """
    if parent1.shape != parent2.shape:
        raise ValueError("Parent genes must have the same shape")

    # Create copies to avoid modifying the original parents
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Randomly select 50% of positions to swap
    gene_length = len(parent1)
    num_swaps = gene_length // 2
    swap_indices = np.random.choice(gene_length, size=num_swaps, replace=False)

    # Swap the selected positions
    offspring1[swap_indices] = parent2[swap_indices]
    offspring2[swap_indices] = parent1[swap_indices]

    return offspring1, offspring2


def mutate(gene: np.ndarray, mutation_probability: float) -> np.ndarray:
    """
    Mutate a gene with the given probability.

    Args:
        gene: Gene to mutate (numpy array)
        mutation_probability: Probability of mutating each gene value (0 to 1)

    Returns:
        Mutated gene
    """
    if not 0 <= mutation_probability <= 1:
        raise ValueError("Mutation probability must be between 0 and 1")

    mutated_gene = gene.copy()

    # Determine which positions to mutate
    mutation_mask = np.random.random(len(gene)) < mutation_probability

    # TODO: Implement specific mutation logic here
    # Placeholder: Add small random noise to mutated positions
    mutated_gene[mutation_mask] += np.random.normal(0, 0.1, size=np.sum(mutation_mask))

    return mutated_gene


def mutate_hyperparams(gene: np.ndarray, mutation_probability: float = 0.3,
                       sigma_scale: float = 1.0) -> np.ndarray:
    """
    Mutate hyperparameter gene with parameter-specific logic.

    Gene format: [max_curvity, min_curvity, mid_curvity, rot_diffusion]

    Args:
        gene: Hyperparameter gene array
        mutation_probability: Probability of mutating each gene value (0 to 1)
        sigma_scale: Scale factor for mutation sigma (1.0 = large mutations,
                     small values = tiny nudges). Decays over generations.

    Returns:
        Mutated gene with valid parameter constraints
    """
    if not 0 <= mutation_probability <= 1:
        raise ValueError("Mutation probability must be between 0 and 1")

    mutated = gene.copy()

    # Base sigma values (used when sigma_scale = 1.0)
    base_sigma_curvity = 0.4
    base_sigma_log = 0.6

    # Scale sigmas by sigma_scale
    sigma_curvity = base_sigma_curvity * sigma_scale
    sigma_log = base_sigma_log * sigma_scale

    # Apply mutation to each gene position
    for i in range(len(gene)):
        if np.random.random() < mutation_probability:
            if i < 3:  # curvity params (max, min, mid)
                # Gaussian mutation with decaying scale
                mutated[i] += np.random.normal(0, sigma_curvity)
                # Clamp to valid range [-1, 1]
                mutated[i] = np.clip(mutated[i], -1, 1)
            elif i == 3:  # rot_diffusion
                # Multiplicative mutation in log space with decaying scale
                mutated[i] *= np.exp(np.random.normal(0, sigma_log))
                # Clamp to valid range [0.001, 0.3]
                mutated[i] = np.clip(mutated[i], 0.001, 0.3)

    # Re-sort curvity params to maintain min < mid < max constraint
    curvity_vals = sorted(mutated[:3])
    mutated[0] = curvity_vals[2]  # max (largest)
    mutated[1] = curvity_vals[0]  # min (smallest)
    mutated[2] = curvity_vals[1]  # mid (middle)

    return mutated


def crossover_and_mutate(parent1: np.ndarray, parent2: np.ndarray,
                         mutation_probability: float,
                         population_size: int = 6,
                         use_hyperparam_mutation: bool = False,
                         sigma_scale: float = 1.0) -> List[np.ndarray]:
    """
    Perform crossover followed by mutation on two parent genes.

    Args:
        parent1: First parent gene
        parent2: Second parent gene
        mutation_probability: Probability of mutating each gene value
        population_size: Number of offspring to generate
        use_hyperparam_mutation: If True, use mutate_hyperparams() for
            hyperparameter optimization (gene format: [max_c, min_c, mid_c, rot_diff])
        sigma_scale: Scale factor for mutation sigma (1.0 = large mutations,
                     small values = tiny nudges). Only used with hyperparam mutation.

    Returns:
        List of mutated offspring genes
    """
    # Perform crossover
    offspring1, offspring2 = crossover(parent1, parent2)

    # Split population_size between the two crossover offspring
    n_from_offspring1 = population_size // 2
    n_from_offspring2 = population_size - n_from_offspring1

    # Create mutations of each offspring
    if use_hyperparam_mutation:
        mutations_offspring1 = [mutate_hyperparams(offspring1, mutation_probability, sigma_scale)
                                for _ in range(n_from_offspring1)]
        mutations_offspring2 = [mutate_hyperparams(offspring2, mutation_probability, sigma_scale)
                                for _ in range(n_from_offspring2)]
    else:
        mutations_offspring1 = [mutate(offspring1, mutation_probability) for _ in range(n_from_offspring1)]
        mutations_offspring2 = [mutate(offspring2, mutation_probability) for _ in range(n_from_offspring2)]

    # Combine all offspring
    all_offspring = mutations_offspring1 + mutations_offspring2

    return all_offspring
