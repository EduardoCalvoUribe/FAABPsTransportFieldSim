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


def mutate_hyperparams(gene: np.ndarray, mutation_probability: float = 0.3) -> np.ndarray:
    """
    Mutate hyperparameter gene with parameter-specific logic.

    Gene format: [max_curvity, min_curvity, mid_curvity, rot_diffusion]

    Args:
        gene: Hyperparameter gene array
        mutation_probability: Probability of mutating each gene value (0 to 1)

    Returns:
        Mutated gene with valid parameter constraints
    """
    if not 0 <= mutation_probability <= 1:
        raise ValueError("Mutation probability must be between 0 and 1")

    mutated = gene.copy()

    # Apply mutation to each gene position
    for i in range(len(gene)):
        if np.random.random() < mutation_probability:
            if i < 3:  # curvity params (max, min, mid)
                # Gaussian mutation with scale 0.2
                mutated[i] += np.random.normal(0, 0.2)
                # Clamp to valid range [-1, 1]
                mutated[i] = np.clip(mutated[i], -1, 1)
            else:  # rot_diffusion (index 3)
                # Multiplicative mutation in log space
                mutated[i] *= np.exp(np.random.normal(0, 0.3))
                # Clamp to valid range [0.01, 0.3]
                mutated[i] = np.clip(mutated[i], 0.01, 0.3)

    # Re-sort curvity params to maintain min < mid < max constraint
    curvity_vals = sorted(mutated[:3])
    mutated[0] = curvity_vals[2]  # max (largest)
    mutated[1] = curvity_vals[0]  # min (smallest)
    mutated[2] = curvity_vals[1]  # mid (middle)

    return mutated


def crossover_and_mutate(parent1: np.ndarray, parent2: np.ndarray,
                         mutation_probability: float,
                         use_hyperparam_mutation: bool = False) -> List[np.ndarray]:
    """
    Perform crossover followed by mutation on two parent genes.

    Args:
        parent1: First parent gene
        parent2: Second parent gene
        mutation_probability: Probability of mutating each gene value
        use_hyperparam_mutation: If True, use mutate_hyperparams() for
            hyperparameter optimization (gene format: [max_c, min_c, mid_c, rot_diff])

    Returns:
        List of 6 mutated offspring genes (3 mutations of each crossover result)
    """
    # Perform crossover
    offspring1, offspring2 = crossover(parent1, parent2)

    # Select mutation function
    mutation_fn = mutate_hyperparams if use_hyperparam_mutation else mutate

    # Create 3 mutations of each offspring
    mutations_offspring1 = [mutation_fn(offspring1, mutation_probability) for _ in range(3)]
    mutations_offspring2 = [mutation_fn(offspring2, mutation_probability) for _ in range(3)]

    # Combine all 6 offspring
    all_offspring = mutations_offspring1 + mutations_offspring2

    return all_offspring
