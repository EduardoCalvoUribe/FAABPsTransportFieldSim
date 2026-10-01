# Goal-Seeking Algorithm using Polarity 

## Overview
Each particle has two key properties:
- **Score (s)**: An integer representing the particle's "distance" from the goal in discrete hops
- **Polarity (p)**: A unit vector that influences the particle's curvity (and therefore turning behavior)

## Algorithm Structure

### 1. Particle Properties
Every particle `i` has:
- **s_i** ∈ ℤ: Score (initialized to 9999)
- **p_i** ∈ ℝ²: Polarity unit vector (||p_i|| = 1)
- **e_i** ∈ ℝ²: Heading/orientation vector (||e_i|| = 1)
- **r**: View range (hyperparameter, `particle_view_range`; default: 1.5 × maze cell size)

### 2. Core Update Rules (Executed Every `score_and_polarity_update_interval` Steps)

For each particle `i` at position **x_i**:

#### **Case A: Line of Sight to Goal**
If ||**x_goal** - **x_i**|| ≤ r (goal within view range) AND no walls or payload block the line of sight:

```
s_i = 0
p_i = (**x_goal** - **x_i**) / ||**x_goal** - **x_i**||
```

**Meaning**: Particle has direct, unobstructed access to goal; points directly toward it with score 0.

---

#### **Case B: No Line of Sight to Goal**
If goal is out of range OR line of sight is blocked by walls or payload:

##### **Step 1: Find Neighbors Within Range**
Define neighbor set:
```
N_i = {j ≠ i : ||**x_j** - **x_i**|| ≤ r AND not separated by wall}
```
**Note**: Uses periodic boundary conditions for distance calculation. Particles separated by walls along the shortest periodic path are excluded from the neighbor set.

---

##### **Step 2: Score Calculation**

```
s_i = {
    s* + 1,     if N_i ≠ ∅
    9999,       if N_i = ∅
}

where s* = min{s_j : j ∈ N_i}
```

**Meaning**:
- Score is minimum neighbor score plus one (gradient descent toward goal)
- If isolated (no neighbors), score resets to 9999
- **Halt condition**: If s_i > 20000, terminate simulation (indicates unreachable goal)

---

##### **Step 3: Polarity — Direction Toward Min-Score Neighbors**

**p_i** points toward the average position of all min-score neighbors.

Two-pass computation over N_i (using scores from the previous update):

```
Pass 1 — find minimum score:
    s* = min{s_j : j ∈ N_i}

Pass 2 — average displacement to all min-score neighbors:
    J = {j ∈ N_i : s_j = s*}
    avg_delta = (1/|J|) · Σ_{j ∈ J} (x_j - x_i)_periodic

p_i = avg_delta / ||avg_delta||    (zero vector if ||avg_delta|| = 0)
```

where `(x_j - x_i)_periodic` is the minimum-image displacement under periodic boundaries.

- All particles update in parallel using a frozen snapshot of scores from the **previous** update, so the score wave advances by exactly one hop per update interval.
- Both passes exclude neighbors blocked by walls along the periodic shortest path.

---

### 3. Curvity Calculation

Once **p_i** is computed, curvity κ_i is a piecewise linear function of the dot product **e_i · p_i**:

```
Let α = e_i · p_i  ∈ [-1, 1]

κ_i = mid_curvity + (min_curvity - mid_curvity) · α     if α ≥ 0
      max_curvity + (mid_curvity - max_curvity) · (α+1) if α < 0
```

Breakpoints:
- α = +1 (heading aligned with polarity)      → κ = min_curvity
- α =  0 (heading perpendicular to polarity)  → κ = mid_curvity
- α = −1 (heading anti-aligned with polarity) → κ = max_curvity

**Effect on dynamics**:
Curvity influences orientation update via torque:
```
τ_i = κ_i · (e_i × F_i)
```
where F_i is the net force on particle i.

---

## Implementation Details

### Efficient Neighbor Search
Uses cell-list algorithm with O(N) complexity:
- Cell size = r (particle view range)
- Search 3×3 neighborhood of cells
- **Uses periodic wrapping** for neighbor search
- Particles separated by walls along shortest periodic path are excluded

### Update Frequency
- Parameter: `score_and_polarity_update_interval` (default: 10 timesteps)
- Polarity vectors and scores update every `score_and_polarity_update_interval` steps
- Reduces computational cost while maintaining gradient propagation

### Initialization
- All particles start with s_i = 9999, p_i = (cos(π/4), sin(π/4))
- Goal position: default (4/5 × box_size, 4/5 × box_size)
- Score propagates outward from goal over time

---

## Key Properties

1. **Gradient Formation**: Scores form a discrete distance field (BFS wavefront) pointing toward goal
2. **Information Propagation**: Score=0 spreads from goal-visible particles at ~1 hop per update interval
3. **Polarity Follows Gradient**: Each particle's polarity points toward the average position of its lowest-score visible neighbors
4. **Parallel Update from Frozen Scores**: All particles read last-step's scores simultaneously; wave advances one hop per interval
5. **Isolation Handling**: Particles with no wall-unblocked neighbors in range reset to s=9999, preventing stale information

---

## Physical Interpretation

This algorithm creates **emergent cooperative transport**:
1. Particles with clear line of sight to goal (within range, unobstructed by walls/payload) set s=0 and polarity pointing to goal
2. Their visible, wall-unblocked neighbors set s=1 and polarity pointing toward those s=0 particles
3. The score wave cascades outward hop-by-hop (one hop per `score_and_polarity_update_interval` timesteps), forming a discrete distance field
4. Each particle's polarity points toward the average position of its lowest-score neighbors — pure spatial gradient following
5. Curvity derived from `e·p` determines how strongly the particle steers: aligned → low curvity, anti-aligned → high curvity
6. Combined with FAABP dynamics (forces, curvity, self-propulsion), this produces collective payload pushing toward goal
