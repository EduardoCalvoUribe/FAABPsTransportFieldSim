# Goal-Seeking Algorithm with Vicsek-Style Alignment

## Overview
Each particle has two key properties:
- **Score (s)**: An integer representing the particle's "distance" from the goal in discrete hops
- **Vector (v)**: A unit vector that influences the particle's curvity (and therefore turning behavior)

## Algorithm Structure

### 1. Particle Properties
Every particle `i` has:
- **s_i** ∈ ℤ: Score (initialized to 9999)
- **v_i** ∈ ℝ²: Unit vector (||v_i|| = 1)
- **e_i** ∈ ℝ²: Heading/orientation vector (||e_i|| = 1)
- **r**: View range (hyperparameter, default: 0.1 × box_size)

### 2. Core Update Rules (Executed Every `goal_update_interval` Steps)

For each particle `i` at position **x_i**:

#### **Case A: Line of Sight to Goal**
If ||**x_goal** - **x_i**|| ≤ r (goal within view range):

```
s_i = 0
v_i = (**x_goal** - **x_i**) / ||**x_goal** - **x_i**||
```

**Meaning**: Particle has direct access to goal; points directly toward it with score 0.

---

#### **Case B: No Line of Sight to Goal**
If ||**x_goal** - **x_i**|| > r (goal out of range):

##### **Step 1: Find Neighbors Within Range**
Define neighbor set:
```
N_i = {j ≠ i : ||**x_j** - **x_i**|| ≤ r}
```
**Note**: Uses direct Euclidean distance (no periodic boundary conditions).

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

##### **Step 3: Vector Alignment (Vicsek-Style)**

**v_i** aligns with neighbors' vectors using score-weighted averaging:

```
v_i = (Σ_j w_j · v_j) / ||Σ_j w_j · v_j||

where j ∈ N_i and weights:

w_j = exp(-(s_j - s*))
```

**Weight Properties**:
- Particles with minimum score s* get weight w = exp(0) = 1.0
- Higher scores decay exponentially: w decreases as s_j increases
- Relative weighting: Independent of absolute score values
  - If s* = 1: s=1 gets w=1.0, s=2 gets w≈0.368, s=3 gets w≈0.135
  - If s* = 500: s=500 gets w=1.0, s=501 gets w≈0.368, s=502 gets w≈0.135
- **Result**: Minimum-score neighbors always dominate alignment

**Normalization**:
```
weighted_v = (Σ_j w_j · v_j) / (Σ_j w_j)    [weighted average]
v_i = weighted_v / ||weighted_v||            [normalize to unit vector]
```

---

### 3. Curvity Calculation

Once **v_i** is computed, it determines particle curvity:

```
κ_i = -(e_i · v_i)
```

Where:
- **e_i**: Current heading/orientation (unit vector)
- **v_i**: Computed goal-seeking vector (unit vector)
- κ_i ∈ [-1, 1]: Curvity parameter

**Interpretation**:
- κ_i = -1: v and e perfectly aligned (no turning needed)
- κ_i = 0: v perpendicular to e
- κ_i = +1: v opposite to e (maximum turning)

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
- **No periodic wrapping** (bounds checking skips out-of-range cells)

### Update Frequency
- Parameter: `goal_update_interval` (default: 10 timesteps)
- Vectors and scores update every `goal_update_interval` steps
- Reduces computational cost while maintaining gradient propagation

### Initialization
- All particles start with s_i = 9999, v_i = (cos(π/4), sin(π/4))
- Goal position: default (4/5 × box_size, 4/5 × box_size)
- Score propagates outward from goal over time

---

## Key Properties

1. **Gradient Formation**: Scores form discrete gradient field pointing toward goal
2. **Information Propagation**: Score=0 spreads from goal at ~1 cell per update
3. **Alignment Consensus**: Particles align with lowest-score neighbors (follow the gradient)
4. **Scale Invariance**: Exponential weighting ensures relative influence independent of absolute scores
5. **Isolation Handling**: Isolated particles reset to s=9999, preventing stale information

---

## Comparison to Standard Vicsek Model

| Aspect | Standard Vicsek | This Algorithm |
|--------|-----------------|----------------|
| Alignment | Equal weights for all neighbors | Score-weighted (lower = stronger) |
| Goal-seeking | None | Direct (if visible) or gradient-based |
| Score/Distance | N/A | Discrete gradient field |
| Update rule | v̄ = average(neighbors' velocities) | v̄ = weighted_avg(neighbors' v, weights=exp(-Δs)) |

---

## Physical Interpretation

This algorithm creates **emergent cooperative transport**:
1. Particles near goal "see" it and orient toward it (s=0)
2. Their neighbors align with them and get s=1
3. Information cascades outward, forming gradient
4. Particles far from goal follow the gradient by aligning with lower-score neighbors
5. Combined with FAABP dynamics (forces, curvity, self-propulsion), this creates collective payload pushing toward goal

The exponential weighting ensures particles strongly prefer following the most direct path (lowest scores), creating efficient cooperative navigation.
