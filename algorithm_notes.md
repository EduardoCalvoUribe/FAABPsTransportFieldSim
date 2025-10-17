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

##### **Step 3: Vector Alignment (Hybrid: Score-Weighted + Gradient)**

**v_i** is computed as a weighted combination of two components:

```
v_i = ((1-d) · v_weighted + d · v_gradient) / ||(1-d) · v_weighted + d · v_gradient||

where:
- d ∈ [0, 1]: directedness hyperparameter (default: 0.5)
- v_weighted: score-weighted alignment with neighbors
- v_gradient: direction toward lowest-score neighbor(s)
```

**Component 1 - Score-Weighted Alignment** (weight: 1-d):
```
v_weighted = (Σ_j w_j · v_j) / ||Σ_j w_j · v_j||    where j ∈ N_i

with weights:
w_j = exp(-(s_j - s*))

where s* = min{s_j : j ∈ N_i}
```
- Exponentially weighted by score difference
- Particles with minimum score s* get weight w = 1.0
- Higher scores decay exponentially (s* + 1 → w ≈ 0.368, s* + 2 → w ≈ 0.135)
- Vicsek-style alignment that favors following lower-score neighbors

**Component 2 - Gradient Following** (weight: d):
```
v_gradient = (**x*** - **x_i**) / ||**x*** - **x_i**||

where **x*** = (1/|J|) · Σ_j **x_j**    for j ∈ J
      J = {j ∈ N_i : s_j = s*}
```
- Points directly toward the position of minimum-score neighbor(s)
- Pure gradient descent behavior (most direct path)

**Directedness Parameter** (d):
- d = 0: Pure score-weighted Vicsek alignment (smooth, consensus-based)
- d = 0.5: Balanced hybrid (default)
- d = 1: Pure gradient descent (direct pursuit)
- Higher d = more direct/aggressive goal-seeking
- Lower d = more collective/smooth navigation

**Normalization**:
```
v_i = combined / ||combined||    [normalize to unit vector]
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
| Alignment | Equal weights for all neighbors | Hybrid: (1-d)×score-weighted + d×gradient |
| Goal-seeking | None | Direct (if visible) or gradient-based |
| Score/Distance | N/A | Discrete gradient field |
| Update rule | v̄ = average(neighbors' velocities) | v̄ = (1-d)×Σ(exp(-Δs)·v_j) + d×toward(min-score) |
| Tunability | Fixed consensus | Adjustable via directedness parameter |

---

## Physical Interpretation

This algorithm creates **emergent cooperative transport**:
1. Particles near goal "see" it and orient toward it (s=0)
2. Their neighbors align with them and get s=1
3. Information cascades outward, forming gradient
4. Particles far from goal follow the gradient through hybrid alignment:
   - **Score-weighted component (1-d)**: Aligns with neighbors' vectors, but strongly favors lower-score neighbors via exponential weighting. Maintains collective behavior while following the gradient.
   - **Gradient component (d)**: Direct spatial pursuit of lowest-score neighbor position. Most direct path but ignores vector field structure.
5. Combined with FAABP dynamics (forces, curvity, self-propulsion), this creates collective payload pushing toward goal

The **directedness parameter** allows tuning between:
- **Low d (→ 0)**: Pure score-weighted Vicsek alignment. Smooth, collective navigation that follows the vector field gradient. More robust to noise and obstacles.
- **High d (→ 1)**: Pure spatial gradient descent. Direct, aggressive pursuit of best neighbor. Faster but may create sharp turns or get stuck.
- **Balanced d=0.5**: Compromise between smooth vector-field following and direct spatial pursuit.
