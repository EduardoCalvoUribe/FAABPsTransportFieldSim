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
- **r**: View range (hyperparameter, default: 0.1 × box_size)

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

##### **Step 3: Polarity Alignment (Hybrid: Score-Weighted + Gradient)**

**p_i** is computed as a weighted combination of two components:

```
p_i = ((1-d) · p_weighted + d · p_gradient) / ||(1-d) · p_weighted + d · p_gradient||

where:
- d ∈ [0, 1]: directedness hyperparameter (default: 0.5)
- p_weighted: score-weighted alignment with neighbors
- p_gradient: direction toward lowest-score neighbor(s)
```

**Component 1 - Score-Weighted Alignment** (weight: 1-d):
```
p_weighted = (Σ_j w_j · p_j) / ||Σ_j w_j · p_j||    where j ∈ N_i

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
p_gradient = (**x*** - **x_i**) / ||**x*** - **x_i**||

where **x*** = (1/|J|) · Σ_j **x_j**    for j ∈ J
      J = {j ∈ N_i : s_j = s*}
```
- Points directly toward the average position of minimum-score neighbor(s)
- Uses periodic boundary conditions to compute relative positions
- Pure gradient descent behavior (most direct path)

**Directedness Parameter** (d):
- d = 0: Pure score-weighted Vicsek alignment (smooth, consensus-based)
- d = 0.5: Balanced hybrid (default)
- d = 1: Pure gradient descent (direct pursuit)
- Higher d = more direct/aggressive goal-seeking
- Lower d = more collective/smooth navigation

**Normalization**:
```
p_i = combined / ||combined||    [normalize to unit vector]
```

---

### 3. Curvity Calculation

Once **p_i** is computed, it determines particle curvity:

```
κ_i = -(e_i · p_i)
```

Where:
- **e_i**: Current heading/orientation (unit vector)
- **p_i**: Computed polarity vector (unit vector)
- κ_i ∈ [-1, 1]: Curvity parameter

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

1. **Gradient Formation**: Scores form discrete gradient field pointing toward goal
2. **Information Propagation**: Score=0 spreads from goal at ~1 cell per update
3. **Alignment Consensus**: Particles align with lowest-score neighbors (follow the gradient)
4. **Scale Invariance**: Exponential weighting ensures relative influence independent of absolute scores
5. **Isolation Handling**: Isolated particles reset to s=9999, preventing stale information

---

## Physical Interpretation

This algorithm creates **emergent cooperative transport**:
1. Particles with clear line of sight to goal (within range, unobstructed by walls/payload) orient toward it (s=0)
2. Their neighbors align with them and get s=1
3. Information cascades outward, forming gradient
4. Particles far from goal follow the gradient through hybrid alignment:
   - **Score-weighted component (1-d)**: Aligns with neighbors' polarity vectors, but strongly favors lower-score neighbors via exponential weighting. Maintains collective behavior while following the gradient.
   - **Gradient component (d)**: Direct spatial pursuit of lowest-score neighbor position (using periodic boundaries). Most direct path but ignores polarity field structure.
5. Combined with FAABP dynamics (forces, curvity, self-propulsion), this creates collective payload pushing toward goal

The **directedness parameter** allows tuning between:
- **Low d (→ 0)**: Pure score-weighted Vicsek alignment. Smooth, collective navigation that follows the polarity field gradient. More robust to noise and obstacles.
- **High d (→ 1)**: Pure spatial gradient descent. Direct, aggressive pursuit of best neighbor. Faster but may create sharp turns or get stuck.
- **Balanced d=0.5**: Compromise between smooth polarity-field following and direct spatial pursuit.
