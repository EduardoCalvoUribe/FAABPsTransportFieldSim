# Force-Aligning Active Brownian Particles (FAABPs) with Polarity Fields

## Overview

This simulation implements Force-Aligning Active Brownian Particles (FAABPs) that learn to cooperatively transport a payload through spatial memory stored in polarity fields. Each particle maintains a personal spatial memory (polarity field) that encodes historical force experiences, enabling collective path learning and emergent cooperative behavior.

## Core Concepts

### Particle Properties
Each particle `i` has the following properties:
- **Position** `x_i ∈ ℝ²`: Spatial location
- **Orientation** `e_i ∈ ℝ²`: Heading direction (unit vector, ||e_i|| = 1)
- **Velocity** `v_i ∈ ℝ²`: Current velocity
- **Active polarity** `p_i ∈ ℝ²`: Polarity vector fetched from field at current position
- **Curvity** `κ_i ∈ [-1, 1]`: Force-alignment parameter computed from polarity

### Polarity Fields
Each particle maintains a **personal polarity field**:
- **Structure**: `F_i[x, y] ∈ ℝ²` for each grid cell (x, y)
- **Dimensions**: `box_size × box_size × 2` (vector at each integer grid position)
- **Purpose**: Spatial memory encoding historical force experiences
- **Learning**: Updated through two mechanisms:
  1. **Force accumulation**: Direct experience of forces at visited locations
  2. **Collision sharing**: Knowledge transfer when particles collide

## Algorithm Structure

### Phase 1: Training Phase (Circular Payload Motion)

During the first `n_training_steps`, the payload follows a prescribed circular path while particles interact with it and learn force patterns.

#### Payload Motion
```
Angular velocity: ω = (2π × n_rotations) / n_training_steps
Angle at step t: θ(t) = ω × t

Position: payload_pos(t) = center + radius × [cos(θ), sin(θ)]
Velocity: payload_vel(t) = radius × ω × [-sin(θ), cos(θ)]
```

#### Particle Learning
Particles accumulate force experiences in their polarity fields through two mechanisms:

**1. Force Accumulation** (every `force_update_interval` steps):
```
For each particle i at position (x, y):
  grid_x = int(x) mod box_size
  grid_y = int(y) mod box_size

  F_i[grid_x, grid_y] = tanh(F_i[grid_x, grid_y] + α × F_external_i)
```
- `α`: Force scaling factor (default: 0.1)
- `F_external_i`: Net force on particle from payload and other particles
- `tanh()`: Prevents unbounded growth, keeps field vectors bounded

**2. Collision-Based Knowledge Sharing** (every `collision_share_interval` steps):
```
When particles i and j collide (new collision, not continuous contact):
  For all grid cells (x, y):
    F_sum = F_i[x, y] + F_j[x, y]
    F_i[x, y] = tanh(F_sum)
    F_j[x, y] = tanh(F_sum)
```
- Particles exchange complete spatial knowledge upon collision
- Only new collisions trigger sharing (tracked via `last_collision_partner`)
- Both particles receive the same updated field

### Phase 2: Test Phase (Passive Payload)

After training, the payload becomes passive and responds only to forces from particles. Learning can be controlled via `test_phase_learning`:
- `0`: No learning (frozen fields)
- `1`: Collision communication only
- `2`: Both force accumulation and collision sharing (default)

#### Payload Dynamics
```
Force: F_payload = Σ(particle-payload forces) + Σ(wall forces)
Velocity: v_payload = μ_payload × F_payload
Position: payload_pos += v_payload × dt
```
- `μ_payload = 1 / payload_radius`: Payload mobility

## Physics Equations

### 1. Force Computation

**Repulsive Forces** (soft-sphere model):
```
F_ij = { S₀ × (r_i + r_j - ||x_j - x_i||) × r̂_ij,  if ||x_j - x_i|| < r_i + r_j
       { 0,                                          otherwise

where:
  r̂_ij = (x_j - x_i) / ||x_j - x_i||  (unit vector from i to j)
  S₀: stiffness parameter
```

Applied between:
- Particle-particle pairs (via efficient cell list, O(N))
- Particle-payload pairs
- Particles/payload and walls

**Wall Forces**:
```
For each wall segment:
  d = point_to_segment_distance(particle_pos, wall_segment)
  if d < r_particle:
    overlap = r_particle - d
    F_wall = S₀ × overlap × n̂
```
- `n̂`: Normal vector from wall toward particle

**Periodic Boundaries**: Distances computed using minimum image convention, but wall intersection checks use the actual periodic shortest path to prevent forces through walls.

### 2. Curvity and Polarity

**Active Polarity Fetch**:
```
At each timestep, particle i at position (x, y):
  grid_x = int(x) mod box_size
  grid_y = int(y) mod box_size

  p_i = F_i[grid_x, grid_y]  (fetch from personal field)
```

**Curvity Calculation**:
```
κ_i = -(e_i · p_i)

where:
  e_i: current orientation (heading)
  p_i: active polarity (fetched from field)
```

**Physical Meaning**:
- `κ > 0`: Polarity opposes heading → particle curves to align with polarity
- `κ < 0`: Polarity aligns with heading → particle continues straight
- `κ = 0`: Polarity perpendicular to heading → no alignment torque

### 3. Orientation Dynamics

**Torque from Force Alignment**:
```
τ_i = κ_i × (e_i × F_i)
    = κ_i × (e_i,x × F_i,y - e_i,y × F_i,x)
```

**Orientation Update**:
```
Change: de_i/dt = τ_i × (e_i × ẑ) + ξ_i

where:
  e_i × ẑ = [-e_i,y, e_i,x]  (perpendicular to orientation)
  ξ_i: rotational diffusion noise
```

**Rotational Noise**:
```
ξ_i ~ N(0, √(2 D_rot dt))  projected perpendicular to e_i
```

**Final Update**:
```
e_i(t + dt) = normalize(e_i(t) + de_i)
```

### 4. Position Dynamics

**Velocity Composition**:
```
v_i = v₀_i × e_i + μ_i × F_i

where:
  v₀_i × e_i: self-propulsion
  μ_i × F_i: force-induced motion
```

**Position Update**:
```
x_i(t + dt) = x_i(t) + v_i × dt  (mod box_size)
```

## Implementation Details

### Efficient Neighbor Search
Uses **cell list algorithm** for O(N) complexity:
```
Cell size: max(2 × max_particle_radius, ...)
Grid: n_cells × n_cells  where n_cells = floor(box_size / cell_size)

For each particle:
  1. Assign to cell based on position
  2. Check 3×3 neighborhood of cells (including own)
  3. Use linked list to iterate particles in each cell
```

**Periodic Boundaries**: Cell wrapping handles periodic boundary conditions automatically.

**Wall Filtering**: Particle pairs separated by walls along the periodic shortest path are excluded from force calculations.

### Wall Handling

**Line Segment Intersection**:
```
Check if segment (p1, p2) intersects wall (w1, w2):
  - Use cross-product method
  - Return True if segments intersect

For periodic boundaries:
  - Compute shortest periodic path
  - Check wall intersection along this path
```

**Collision Detection**:
```
For particle at position p with radius r:
  For each wall segment:
    d = distance from p to segment
    if d < r:
      Apply repulsive force
```

### Numerical Integration

**Time Integration**:
- **Method**: Forward Euler
- **Time step**: `dt = 0.01` (typical)
- **Update order**:
  1. Compute forces
  2. Fetch polarity from fields
  3. Compute curvity
  4. Update polarity fields (training phase)
  5. Share fields on collision (training phase)
  6. Update orientations
  7. Update positions
  8. Apply periodic boundaries

### JIT Compilation

All performance-critical functions use Numba's `@njit(fastmath=True)`:
- `compute_all_forces()`
- `simulate_single_step()`
- `update_orientation_vectors()`
- `update_polarity_fields_from_forces()`
- `share_polarity_fields_on_collision()`
- All physics utility functions

## Key Parameters

### Particle Parameters
- `N_PARTICLES`: Number of particles (default: 1200)
- `PARTICLE_RADIUS`: Particle size (default: 1.0)
- `PARTICLE_V0`: Self-propulsion speed (default: 3.75)
- `PARTICLE_MOBILITY`: Response to forces (default: 1.0)
- `ROTATIONAL_DIFFUSION`: Orientational noise (default: 0.05)

### Payload Parameters
- `PAYLOAD_RADIUS`: Payload size (default: 20)
- `PAYLOAD_MOBILITY`: `1 / PAYLOAD_RADIUS` (size-dependent)
- `PAYLOAD_CIRCLE_RADIUS`: Training path radius (default: BOX_SIZE/3)
- `PAYLOAD_N_ROTATIONS`: Number of training rotations (default: 6)

### Force Parameters
- `STIFFNESS`: Repulsive force strength (default: 25.0)
- `POLARITY_FORCE_SCALING`: Learning rate for force accumulation (default: 0.1)

### Learning Parameters
- `FORCE_UPDATE_INTERVAL`: How often to update fields with forces (default: 20 steps)
- `COLLISION_SHARE_INTERVAL`: How often to share on collision (default: 10 steps)
- `TEST_PHASE_LEARNING`: Learning control in test phase (default: 0, no learning)

### Simulation Parameters
- `BOX_SIZE`: Domain size (default: 300)
- `N_TRAINING_STEPS`: Training duration (default: 20000)
- `N_TEST_STEPS`: Test duration (default: 60000)
- `DT`: Time step (default: 0.01)

## Emergent Behavior

### Collective Transport Mechanism

1. **Training Phase Learning**:
   - Payload follows circular path
   - Particles pushed by payload experience forces
   - Forces accumulate in spatial fields at visited locations
   - Collision sharing spreads knowledge across swarm
   - Each particle builds personal map of "good directions" at each location

2. **Test Phase Exploitation**:
   - Payload becomes passive
   - Particles fetch polarity vectors from learned fields
   - Curvity computed from polarity → particles align with learned directions
   - Particles collectively push payload along learned path
   - Emergent cooperative transport without explicit goal-seeking

3. **Key Mechanisms**:
   - **Spatial memory**: Polarity fields encode location-dependent behavior
   - **Distributed learning**: Each particle has independent memory, shared via collisions
   - **Force alignment**: Curvity mechanism couples motion to learned polarity
   - **Collective action**: Multiple particles push payload coherently

### Advantages of Polarity Field Approach

1. **Scalability**: Each particle operates independently, no global coordination
2. **Robustness**: Distributed memory survives individual particle failure
3. **Adaptability**: Fields can adapt if environment changes (with learning enabled)
4. **Simplicity**: No explicit path planning or communication protocol
5. **Biological plausibility**: Similar to stigmergy in social insects

## Data Storage and Visualization

### Saved Data
- Particle positions, orientations, velocities
- Payload positions and velocities
- Curvity values over time
- Active polarity vectors (fetched from fields)

### Visualization
- Particles colored by curvity (`κ > 0` = red/orange, `κ < 0` = blue/cyan)
- Optional polarity vector arrows
- Payload trajectory
- Wall segments (if present)
- Goal position marker

## References

This simulation is based on the Force-Aligning Active Brownian Particle (FAABP) model, which couples particle orientation to experienced forces through the curvity parameter. The polarity field extension adds spatial memory for learning-based collective transport.