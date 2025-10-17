# Wall Implementation Plan - Detailed Design Document

## Overview
Implement static walls in the FAABP simulation that:
1. Block particle and payload movement (collision response)
2. Block line-of-sight for goal detection
3. Prevent particle v-vector alignment across walls
4. Create a maze-like path: start (bottom-left) → right → left → goal (top-right)

---

## 1. Wall Data Structure

### 1.1 Wall Representation
- **Format**: NumPy array of shape `(n_walls, 4)` containing `[x1, y1, x2, y2]` for each wall segment
- **Type**: `np.float64` for numba compatibility
- **Storage**: Add to `params['walls']` dictionary

### 1.2 Example Configuration
For a maze forcing right → left → goal path:
```python
box_size = 350
walls = np.array([
    # Wall 1: Vertical barrier on right (blocks direct diagonal path)
    [box_size*0.6, box_size*0.2, box_size*0.6, box_size*0.7],  # x1, y1, x2, y2

    # Wall 2: Vertical barrier on left (forces turn to goal)
    [box_size*0.3, box_size*0.4, box_size*0.3, box_size*0.9]
])
```

**Path created**:
- Payload starts at `(87.5, 87.5)` (box_size/4)
- Wall 1 at x=210 blocks direct path to goal at `(280, 280)` (4*box_size/5)
- Forces movement right (x > 210) to get around Wall 1
- Wall 2 at x=105 then forces movement left (x < 210) to reach goal
- Creates S-shaped or zigzag path

### 1.3 Alternative: Empty Walls
```python
walls = np.zeros((0, 4), dtype=np.float64)  # Empty array for no walls
```

---

## 2. Geometry Utility Functions

Add to the "Physics utility functions" section (after `normalize()`, line 26):

### 2.1 Line-Segment Intersection Test
```python
@njit(fastmath=True)
def line_segments_intersect(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y):
    """Check if line segment (p1, p2) intersects with line segment (p3, p4).

    Uses cross-product method:
    - Calculate direction vectors: d1 = p2-p1, d2 = p4-p3
    - Cross product: cross = d1 × d2
    - If parallel (cross ≈ 0): return False
    - Calculate intersection parameters t1, t2
    - Intersection exists if both 0 ≤ t1 ≤ 1 and 0 ≤ t2 ≤ 1

    Returns:
        bool: True if segments intersect, False otherwise
    """
```

**Implementation details**:
- Fast rejection for parallel lines (cross product ≈ 0)
- Parameter `t1` represents position along segment 1
- Parameter `t2` represents position along segment 2
- Intersection point = `p1 + t1 * (p2 - p1)` = `p3 + t2 * (p4 - p3)`

### 2.2 Point-to-Segment Distance
```python
@njit(fastmath=True)
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Calculate minimum distance from point (px, py) to line segment (x1,y1)-(x2,y2).

    Algorithm:
    1. Project point onto infinite line containing segment
    2. Clamp projection parameter t to [0, 1] to stay on segment
    3. Calculate distance from point to closest point on segment

    Returns:
        distance: float, minimum distance
        closest_x: float, x-coordinate of closest point on segment
        closest_y: float, y-coordinate of closest point on segment
    """
```

**Implementation details**:
- Use dot product to find projection: `t = dot(point-p1, p2-p1) / ||p2-p1||²`
- Clamp `t` to `[0, 1]` to handle endpoints
- Handle degenerate case where `x1=x2` and `y1=y2`

### 2.3 Check Line-Wall Intersection
```python
@njit(fastmath=True)
def line_intersects_any_wall(p1_x, p1_y, p2_x, p2_y, walls):
    """Check if line segment (p1, p2) intersects any wall.

    Args:
        p1_x, p1_y: Start point coordinates
        p2_x, p2_y: End point coordinates
        walls: np.ndarray of shape (n_walls, 4) with [x1, y1, x2, y2] per wall

    Returns:
        bool: True if line intersects any wall, False otherwise
    """
    n_walls = walls.shape[0]
    for i in range(n_walls):
        if line_segments_intersect(p1_x, p1_y, p2_x, p2_y,
                                   walls[i, 0], walls[i, 1],
                                   walls[i, 2], walls[i, 3]):
            return True
    return False
```

---

## 3. Wall Collision Forces

Add after `compute_repulsive_force()` (around line 72):

### 3.1 Wall Collision Force Function
```python
@njit(fastmath=True)
def compute_wall_forces(pos, radius, walls, stiffness):
    """Compute repulsive forces from all walls on a particle/payload.

    For each wall:
    1. Calculate distance from particle center to wall segment
    2. If distance < radius: particle is colliding with wall
    3. Apply force: F = stiffness * (radius - distance) * normal_direction
       - Force magnitude increases as penetration increases
       - Direction is perpendicular to wall, pushing away from it

    Args:
        pos: np.ndarray [x, y], particle/payload position
        radius: float, particle/payload radius
        walls: np.ndarray of shape (n_walls, 4)
        stiffness: float, wall stiffness (same as particle stiffness)

    Returns:
        force: np.ndarray [fx, fy], total force from all walls
    """
```

**Algorithm**:
```
force = [0, 0]
for each wall in walls:
    distance, closest_x, closest_y = point_to_segment_distance(pos, wall)

    if distance < radius:
        # Collision detected
        overlap = radius - distance

        # Normal vector: from closest point on wall toward particle center
        if distance > 1e-10:
            normal_x = (pos[0] - closest_x) / distance
            normal_y = (pos[1] - closest_y) / distance
        else:
            # Particle exactly on wall - use wall perpendicular
            wall_vec = [wall[2] - wall[0], wall[3] - wall[1]]
            normal = perpendicular(wall_vec)  # Rotate 90°

        # Repulsive force
        force_magnitude = stiffness * overlap
        force += force_magnitude * normal

return force
```

**Key points**:
- Use same stiffness as particle collisions (`25.0`)
- Force direction is always away from wall (outward normal)
- Multiple wall collisions accumulate (sum of forces)
- Handle edge case: particle center exactly on wall line

---

## 4. Integration into Force Computation

Modify `compute_all_forces()` (lines 96-143):

### 4.1 Function Signature Update
```python
@njit(fastmath=True)
def compute_all_forces(positions, payload_pos, radii, payload_radius, stiffness,
                      n_particles, box_size, walls):  # <-- Add walls parameter
```

### 4.2 Wall Forces for Particles
Add after line 115 (after particle-payload forces):
```python
    # Compute forces between particles and walls (O(N * n_walls))
    for i in range(n_particles):
        wall_force = compute_wall_forces(positions[i], radii[i], walls, stiffness)
        particle_forces[i] += wall_force
```

### 4.3 Wall Forces for Payload
Add after line 115 (or near payload force initialization):
```python
    # Compute force between payload and walls
    payload_wall_force = compute_wall_forces(payload_pos, payload_radius, walls, stiffness)
    payload_force += payload_wall_force
```

### 4.4 Update All Callers
Modify `simulate_single_step()` line 403:
```python
    particle_forces, payload_force = compute_all_forces(
        positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls  # <-- Add
    )
```

---

## 5. Line-of-Sight Blocking by Walls

### 5.1 Extend `has_line_of_sight()` Function

**Current implementation** (lines 220-268): Only checks if payload blocks view.

**New implementation**: Add wall checks.

Modify function signature (line 220):
```python
@njit(fastmath=True)
def has_line_of_sight(pos_i, goal_position, payload_pos, payload_radius, box_size, walls):  # <-- Add walls
```

Add before return statement (around line 241, after bounding box check):
```python
    # Check if any wall blocks line of sight
    if line_intersects_any_wall(x_i, y_i, x_goal, y_goal, walls):
        return False  # Wall blocks line of sight

    # [existing payload intersection code continues...]
```

**Logic flow**:
1. Quick bounding box check (existing)
2. **NEW**: Check wall intersections (fast reject if blocked)
3. Check payload circle intersection (existing)

### 5.2 Update Caller in `point_vector_to_goal()`

Modify line 304-308 region:
```python
    # If goal is within range, check line of sight
    if dist_to_goal <= r:
        # NEW: Check if line of sight is clear (no walls or payload blocking)
        if has_line_of_sight(pos_i, goal_position, payload_pos, payload_radius, box_size, walls):
            if dist_to_goal > 0:
                return np.array([dx_goal / dist_to_goal, dy_goal / dist_to_goal]), 0
            return np.array([0.0, 0.0]), 0
        # If blocked, fall through to gradient-following behavior
```

**Current behavior**: Directly points to goal if within range `r`.

**New behavior**: Only points to goal if:
- Within range `r` AND
- Clear line of sight (no walls or payload blocking)

If blocked by wall, particle falls back to gradient-following (aligning with neighbors).

---

## 6. Neighbor Filtering for Alignment

### 6.1 Check if Walls Separate Two Particles

Add new utility function (near other wall utilities):
```python
@njit(fastmath=True)
def particles_separated_by_wall(pos_i, pos_j, walls):
    """Check if a wall blocks the line segment between two particles.

    Args:
        pos_i: np.ndarray [x, y], position of particle i
        pos_j: np.ndarray [x, y], position of particle j
        walls: np.ndarray of shape (n_walls, 4)

    Returns:
        bool: True if any wall separates the particles, False otherwise
    """
    return line_intersects_any_wall(pos_i[0], pos_i[1], pos_j[0], pos_j[1], walls)
```

### 6.2 Filter Neighbors in `point_vector_to_goal()`

Modify the neighbor collection loop (lines 339-353):

**Before** (current):
```python
            while j != -1:
                if i != j:
                    # Compute direct distance (NON PERIODIC)
                    r_ij = positions[j] - pos_i
                    dist_j = np.sqrt(np.sum(r_ij**2))

                    if dist_j <= r:
                        neighbor_indices.append(j)
                        neighbor_scores.append(particle_scores[j])

                j = list_next[j]
```

**After** (with wall filtering):
```python
            while j != -1:
                if i != j:
                    # Compute direct distance (NON PERIODIC)
                    r_ij = positions[j] - pos_i
                    dist_j = np.sqrt(np.sum(r_ij**2))

                    # NEW: Only include neighbor if within range AND not separated by wall
                    if dist_j <= r and not particles_separated_by_wall(pos_i, positions[j], walls):
                        neighbor_indices.append(j)
                        neighbor_scores.append(particle_scores[j])

                j = list_next[j]
```

**Effect**: Particles on opposite sides of a wall will not align their v-vectors with each other, even if they are within range `r`.

### 6.3 Update Function Signature

Modify `point_vector_to_goal()` signature (line 272):
```python
@njit(fastmath=True)
def point_vector_to_goal(pos_i, goal_position, positions, particle_scores, i, n_particles,
                        r, box_size, current_score, head, list_next, n_cells,
                        all_particle_vectors, walls):  # <-- Add walls parameter
```

### 6.4 Update Caller in `simulate_single_step()`

Modify line 414:
```python
            particle_vectors[i], particle_scores[i] = point_vector_to_goal(
                positions[i], goal_position, positions, particle_scores, i, n_particles,
                particle_view_range, box_size, particle_scores[i], head_goal, list_next_goal, n_cells_goal,
                particle_vectors, walls  # <-- Add
            )
```

---

## 7. Parameter Updates

### 7.1 Modify `default_payload_params()` (line 789)

Add `walls` parameter:
```python
def default_payload_params(n_particles=1000, curvity=0, payload_radius=20,
                          goal_position=None, use_goal=False,
                          particle_view_range=None, goal_update_interval=10,
                          walls=None):  # <-- Add walls parameter
```

Add to return dictionary (around line 813):
```python
    # Default: no walls
    if walls is None:
        walls = np.zeros((0, 4), dtype=np.float64)

    return {
        # ... existing parameters ...
        'walls': walls,  # <-- Add to return dict
    }
```

### 7.2 Update `simulate_single_step()` Signature (line 397)

Add `walls` parameter:
```python
def simulate_single_step(positions, orientations, velocities, payload_pos, payload_vel,
                        radii, v0s, mobilities, payload_mobility, particle_vectors, particle_scores,
                        stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
                        step, goal_position, use_goal, particle_view_range, goal_update_interval,
                        walls):  # <-- Add walls parameter
```

### 7.3 Update Caller in `run_payload_simulation()` (line 529)

Add `walls` to extraction and passing:
```python
    # Extract parameters (around line 476)
    walls = params['walls']

    # ...

    # Call simulate_single_step (line 529)
    positions, orientations, velocities, payload_pos, payload_vel, curvity = simulate_single_step(
        positions, orientations, velocities, payload_pos, payload_vel,
        params['particle_radius'], params['v0'], params['mobility'], params['payload_mobility'],
        particle_vectors, particle_scores, params['stiffness'],
        params['box_size'], params['payload_radius'], params['dt'], params['rot_diffusion'],
        n_particles, step, goal_position, use_goal, particle_view_range, goal_update_interval,
        walls  # <-- Add
    )
```

---

## 8. Visualization

### 8.1 Update `create_payload_animation()` (line 589)

Add walls to function signature:
```python
def create_payload_animation(positions, orientations, velocities, payload_positions, params,
                            curvity_values, output_file='visualizations/payload_animation_00.mp4',
                            show_vectors=False, particle_vectors=None):
```

Extract walls from params (around line 608):
```python
    walls = params.get('walls', np.zeros((0, 4)))
```

### 8.2 Add Wall Rendering

Add after creating the goal visualization (around line 666, after goal marker):
```python
    # Draw walls
    wall_lines = []
    for i in range(walls.shape[0]):
        line, = ax.plot(
            [walls[i, 0], walls[i, 2]],  # x-coordinates: [x1, x2]
            [walls[i, 1], walls[i, 3]],  # y-coordinates: [y1, y2]
            color='black',
            linewidth=4,
            solid_capstyle='round',
            zorder=10  # Draw on top of particles
        )
        wall_lines.append(line)
```

### 8.3 Update `init()` and `update()` Functions

In `init()` (line 706):
```python
    def init():
        """Initialize the animation."""
        artists = [scatter, payload, trajectory, time_text, params_text, params_text_2]
        if goal is not None:
            artists.append(goal)
        if quiver is not None:
            artists.append(quiver)
        # Add wall lines (they don't change, but include for completeness)
        artists.extend(wall_lines)
        return artists
```

No changes needed to `update()` since walls are static.

---

## 9. Main Execution Test

### 9.1 Create Wall Configuration

Modify `__main__` section (around line 883):

**Before**:
```python
    params = default_payload_params(n_particles=1000, use_goal=True)
```

**After**:
```python
    # Define maze walls
    box_size = 350
    walls = np.array([
        # Right barrier: vertical wall at x=210, blocking direct path
        [box_size*0.6, box_size*0.2, box_size*0.6, box_size*0.7],

        # Left barrier: vertical wall at x=105, forcing final turn
        [box_size*0.3, box_size*0.4, box_size*0.3, box_size*0.9]
    ], dtype=np.float64)

    params = default_payload_params(n_particles=1000, use_goal=True, walls=walls)
```

### 9.2 Verify Path

Expected behavior:
1. **Start**: Payload at `(87.5, 87.5)` (bottom-left quarter)
2. **Wall 1 blocks**: Direct path to goal at `(280, 280)` blocked by wall at x=210
3. **Move right**: Particles push payload to x > 210 to clear Wall 1
4. **Wall 2 blocks**: Wall at x=105 blocks leftward movement below y=140
5. **Move up and left**: Particles navigate around Wall 2 to reach goal
6. **Goal reached**: Payload arrives at `(280, 280)` (top-right)

---

## 10. Implementation Checklist

### Phase 1: Geometry Utilities
- [ ] Add `line_segments_intersect()` function
- [ ] Add `point_to_segment_distance()` function
- [ ] Add `line_intersects_any_wall()` function
- [ ] Add `particles_separated_by_wall()` function

### Phase 2: Collision Forces
- [ ] Add `compute_wall_forces()` function
- [ ] Update `compute_all_forces()` signature to include walls
- [ ] Add particle-wall force computation in `compute_all_forces()`
- [ ] Add payload-wall force computation in `compute_all_forces()`

### Phase 3: Line-of-Sight
- [ ] Update `has_line_of_sight()` signature to include walls
- [ ] Add wall intersection check in `has_line_of_sight()`
- [ ] Update `has_line_of_sight()` calls in `point_vector_to_goal()`

### Phase 4: Neighbor Filtering
- [ ] Update `point_vector_to_goal()` signature to include walls
- [ ] Add wall separation check in neighbor collection loop
- [ ] Update `point_vector_to_goal()` call in `simulate_single_step()`

### Phase 5: Parameter Propagation
- [ ] Update `default_payload_params()` to accept and return walls
- [ ] Update `simulate_single_step()` signature to include walls
- [ ] Update `simulate_single_step()` call in `run_payload_simulation()`
- [ ] Extract walls from params in `run_payload_simulation()`

### Phase 6: Visualization
- [ ] Extract walls from params in `create_payload_animation()`
- [ ] Add wall rendering code in `create_payload_animation()`
- [ ] Update `init()` to include wall artists
- [ ] Update animation call to pass walls via params

### Phase 7: Testing
- [ ] Create wall configuration in `__main__`
- [ ] Run compilation test with walls
- [ ] Run full simulation with walls
- [ ] Verify maze navigation behavior
- [ ] Check animation displays walls correctly

---

## 11. Performance Considerations

### 11.1 Computational Complexity
- **Wall collision forces**: O(N × W) where N = particles, W = walls
  - For 1000 particles and 2 walls: 2000 distance calculations per step
  - Negligible compared to O(N) particle-particle forces

- **Line-of-sight checks**: O(N × W) during goal updates
  - Only executed every `goal_update_interval` steps (default: 10)
  - For 1000 particles, 2 walls: 2000 intersection tests per 10 steps

- **Neighbor filtering**: O(N × neighbors × W)
  - Average neighbors ≈ 10-20 per particle
  - For 1000 particles, 15 avg neighbors, 2 walls: ~30k tests per goal update
  - Still much faster than O(N²) brute force

### 11.2 Optimization Notes
- All geometry functions are `@njit` compiled (zero Python overhead)
- Early rejection in line intersection (parallel check)
- Wall forces only computed when distance < radius
- No memory allocations in hot loops

### 11.3 Numba Compatibility
- Use `np.ndarray` for walls (not Python lists)
- All functions decorated with `@njit`
- No dynamic typing or Python objects in jitted code
- Use `dtype=np.float64` explicitly for walls array

---

## 12. Edge Cases & Error Handling

### 12.1 Empty Walls
- `walls = np.zeros((0, 4))` → loops over 0 walls, no forces/checks
- Simulation behaves identically to no-wall case
- No performance penalty

### 12.2 Degenerate Walls
- Zero-length walls (x1=x2, y1=y2) handled in `point_to_segment_distance()`
- Returns distance to point instead of segment

### 12.3 Particle Inside Wall
- If particle spawns inside wall or clips through
- `compute_wall_forces()` applies strong outward force
- Particle should be pushed out within a few timesteps

### 12.4 Particle Exactly on Wall
- Distance = 0 case handled with normal vector calculation
- Falls back to wall perpendicular direction

### 12.5 Periodic Boundaries
- Current implementation uses modulo boundaries (line 453-454)
- Walls use absolute coordinates (no wrapping)
- Walls near boundaries (0 or box_size) may have edge behavior
- **Recommendation**: Keep walls away from boundaries (e.g., 0.1*box_size < walls < 0.9*box_size)

---

## 13. Testing Strategy

### 13.1 Unit Tests (Manual Verification)
1. **Geometry**: Test `line_segments_intersect()` with known cases
   - Parallel lines → False
   - Perpendicular intersecting lines → True
   - Non-intersecting segments → False

2. **Wall forces**: Place particle against wall, check force direction
   - Force should point away from wall
   - Magnitude should increase with penetration depth

3. **Line-of-sight**: Place particle, goal, and wall in line
   - Should return False if wall intersects
   - Should return True if wall is parallel but not intersecting

### 13.2 Integration Tests
1. **Single wall barrier**:
   - Place vertical wall between start and goal
   - Verify particles navigate around it

2. **Maze navigation**:
   - Use two-wall configuration (right → left path)
   - Verify payload follows zigzag path
   - Check that particles don't align through walls

3. **Collision response**:
   - Verify particles bounce off walls (no clipping)
   - Verify payload bounces off walls
   - Check for stable contact (no jittering)

### 13.3 Visual Verification
- Walls should appear as black lines
- Payload trajectory should avoid walls
- Particle distribution should show "shadow" behind walls
- v-vector arrows should not point through walls

---

## 14. Future Extensions (Out of Scope)

### Not Implemented in This Plan
1. **Moving walls**: All walls are static
2. **Curved walls**: Only straight line segments supported
3. **Wall friction**: Walls are frictionless (only normal force)
4. **Wall thickness**: Walls are infinitesimally thin line segments
5. **Partial transparency**: Walls completely block line-of-sight (no partial visibility)
6. **Wall-payload attraction**: Walls are purely repulsive
7. **Dynamic wall generation**: Walls defined at initialization only

### Possible Future Work
- Add wall-tangential friction force
- Support circular arc walls
- Implement wall thickness with two parallel segments
- Add wall "porosity" for partial line-of-sight blocking
- Generate maze walls procedurally

---

## 15. Summary of All File Modifications

| Location | Type | Description |
|----------|------|-------------|
| Line 26 (after `normalize()`) | Add | 4 geometry utility functions |
| Line 72 (after `compute_repulsive_force()`) | Add | `compute_wall_forces()` function |
| Line 97 (`compute_all_forces()`) | Modify | Add walls parameter, compute wall forces |
| Line 220 (`has_line_of_sight()`) | Modify | Add walls parameter, check wall intersections |
| Line 272 (`point_vector_to_goal()`) | Modify | Add walls parameter, filter neighbors by walls |
| Line 397 (`simulate_single_step()`) | Modify | Add walls parameter, pass to subfunctions |
| Line 462 (`run_payload_simulation()`) | Modify | Extract walls from params, pass to `simulate_single_step()` |
| Line 589 (`create_payload_animation()`) | Modify | Extract walls, render wall lines |
| Line 789 (`default_payload_params()`) | Modify | Add walls parameter, return walls in dict |
| Line 883 (`__main__`) | Modify | Define wall array, pass to `default_payload_params()` |

**Total estimated additions**: ~200 lines of code
**Total estimated modifications**: ~15 function signatures + ~10 function calls

---

## 16. Verification Checklist

After implementation, verify:
- [ ] Simulation compiles without numba errors
- [ ] Particles collide with walls (visible in animation)
- [ ] Payload collides with walls (visible in animation)
- [ ] Walls render correctly (black lines in animation)
- [ ] Particles navigate around walls to reach goal
- [ ] Particle v-vectors don't align through walls (check with `show_vectors=True`)
- [ ] No particles clip through walls during simulation
- [ ] Performance is acceptable (similar to no-wall case)
- [ ] Score propagation works correctly around walls
- [ ] Goal is reached despite wall obstacles

---

## End of Plan Document
