# Score/Polarity System — Pseudocode

Derived from `src/simulation.py`: `point_polarity_to_goal` and `simulate_single_step`.

---

## When it runs

Every `score_and_polarity_update_interval` timesteps, the score and polarity of all
particles are recomputed in parallel. A frozen snapshot of scores (`old_scores`) is
taken before the update so every particle sees the same previous state.

---

## Helper: `periodic_displacement(pos_i, pos_j)`

Returns the minimum-image displacement vector from `pos_i` to `pos_j`:

```
delta = pos_j - pos_i
delta = delta - box_size * round(delta / box_size)   # wrap each component
return delta
```

---

## Helper: `line_of_sight_to_goal(pos_i, goal)`

Returns `True` if the straight segment `pos_i → goal` is unobstructed.

```
if any wall intersects segment(pos_i, goal):
    return False

# Line-circle intersection test for the payload
a = |goal - pos_i|²
b = -2 * dot(payload_center - pos_i, goal - pos_i)
c = |payload_center - pos_i|² - payload_radius²
discriminant = b² - 4ac

if discriminant < 0:
    return True                      # no intersection

t1, t2 = roots of quadratic
if t1 ∈ [0,1] or t2 ∈ [0,1]:
    return False                     # payload blocks LoS

return True
```

---

## Helper: `wall_blocks(pos_i, endpoint)`

```
return any wall intersects segment(pos_i, endpoint)
```

where `endpoint = pos_i + periodic_displacement(pos_i, pos_j)` (i.e. the neighbor's
position unwrapped along the shortest periodic path).

---

## Main update: `update_scores_and_polarity()`

```
old_scores = particle_scores.copy()
build cell_list with cell_size = particle_view_range   # O(N)

for each particle i  (in parallel):

    r = particle_view_range

    # ── Goal check ────────────────────────────────────────────────────────────
    delta_goal = periodic_displacement(pos_i, goal)

    if |delta_goal| <= r  and  line_of_sight_to_goal(pos_i, goal):
        polarity[i] = normalize(delta_goal)
        score[i]    = 0
        continue

    # ── Pass 1: find minimum score among valid neighbors ───────────────────
    min_score   = +∞
    found_any   = False

    for each j in cell_list neighbors of i  (cells within 1-hop of i's cell):
        if j == i: skip
        delta_ij = periodic_displacement(pos_i, pos_j)
        if |delta_ij| > r: skip
        if wall_blocks(pos_i, pos_i + delta_ij): skip

        found_any = True
        if old_scores[j] < min_score:
            min_score = old_scores[j]

    if not found_any:
        polarity[i] = (0, 0)
        score[i]    = 9999
        continue

    # ── Pass 2: average displacement toward all min-score neighbors ────────
    sum_delta = (0, 0)
    count     = 0

    for each j in cell_list neighbors of i:
        if j == i: skip
        if old_scores[j] != min_score: skip
        delta_ij = periodic_displacement(pos_i, pos_j)
        if |delta_ij| > r: skip
        if wall_blocks(pos_i, pos_i + delta_ij): skip

        sum_delta += delta_ij
        count     += 1

    if count == 0:           # all min-score neighbors were blocked in pass 2
        polarity[i] = (0, 0)
        score[i]    = 9999
        continue

    polarity[i] = normalize(sum_delta / count)
    score[i]    = min_score + 1
```

---

## Score semantics

| Score | Meaning |
|-------|---------|
| 0 | Unobstructed line of sight to goal within `particle_view_range` |
| k | Closest unobstructed neighbor has score k-1; wave-front distance from goal |
| 9999 | No visible, wall-unblocked neighbor found |

Scores propagate outward from goal-visible particles like a discrete BFS wave, but
they are computed from the **previous timestep's scores** so the wave front advances
by one hop per update interval rather than converging instantly.

---

## Polarity semantics

`polarity[i]` is the unit vector pointing from particle `i` toward the **average
position** of all its visible neighbors that share the minimum score. It is the
direction particle `i` "wants" to move to progress toward the goal.

When `score[i] = 0` (direct LoS to goal), polarity points straight at the goal.
When `score[i] = 9999` (isolated), polarity is the zero vector.
