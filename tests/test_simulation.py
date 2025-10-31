import pytest
import numpy as np
from src.simulation import (
    compute_all_forces,
    update_orientation_vectors,
    compute_curvity_from_polarity,
    has_line_of_sight,
    compute_polarity_weighted_vicsek,
    compute_polarity_toward_minscore_pos,
    compute_polarity_toward_minscore_ang,
    point_polarity_to_goal
)


class TestCurvityComputation:
    """Tests for curvity computation from polarity."""

    def test_compute_curvity_from_polarity_aligned(self):
        """Test curvity when orientation and polarity are aligned."""
        orientations = np.array([[1.0, 0.0], [0.0, 1.0]])
        polarity = np.array([[1.0, 0.0], [0.0, 1.0]])
        n_particles = 2

        curvity = compute_curvity_from_polarity(orientations, polarity, n_particles)

        # When aligned, dot product is 1, curvity is -1
        np.testing.assert_array_almost_equal(curvity, np.array([-1.0, -1.0]))

    def test_compute_curvity_from_polarity_opposite(self):
        """Test curvity when orientation and polarity are opposite."""
        orientations = np.array([[1.0, 0.0], [0.0, 1.0]])
        polarity = np.array([[-1.0, 0.0], [0.0, -1.0]])
        n_particles = 2

        curvity = compute_curvity_from_polarity(orientations, polarity, n_particles)

        # When opposite, dot product is -1, curvity is 1
        np.testing.assert_array_almost_equal(curvity, np.array([1.0, 1.0]))

    def test_compute_curvity_from_polarity_perpendicular(self):
        """Test curvity when orientation and polarity are perpendicular."""
        orientations = np.array([[1.0, 0.0]])
        polarity = np.array([[0.0, 1.0]])
        n_particles = 1

        curvity = compute_curvity_from_polarity(orientations, polarity, n_particles)

        # When perpendicular, dot product is 0, curvity is 0
        np.testing.assert_array_almost_equal(curvity, np.array([0.0]))


class TestLineOfSight:
    """Tests for line of sight calculations."""

    def test_has_line_of_sight_clear(self):
        """Test line of sight when clear."""
        pos_i = np.array([0.0, 0.0])
        goal_position = np.array([10.0, 10.0])
        payload_pos = np.array([50.0, 50.0])  # Far away
        payload_radius = 5.0
        walls = None  # No walls

        result = has_line_of_sight(pos_i, goal_position, payload_pos, payload_radius, walls)

        assert result == True

    def test_has_line_of_sight_blocked_by_payload(self):
        """Test line of sight blocked by payload."""
        pos_i = np.array([0.0, 0.0])
        goal_position = np.array([20.0, 20.0])
        payload_pos = np.array([10.0, 10.0])  # Directly in the way
        payload_radius = 5.0
        walls = None

        result = has_line_of_sight(pos_i, goal_position, payload_pos, payload_radius, walls)

        assert result == False

    def test_has_line_of_sight_blocked_by_wall(self):
        """Test line of sight blocked by wall."""
        pos_i = np.array([0.0, 0.0])
        goal_position = np.array([20.0, 0.0])
        payload_pos = np.array([50.0, 50.0])  # Out of the way
        payload_radius = 5.0
        walls = np.array([[10, -5, 10, 5]])  # Wall crosses path

        result = has_line_of_sight(pos_i, goal_position, payload_pos, payload_radius, walls)

        assert result == False

    def test_has_line_of_sight_payload_nearby_not_blocking(self):
        """Test line of sight with payload nearby but not blocking."""
        pos_i = np.array([0.0, 0.0])
        goal_position = np.array([20.0, 0.0])
        payload_pos = np.array([10.0, 10.0])  # Above the line
        payload_radius = 5.0
        walls = None

        result = has_line_of_sight(pos_i, goal_position, payload_pos, payload_radius, walls)

        assert result == True


class TestPolarityComputation:
    """Tests for polarity vector computations."""

    def test_compute_polarity_weighted_vicsek(self):
        """Test score-weighted alignment computation."""
        neighbor_indices = np.array([0, 1, 2])
        neighbor_scores = np.array([5, 5, 10])  # Two with min score, one higher
        all_polarity = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        result = compute_polarity_weighted_vicsek(neighbor_indices, neighbor_scores, all_polarity)

        # Lower scores should dominate
        # Score 5 particles (weight=1) point in [1,0]
        # Score 10 particle (weight=exp(-5)) points in [0,1]
        # Result should be closer to [1,0] than [0,1]
        assert result[0] > result[1]
        # Result should be normalized
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_compute_polarity_weighted_vicsek_all_same_score(self):
        """Test when all neighbors have same score."""
        neighbor_indices = np.array([0, 1])
        neighbor_scores = np.array([5, 5])
        all_polarity = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        result = compute_polarity_weighted_vicsek(neighbor_indices, neighbor_scores, all_polarity)

        # Equal weighting, should average to diagonal
        expected = np.array([1.0, 1.0]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_compute_polarity_toward_minscore_pos(self):
        """Test direction toward minimum score particle."""
        pos_i = np.array([0.0, 0.0])
        neighbor_scores = np.array([10, 5, 5])  # Two particles with min score
        neighbor_positions = np.array([
            [10.0, 0.0],
            [5.0, 0.0],   # Min score
            [5.0, 5.0]    # Min score
        ])
        box_size = 100.0

        result = compute_polarity_toward_minscore_pos(pos_i, neighbor_scores, neighbor_positions, box_size)

        # Should point toward average of [5,0] and [5,5], which is [5, 2.5]
        # Direction from [0,0] to [5, 2.5] normalized
        expected_dir = np.array([5.0, 2.5])
        expected_dir = expected_dir / np.linalg.norm(expected_dir)
        np.testing.assert_array_almost_equal(result, expected_dir)

    def test_compute_polarity_toward_minscore_ang(self):
        """Test average angle toward minimum score particles."""
        pos_i = np.array([0.0, 0.0])
        neighbor_scores = np.array([10, 5, 5])
        neighbor_positions = np.array([
            [10.0, 0.0],
            [5.0, 0.0],   # Min score: direction [1, 0]
            [0.0, 5.0]    # Min score: direction [0, 1]
        ])
        box_size = 100.0

        result = compute_polarity_toward_minscore_ang(pos_i, neighbor_scores, neighbor_positions, box_size)

        # Average of unit vectors [1,0] and [0,1] is [1,1] normalized to [0.707, 0.707]
        expected = np.array([1.0, 1.0]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_compute_polarity_toward_minscore_pos_single_neighbor(self):
        """Test with single minimum score neighbor."""
        pos_i = np.array([0.0, 0.0])
        neighbor_scores = np.array([10, 5, 10])
        neighbor_positions = np.array([
            [10.0, 0.0],
            [5.0, 5.0],   # Min score
            [0.0, 10.0]
        ])
        box_size = 100.0

        result = compute_polarity_toward_minscore_pos(pos_i, neighbor_scores, neighbor_positions, box_size)

        # Should point toward [5, 5]
        expected = np.array([5.0, 5.0]) / np.linalg.norm(np.array([5.0, 5.0]))
        np.testing.assert_array_almost_equal(result, expected)


class TestGoalDirectedPolarity:
    """Tests for goal-directed polarity computation."""

    def test_point_polarity_to_goal_at_goal(self):
        """Test polarity computation when goal is in range and visible."""
        pos_i = np.array([5.0, 5.0])
        goal_position = np.array([10.0, 5.0])
        positions = np.array([[5.0, 5.0]])
        particle_scores = np.array([9999])
        i = 0
        n_particles = 1
        r = 10.0  # Goal is in range
        box_size = 100.0
        current_score = 9999
        payload_pos = np.array([50.0, 50.0])  # Not blocking
        payload_radius = 5.0
        walls = np.zeros((0, 4))
        directedness = 1.0

        # Create simple cell list
        cell_size = r
        head = np.ones((10, 10), dtype=np.int64) * -1
        list_next = np.ones(n_particles, dtype=np.int64) * -1
        n_cells = 10
        all_polarity = np.array([[1.0, 0.0]])

        polarity, score = point_polarity_to_goal(
            pos_i, goal_position, positions, particle_scores, i, n_particles, r, box_size,
            current_score, head, list_next, n_cells, all_polarity, payload_pos, payload_radius,
            walls, directedness
        )

        # Should point directly at goal and have score 0
        assert score == 0
        expected_dir = np.array([1.0, 0.0])  # Goal is to the right
        np.testing.assert_array_almost_equal(polarity, expected_dir)

    def test_point_polarity_to_goal_out_of_range_no_neighbors(self):
        """Test when goal is out of range and no neighbors."""
        pos_i = np.array([5.0, 5.0])
        goal_position = np.array([50.0, 50.0])  # Far away
        positions = np.array([[5.0, 5.0]])
        particle_scores = np.array([9999])
        i = 0
        n_particles = 1
        r = 10.0
        box_size = 100.0
        current_score = 9999
        payload_pos = np.array([0.0, 0.0])
        payload_radius = 5.0
        walls = np.zeros((0, 4))
        directedness = 1.0

        cell_size = r
        head = np.ones((10, 10), dtype=np.int64) * -1
        head[0, 0] = 0
        list_next = np.ones(n_particles, dtype=np.int64) * -1
        n_cells = 10
        all_polarity = np.array([[1.0, 0.0]])

        polarity, score = point_polarity_to_goal(
            pos_i, goal_position, positions, particle_scores, i, n_particles, r, box_size,
            current_score, head, list_next, n_cells, all_polarity, payload_pos, payload_radius,
            walls, directedness
        )

        # No neighbors, should return score 9999
        assert score == 9999
        np.testing.assert_array_almost_equal(polarity, np.zeros(2))


class TestOrientationUpdate:
    """Tests for orientation vector updates."""

    def test_update_orientation_vectors_no_noise(self):
        """Test orientation update without noise."""
        orientations = np.array([[1.0, 0.0]])
        forces = np.array([[0.0, 1.0]])  # Force in y direction
        curvity = np.array([1.0])
        dt = 0.01
        rot_diffusion = np.array([0.0])  # No noise
        n_particles = 1

        new_orientations = update_orientation_vectors(
            orientations, forces, curvity, dt, rot_diffusion, n_particles
        )

        # Should rotate based on torque
        # Torque = curvity * (n × F) = 1.0 * (1*1 - 0*0) = 1.0
        # Rotation should be counterclockwise
        assert new_orientations[0, 1] > 0  # Should have positive y component
        # Should be normalized
        assert abs(np.linalg.norm(new_orientations[0]) - 1.0) < 1e-6

    def test_update_orientation_vectors_no_force(self):
        """Test orientation update with no force."""
        orientations = np.array([[1.0, 0.0]])
        forces = np.array([[0.0, 0.0]])
        curvity = np.array([1.0])
        dt = 0.01
        rot_diffusion = np.array([0.0])
        n_particles = 1

        new_orientations = update_orientation_vectors(
            orientations, forces, curvity, dt, rot_diffusion, n_particles
        )

        # No force, orientation should remain the same
        np.testing.assert_array_almost_equal(new_orientations, orientations)

    def test_update_orientation_vectors_normalization(self):
        """Test that orientations are always normalized."""
        np.random.seed(42)
        orientations = np.array([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]])
        forces = np.random.randn(3, 2)
        curvity = np.random.randn(3)
        dt = 0.01
        rot_diffusion = np.array([0.1, 0.1, 0.1])
        n_particles = 3

        new_orientations = update_orientation_vectors(
            orientations, forces, curvity, dt, rot_diffusion, n_particles
        )

        # All orientations should be normalized
        for i in range(n_particles):
            norm = np.linalg.norm(new_orientations[i])
            assert abs(norm - 1.0) < 1e-6


class TestAllForces:
    """Tests for complete force computation."""

    def test_compute_all_forces_no_overlap(self):
        """Test force computation when no particles overlap."""
        positions = np.array([
            [10.0, 10.0],
            [20.0, 20.0]
        ])
        payload_pos = np.array([50.0, 50.0])
        radii = np.array([1.0, 1.0])
        payload_radius = 5.0
        stiffness = 10.0
        n_particles = 2
        box_size = 100.0
        walls = np.zeros((0, 4))

        particle_forces, payload_force = compute_all_forces(
            positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls
        )

        # No overlap means no forces
        np.testing.assert_array_almost_equal(particle_forces, np.zeros((2, 2)))
        np.testing.assert_array_almost_equal(payload_force, np.zeros(2))

    def test_compute_all_forces_with_overlap(self):
        """Test force computation when particles overlap."""
        positions = np.array([
            [10.0, 10.0],
            [11.5, 10.0]  # Overlapping with first particle
        ])
        payload_pos = np.array([50.0, 50.0])
        radii = np.array([1.0, 1.0])
        payload_radius = 5.0
        stiffness = 10.0
        n_particles = 2
        box_size = 100.0
        walls = np.zeros((0, 4))

        particle_forces, payload_force = compute_all_forces(
            positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls
        )

        # Particles overlap, should have repulsive forces
        # Particle 0 should be pushed left (negative x)
        assert particle_forces[0, 0] < 0
        # Particle 1 should be pushed right (positive x)
        assert particle_forces[1, 0] > 0
        # Forces should be equal and opposite (Newton's third law)
        np.testing.assert_array_almost_equal(particle_forces[0], -particle_forces[1])

    def test_compute_all_forces_payload_overlap(self):
        """Test forces when particle overlaps with payload."""
        positions = np.array([[50.0, 50.0]])
        payload_pos = np.array([52.0, 50.0])  # Overlapping
        radii = np.array([1.5])
        payload_radius = 1.5
        stiffness = 10.0
        n_particles = 1
        box_size = 100.0
        walls = np.zeros((0, 4))

        particle_forces, payload_force = compute_all_forces(
            positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size, walls
        )

        # Particle should be pushed left, payload right
        assert particle_forces[0, 0] < 0
        assert payload_force[0] > 0
        # Forces should be equal and opposite
        np.testing.assert_array_almost_equal(particle_forces[0], -payload_force)
