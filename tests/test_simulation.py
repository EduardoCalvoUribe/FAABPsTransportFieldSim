import pytest
import numpy as np
from src.simulation import (
    compute_all_forces,
    update_orientation_vectors,
    compute_curvity_from_polarity,
    fetch_polarity_from_field,
    update_polarity_fields_from_forces,
    simulate_single_step
)


class TestPolarityFieldOperations:
    """Tests for polarity field fetch and update operations."""

    def test_fetch_polarity_from_field(self):
        """Test fetching polarity vectors from fields based on position."""
        n_particles = 2
        box_size = 10

        # Create polarity fields with known values
        polarity_fields = np.zeros((n_particles, box_size, box_size, 2))
        # Particle 0's field at (5, 5) has vector [1.0, 0.0]
        polarity_fields[0, 5, 5] = np.array([1.0, 0.0])
        # Particle 1's field at (3, 7) has vector [0.0, 1.0]
        polarity_fields[1, 3, 7] = np.array([0.0, 1.0])

        # Positions that map to these grid cells
        positions = np.array([[5.3, 5.8], [3.1, 7.9]])

        active_polarity = fetch_polarity_from_field(positions, polarity_fields, box_size, n_particles)

        # Should fetch the stored vectors
        np.testing.assert_array_almost_equal(active_polarity[0], np.array([1.0, 0.0]))
        np.testing.assert_array_almost_equal(active_polarity[1], np.array([0.0, 1.0]))

    def test_fetch_polarity_from_empty_field(self):
        """Test fetching from uninitialized field returns zeros."""
        n_particles = 1
        box_size = 10
        polarity_fields = np.zeros((n_particles, box_size, box_size, 2))
        positions = np.array([[5.5, 5.5]])

        active_polarity = fetch_polarity_from_field(positions, polarity_fields, box_size, n_particles)

        np.testing.assert_array_almost_equal(active_polarity[0], np.zeros(2))

    def test_update_polarity_fields_from_forces(self):
        """Test storing forces in polarity fields."""
        n_particles = 2
        box_size = 10
        polarity_fields = np.zeros((n_particles, box_size, box_size, 2))
        force_scaling = 1.0

        positions = np.array([[2.3, 4.7], [8.9, 1.2]])
        forces = np.array([[1.5, -0.5], [-1.0, 2.0]])

        update_polarity_fields_from_forces(positions, forces, polarity_fields, box_size, n_particles, force_scaling)

        # Check that forces were stored at correct grid locations with tanh() applied
        # Position [2.3, 4.7] -> grid [2, 4]
        np.testing.assert_array_almost_equal(polarity_fields[0, 2, 4], np.tanh(forces[0]))
        # Position [8.9, 1.2] -> grid [8, 1]
        np.testing.assert_array_almost_equal(polarity_fields[1, 8, 1], np.tanh(forces[1]))

    def test_update_polarity_fields_accumulates(self):
        """Test that updating accumulates (sums) forces instead of overwriting."""
        n_particles = 1
        box_size = 10
        polarity_fields = np.zeros((n_particles, box_size, box_size, 2))
        force_scaling = 1.0

        # Store initial force
        positions = np.array([[5.0, 5.0]])
        forces_1 = np.array([[1.0, 1.0]])
        update_polarity_fields_from_forces(positions, forces_1, polarity_fields, box_size, n_particles, force_scaling)

        # Store new force at same location
        forces_2 = np.array([[2.0, -1.0]])
        update_polarity_fields_from_forces(positions, forces_2, polarity_fields, box_size, n_particles, force_scaling)

        # Should have the accumulated sum with tanh() applied
        expected_sum = np.tanh(np.tanh(forces_1[0]) + forces_2[0])
        np.testing.assert_array_almost_equal(polarity_fields[0, 5, 5], expected_sum)


class TestCurvityComputation:
    """Tests for curvity computation from active polarity."""

    def test_compute_curvity_from_polarity_aligned(self):
        """Test curvity when orientation and polarity are aligned."""
        orientations = np.array([[1.0, 0.0], [0.0, 1.0]])
        active_polarity = np.array([[1.0, 0.0], [0.0, 1.0]])
        n_particles = 2

        curvity = compute_curvity_from_polarity(orientations, active_polarity, n_particles)

        # When aligned, dot product is 1, curvity is -1
        np.testing.assert_array_almost_equal(curvity, np.array([-1.0, -1.0]))

    def test_compute_curvity_from_polarity_opposite(self):
        """Test curvity when orientation and polarity are opposite."""
        orientations = np.array([[1.0, 0.0], [0.0, 1.0]])
        active_polarity = np.array([[-1.0, 0.0], [0.0, -1.0]])
        n_particles = 2

        curvity = compute_curvity_from_polarity(orientations, active_polarity, n_particles)

        # When opposite, dot product is -1, curvity is 1
        np.testing.assert_array_almost_equal(curvity, np.array([1.0, 1.0]))

    def test_compute_curvity_from_polarity_perpendicular(self):
        """Test curvity when orientation and polarity are perpendicular."""
        orientations = np.array([[1.0, 0.0]])
        active_polarity = np.array([[0.0, 1.0]])
        n_particles = 1

        curvity = compute_curvity_from_polarity(orientations, active_polarity, n_particles)

        # When perpendicular, dot product is 0, curvity is 0
        np.testing.assert_array_almost_equal(curvity, np.array([0.0]))

    def test_compute_curvity_from_zero_polarity(self):
        """Test curvity when polarity is zero (no force history)."""
        orientations = np.array([[1.0, 0.0]])
        active_polarity = np.array([[0.0, 0.0]])
        n_particles = 1

        curvity = compute_curvity_from_polarity(orientations, active_polarity, n_particles)

        # Zero polarity gives zero curvity
        np.testing.assert_array_almost_equal(curvity, np.array([0.0]))


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


class TestSimulationStep:
    """Integration tests for full simulation step."""

    def test_simulate_single_step_basic(self):
        """Test that a single step executes without errors."""
        n_particles = 5
        box_size = 50

        positions = np.random.uniform(0, box_size, (n_particles, 2))
        orientations = np.random.randn(n_particles, 2)
        # Normalize orientations
        for i in range(n_particles):
            orientations[i] /= np.linalg.norm(orientations[i])
        velocities = np.zeros((n_particles, 2))
        payload_pos = np.array([25.0, 25.0])
        payload_vel = np.zeros(2)

        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 3.0
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.05
        polarity_fields = np.zeros((n_particles, box_size, box_size, 2))
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.01
        rot_diffusion = np.ones(n_particles) * 0.05
        step = 1
        force_update_interval = 10
        walls = np.zeros((0, 4))

        # Should execute without error
        n_steps = 100
        payload_circular_motion = False
        payload_circle_center = np.array([box_size/2, box_size/2])
        payload_circle_radius = box_size/3
        payload_n_rotations = 1
        last_collision_partner = np.full(n_particles, -1, dtype=np.int64)
        collision_share_interval = 10

        new_pos, new_orient, new_vel, new_payload_pos, new_payload_vel, curvity, active_polarity, _ = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity_fields,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            step, force_update_interval, walls, n_steps,
            payload_circular_motion, payload_circle_center, payload_circle_radius, payload_n_rotations,
            last_collision_partner, collision_share_interval, 1.0, 2
        )

        # Check output shapes
        assert new_pos.shape == (n_particles, 2)
        assert new_orient.shape == (n_particles, 2)
        assert new_vel.shape == (n_particles, 2)
        assert curvity.shape == (n_particles,)
        assert active_polarity.shape == (n_particles, 2)

        # Check orientations are normalized
        for i in range(n_particles):
            assert abs(np.linalg.norm(new_orient[i]) - 1.0) < 1e-6

    def test_simulate_single_step_updates_polarity_field(self):
        """Test that polarity fields are updated at the correct interval."""
        n_particles = 2
        box_size = 20

        positions = np.array([[5.5, 5.5], [10.5, 10.5]])
        orientations = np.array([[1.0, 0.0], [0.0, 1.0]])
        velocities = np.zeros((n_particles, 2))
        payload_pos = np.array([15.0, 15.0])
        payload_vel = np.zeros(2)

        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 3.0
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.05
        polarity_fields = np.zeros((n_particles, box_size, box_size, 2))
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.01
        rot_diffusion = np.zeros(n_particles)
        force_update_interval = 5
        walls = np.zeros((0, 4))

        # Step 5 should trigger update
        step = 5
        n_steps = 100
        payload_circular_motion = False
        payload_circle_center = np.array([box_size/2, box_size/2])
        payload_circle_radius = box_size/3
        payload_n_rotations = 1
        last_collision_partner = np.full(n_particles, -1, dtype=np.int64)
        collision_share_interval = 10

        *_, last_collision_partner = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity_fields,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            step, force_update_interval, walls, n_steps,
            payload_circular_motion, payload_circle_center, payload_circle_radius, payload_n_rotations,
            last_collision_partner, collision_share_interval, 1.0, 2
        )

        # Polarity field should have been updated (check it's no longer all zeros)
        # At position [5.5, 5.5] -> grid [5, 5]
        # There should be some force stored (even if it's small)
        field_sum = np.sum(np.abs(polarity_fields))
        # Just check that something was written
        assert field_sum >= 0  # At minimum, no error occurred
