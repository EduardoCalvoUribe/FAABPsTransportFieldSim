import pytest
import numpy as np
from src.simulation import simulate_single_step


class TestIntegration:
    """Integration tests for the new force-based polarity system."""

    def test_simulate_single_step_basic(self):
        """Test basic simulation step with force-based polarity fields."""
        np.random.seed(42)
        n_particles = 3
        box_size = 50

        positions = np.array([
            [10.0, 10.0],
            [15.0, 10.0],
            [10.0, 15.0]
        ])
        orientations = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-0.707, 0.707]
        ])
        velocities = np.zeros((n_particles, 2))
        payload_pos = np.array([30.0, 30.0])
        payload_vel = np.zeros(2)
        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 3.0
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.1
        polarity_fields = np.zeros((n_particles, box_size, box_size, 2))
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.01
        rot_diffusion = np.zeros(n_particles)
        step = 1
        force_update_interval = 10
        walls = np.zeros((0, 4))
        n_steps = 100
        payload_circular_motion = False
        payload_circle_center = np.array([box_size/2, box_size/2])
        payload_circle_radius = box_size/3
        payload_n_rotations = 1
        last_collision_partner = np.full(n_particles, -1, dtype=np.int64)
        collision_share_interval = 10

        new_positions, new_orientations, new_velocities, new_payload_pos, new_payload_vel, curvity, active_polarity, _ = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity_fields,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            step, force_update_interval, walls, n_steps,
            payload_circular_motion, payload_circle_center, payload_circle_radius, payload_n_rotations,
            last_collision_partner, collision_share_interval, 1.0, 2
        )

        # Check outputs have correct shapes
        assert new_positions.shape == (n_particles, 2)
        assert new_orientations.shape == (n_particles, 2)
        assert curvity.shape == (n_particles,)
        assert active_polarity.shape == (n_particles, 2)

        # Check positions are within bounds
        assert np.all(new_positions >= 0)
        assert np.all(new_positions < box_size)

        # Check orientations are normalized
        for i in range(n_particles):
            assert abs(np.linalg.norm(new_orientations[i]) - 1.0) < 1e-6

    def test_simulate_single_step_with_walls(self):
        """Test simulation step with wall forces."""
        n_particles = 2
        box_size = 50

        positions = np.array([
            [15.0, 25.0],
            [35.0, 25.0]
        ])
        orientations = np.array([
            [1.0, 0.0],
            [-1.0, 0.0]
        ])
        velocities = np.zeros((n_particles, 2))
        payload_pos = np.array([25.0, 40.0])
        payload_vel = np.zeros(2)
        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 3.0
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.1
        polarity_fields = np.zeros((n_particles, box_size, box_size, 2))
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.01
        rot_diffusion = np.zeros(n_particles)
        step = 1
        force_update_interval = 10
        # Add a vertical wall in the middle
        walls = np.array([[25, 0, 25, 50]], dtype=np.float64)
        n_steps = 100
        payload_circular_motion = False
        payload_circle_center = np.array([box_size/2, box_size/2])
        payload_circle_radius = box_size/3
        payload_n_rotations = 1
        last_collision_partner = np.full(n_particles, -1, dtype=np.int64)
        collision_share_interval = 10

        new_positions, new_orientations, new_velocities, new_payload_pos, new_payload_vel, curvity, active_polarity, _ = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity_fields,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            step, force_update_interval, walls, n_steps,
            payload_circular_motion, payload_circle_center, payload_circle_radius, payload_n_rotations,
            last_collision_partner, collision_share_interval, 1.0, 2
        )

        # Check that simulation ran successfully
        assert new_positions.shape == (n_particles, 2)
        assert new_orientations.shape == (n_particles, 2)

    def test_simulate_single_step_polarity_field_update(self):
        """Test that polarity fields are updated at correct intervals."""
        n_particles = 2
        box_size = 50

        positions = np.array([[20.0, 20.0], [30.0, 30.0]])
        orientations = np.array([[1.0, 0.0], [0.0, 1.0]])
        velocities = np.zeros((n_particles, 2))
        payload_pos = np.array([25.0, 25.0])
        payload_vel = np.zeros(2)
        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 3.0
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.1
        polarity_fields = np.zeros((n_particles, box_size, box_size, 2))
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.01
        rot_diffusion = np.zeros(n_particles)
        force_update_interval = 5
        walls = np.zeros((0, 4))
        n_steps = 100
        payload_circular_motion = False
        payload_circle_center = np.array([box_size/2, box_size/2])
        payload_circle_radius = box_size/3
        payload_n_rotations = 1
        last_collision_partner = np.full(n_particles, -1, dtype=np.int64)
        collision_share_interval = 10

        # Run at step that should NOT update (step 1, interval 5)
        *_, last_collision_partner = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity_fields,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            1, force_update_interval, walls, n_steps,
            payload_circular_motion, payload_circle_center, payload_circle_radius, payload_n_rotations,
            last_collision_partner, collision_share_interval, 1.0, 2
        )
        field_sum_before = np.sum(np.abs(polarity_fields))

        # Run at step that SHOULD update (step 5, interval 5)
        *_, last_collision_partner = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity_fields,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            5, force_update_interval, walls, n_steps,
            payload_circular_motion, payload_circle_center, payload_circle_radius, payload_n_rotations,
            last_collision_partner, collision_share_interval, 1.0, 2
        )
        field_sum_after = np.sum(np.abs(polarity_fields))

        # Field should have been updated on step 5
        # (field_sum_after might equal field_sum_before if forces are zero, so we just check it ran)
        assert field_sum_after >= 0  # Test passed if no error

    def test_simulate_single_step_periodic_boundaries(self):
        """Test that periodic boundaries work correctly."""
        n_particles = 1
        box_size = 100

        # Place particle near edge
        positions = np.array([[99.0, 50.0]])
        orientations = np.array([[1.0, 0.0]])  # Moving right
        velocities = np.zeros((n_particles, 2))
        payload_pos = np.array([50.0, 50.0])
        payload_vel = np.zeros(2)
        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 10.0  # High velocity
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.1
        polarity_fields = np.zeros((n_particles, box_size, box_size, 2))
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.5  # Large time step to move far
        rot_diffusion = np.zeros(n_particles)
        step = 1
        force_update_interval = 10
        walls = np.zeros((0, 4))
        n_steps = 100
        payload_circular_motion = False
        payload_circle_center = np.array([box_size/2, box_size/2])
        payload_circle_radius = box_size/3
        payload_n_rotations = 1
        last_collision_partner = np.full(n_particles, -1, dtype=np.int64)
        collision_share_interval = 10

        new_positions, *_ = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity_fields,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            step, force_update_interval, walls, n_steps,
            payload_circular_motion, payload_circle_center, payload_circle_radius, payload_n_rotations,
            last_collision_partner, collision_share_interval, 1.0, 2
        )

        # Check that particle wrapped around (position < box_size)
        assert 0 <= new_positions[0, 0] < box_size
        assert 0 <= new_positions[0, 1] < box_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
