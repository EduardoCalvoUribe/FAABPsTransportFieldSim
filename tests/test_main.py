import pytest
import numpy as np
from src.simulation import simulate_single_step


class TestIntegration:
    """Integration tests for complete simulation step."""

    def test_simulate_single_step_basic(self):
        """Test a complete simulation step with basic setup."""
        n_particles = 3
        box_size = 100.0

        positions = np.array([
            [25.0, 25.0],
            [30.0, 25.0],
            [75.0, 75.0]
        ])
        orientations = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        velocities = np.zeros((n_particles, 2))
        payload_pos = np.array([50.0, 50.0])
        payload_vel = np.zeros(2)
        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 3.0
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.1
        polarity = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        particle_scores = np.array([9999, 9999, 9999])
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.01
        rot_diffusion = np.zeros(n_particles)
        step = 1
        goal_position = np.array([10.0, 10.0])
        particle_view_range = 20.0
        score_and_polarity_update_interval = 10
        walls = np.zeros((0, 4))
        directedness = 1.0

        new_positions, new_orientations, new_velocities, new_payload_pos, new_payload_vel, curvity = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity, particle_scores,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            step, goal_position, particle_view_range, score_and_polarity_update_interval,
            walls, directedness
        )

        # Check that arrays have correct shapes
        assert new_positions.shape == (n_particles, 2)
        assert new_orientations.shape == (n_particles, 2)
        assert new_velocities.shape == (n_particles, 2)
        assert new_payload_pos.shape == (2,)
        assert new_payload_vel.shape == (2,)
        assert curvity.shape == (n_particles,)

        # Check that positions are within box (periodic boundaries)
        assert np.all(new_positions >= 0)
        assert np.all(new_positions < box_size)
        assert np.all(new_payload_pos >= 0)
        assert np.all(new_payload_pos < box_size)

        # Check that orientations are normalized
        for i in range(n_particles):
            norm = np.linalg.norm(new_orientations[i])
            assert abs(norm - 1.0) < 1e-6

    def test_simulate_single_step_with_walls(self):
        """Test simulation step with walls present."""
        np.random.seed(42)
        n_particles = 5
        box_size = 50.0

        positions = np.random.uniform(10, 40, (n_particles, 2))
        angles = np.random.uniform(0, 2*np.pi, n_particles)
        orientations = np.column_stack([np.cos(angles), np.sin(angles)])
        velocities = np.zeros((n_particles, 2))
        payload_pos = np.array([25.0, 25.0])
        payload_vel = np.zeros(2)
        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 3.0
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.1
        polarity = orientations.copy()
        particle_scores = np.full(n_particles, 9999)
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.01
        rot_diffusion = np.ones(n_particles) * 0.05
        step = 1
        goal_position = np.array([5.0, 5.0])
        particle_view_range = 15.0
        score_and_polarity_update_interval = 5
        walls = np.array([
            [0, 0, 0, box_size],
            [0, 0, box_size, 0],
            [box_size, box_size, 0, box_size],
            [box_size, box_size, box_size, 0]
        ])
        directedness = 0.5

        new_positions, new_orientations, new_velocities, new_payload_pos, new_payload_vel, curvity = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity, particle_scores,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            step, goal_position, particle_view_range, score_and_polarity_update_interval,
            walls, directedness
        )

        # Verify output shapes
        assert new_positions.shape == (n_particles, 2)
        assert new_orientations.shape == (n_particles, 2)
        assert new_velocities.shape == (n_particles, 2)
        assert curvity.shape == (n_particles,)

        # Verify normalization
        for i in range(n_particles):
            assert abs(np.linalg.norm(new_orientations[i]) - 1.0) < 1e-6

    def test_simulate_single_step_polarity_update(self):
        """Test that polarity updates occur at correct intervals."""
        np.random.seed(123)
        n_particles = 4
        box_size = 100.0

        positions = np.array([
            [20.0, 20.0],
            [25.0, 20.0],
            [20.0, 25.0],
            [25.0, 25.0]
        ])
        orientations = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0]
        ])
        velocities = np.zeros((n_particles, 2))
        payload_pos = np.array([50.0, 50.0])
        payload_vel = np.zeros(2)
        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 3.0
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.1
        polarity = orientations.copy()
        particle_scores = np.full(n_particles, 9999)
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.01
        rot_diffusion = np.zeros(n_particles)
        goal_position = np.array([10.0, 10.0])
        particle_view_range = 20.0
        score_and_polarity_update_interval = 5
        walls = np.zeros((0, 4))
        directedness = 1.0

        # Run at step that should NOT update (step 1, interval 5)
        _, _, _, _, _, _ = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity, particle_scores,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            1, goal_position, particle_view_range, score_and_polarity_update_interval,
            walls, directedness
        )

        scores_before_update = particle_scores.copy()

        # Run at step that SHOULD update (step 5, interval 5)
        _, _, _, _, _, _ = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity, particle_scores,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            5, goal_position, particle_view_range, score_and_polarity_update_interval,
            walls, directedness
        )

        # Scores should have been updated (particles close to goal should have lower scores)
        # At least one particle should have a different score
        assert not np.array_equal(particle_scores, scores_before_update)

    def test_simulate_single_step_periodic_boundaries(self):
        """Test that periodic boundaries work correctly."""
        n_particles = 1
        box_size = 100.0

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
        polarity = np.array([[1.0, 0.0]])
        particle_scores = np.array([9999])
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.5  # Large time step to move far
        rot_diffusion = np.zeros(n_particles)
        step = 1
        goal_position = np.array([50.0, 50.0])
        particle_view_range = 20.0
        score_and_polarity_update_interval = 10
        walls = np.zeros((0, 4))
        directedness = 1.0

        new_positions, _, _, _, _, _ = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel,
            radii, v0s, mobilities, payload_mobility, polarity, particle_scores,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles,
            step, goal_position, particle_view_range, score_and_polarity_update_interval,
            walls, directedness
        )

        # Particle should wrap around to other side
        assert 0 <= new_positions[0, 0] < box_size
        assert 0 <= new_positions[0, 1] < box_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
