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
        payload_orientation = np.array([1.0, 0.0])
        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 3.0
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.1
        payload_v0 = 1.0
        payload_rot_diffusion = 0.0
        curvity = np.array([1.0, -1.0, 0.0])  # Static curvity values
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.01
        rot_diffusion = np.zeros(n_particles)
        walls = np.zeros((0, 4))

        new_positions, new_orientations, new_velocities, new_payload_pos, new_payload_vel, new_payload_orientation = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel, payload_orientation,
            radii, v0s, mobilities, payload_mobility, payload_v0, payload_rot_diffusion, curvity,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles, walls
        )

        # Check that arrays have correct shapes
        assert new_positions.shape == (n_particles, 2)
        assert new_orientations.shape == (n_particles, 2)
        assert new_velocities.shape == (n_particles, 2)
        assert new_payload_pos.shape == (2,)
        assert new_payload_vel.shape == (2,)
        assert new_payload_orientation.shape == (2,)

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
        payload_orientation = np.array([0.0, 1.0])
        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 3.0
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.1
        payload_v0 = 1.0
        payload_rot_diffusion = 0.0
        curvity = np.array([1.0, -1.0, 0.0, 1.0, -1.0])  # Static curvity values
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.01
        rot_diffusion = np.ones(n_particles) * 0.05
        walls = np.array([
            [10, 10, 10, 40],
            [10, 40, 40, 40],
            [40, 40, 40, 10],
            [40, 10, 10, 10]
        ])

        new_positions, new_orientations, new_velocities, new_payload_pos, new_payload_vel, new_payload_orientation = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel, payload_orientation,
            radii, v0s, mobilities, payload_mobility, payload_v0, payload_rot_diffusion, curvity,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles, walls
        )

        # Verify output shapes
        assert new_positions.shape == (n_particles, 2)
        assert new_orientations.shape == (n_particles, 2)
        assert new_velocities.shape == (n_particles, 2)

        # Verify normalization
        for i in range(n_particles):
            assert abs(np.linalg.norm(new_orientations[i]) - 1.0) < 1e-6

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
        payload_orientation = np.array([1.0, 0.0])
        radii = np.ones(n_particles) * 1.0
        v0s = np.ones(n_particles) * 10.0  # High velocity
        mobilities = np.ones(n_particles) * 1.0
        payload_mobility = 0.1
        payload_v0 = 1.0
        payload_rot_diffusion = 0.0
        curvity = np.array([1.0])
        stiffness = 10.0
        payload_radius = 5.0
        dt = 0.5  # Large time step to move far
        rot_diffusion = np.zeros(n_particles)
        walls = np.zeros((0, 4))

        new_positions, _, _, _, _, _ = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel, payload_orientation,
            radii, v0s, mobilities, payload_mobility, payload_v0, payload_rot_diffusion, curvity,
            stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles, walls
        )

        # Particle should wrap around to other side
        assert 0 <= new_positions[0, 0] < box_size
        assert 0 <= new_positions[0, 1] < box_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
