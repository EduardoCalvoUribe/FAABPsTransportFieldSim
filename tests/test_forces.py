import pytest
import numpy as np
from src.forces import (
    compute_repulsive_force,
    compute_wall_forces,
    create_cell_list
)


class TestRepulsiveForce:
    """Tests for repulsive force between particles."""

    def test_compute_repulsive_force_overlapping(self):
        """Test repulsive force when particles overlap."""
        pos_i = np.array([0.0, 0.0])
        pos_j = np.array([1.5, 0.0])
        radius_i = 1.0
        radius_j = 1.0
        stiffness = 10.0
        box_size = 100.0

        force = compute_repulsive_force(pos_i, pos_j, radius_i, radius_j, stiffness, box_size)

        # Particles overlap by 0.5 (distance 1.5, sum of radii 2.0)
        # Force should push particle i away from j (negative x direction)
        assert force[0] < 0  # Force in negative x direction
        assert abs(force[1]) < 1e-6  # No force in y direction
        expected_magnitude = stiffness * 0.5
        assert abs(np.linalg.norm(force) - expected_magnitude) < 1e-6

    def test_compute_repulsive_force_no_overlap(self):
        """Test no force when particles don't overlap."""
        pos_i = np.array([0.0, 0.0])
        pos_j = np.array([5.0, 0.0])
        radius_i = 1.0
        radius_j = 1.0
        stiffness = 10.0
        box_size = 100.0

        force = compute_repulsive_force(pos_i, pos_j, radius_i, radius_j, stiffness, box_size)

        # No overlap, no force
        np.testing.assert_array_almost_equal(force, np.zeros(2))

    def test_compute_repulsive_force_different_radii(self):
        """Test repulsive force with different particle sizes."""
        pos_i = np.array([0.0, 0.0])
        pos_j = np.array([2.5, 0.0])
        radius_i = 1.0
        radius_j = 2.0
        stiffness = 10.0
        box_size = 100.0

        force = compute_repulsive_force(pos_i, pos_j, radius_i, radius_j, stiffness, box_size)

        # Sum of radii is 3.0, distance is 2.5, overlap is 0.5
        assert force[0] < 0
        expected_magnitude = stiffness * 0.5
        assert abs(np.linalg.norm(force) - expected_magnitude) < 1e-6

    def test_compute_repulsive_force_periodic_wrapping(self):
        """Test repulsive force across periodic boundary."""
        pos_i = np.array([5.0, 5.0])
        pos_j = np.array([95.0, 5.0])
        radius_i = 1.0
        radius_j = 1.0
        stiffness = 10.0
        box_size = 100.0

        force = compute_repulsive_force(pos_i, pos_j, radius_i, radius_j, stiffness, box_size)

        # With periodic boundaries, distance is 10, not 90
        # No overlap (distance 10 > sum of radii 2)
        np.testing.assert_array_almost_equal(force, np.zeros(2))


class TestWallForces:
    """Tests for wall collision forces."""

    def test_compute_wall_forces_collision(self):
        """Test wall force when particle collides with wall."""
        pos = np.array([0.5, 5.0])
        radius = 1.0
        walls = np.array([[0, 0, 0, 10]])  # Wall along y-axis at x=0
        stiffness = 10.0

        force = compute_wall_forces(pos, radius, walls, stiffness)

        # Particle overlaps with wall by 0.5
        # Force should push particle in positive x direction
        assert force[0] > 0
        assert abs(force[1]) < 1e-6
        expected_magnitude = stiffness * 0.5
        assert abs(np.linalg.norm(force) - expected_magnitude) < 1e-6

    def test_compute_wall_forces_no_collision(self):
        """Test no force when particle doesn't collide with wall."""
        pos = np.array([5.0, 5.0])
        radius = 1.0
        walls = np.array([[0, 0, 0, 10]])
        stiffness = 10.0

        force = compute_wall_forces(pos, radius, walls, stiffness)

        np.testing.assert_array_almost_equal(force, np.zeros(2))

    def test_compute_wall_forces_multiple_walls(self):
        """Test forces from multiple walls."""
        pos = np.array([0.5, 0.5])
        radius = 1.0
        walls = np.array([
            [0, 0, 0, 10],   # Left wall
            [0, 0, 10, 0]    # Bottom wall
        ])
        stiffness = 10.0

        force = compute_wall_forces(pos, radius, walls, stiffness)

        # Should be pushed away from both walls (positive x and y)
        assert force[0] > 0
        assert force[1] > 0

    def test_compute_wall_forces_no_walls(self):
        """Test with no walls present."""
        pos = np.array([5.0, 5.0])
        radius = 1.0
        walls = np.zeros((0, 4))
        stiffness = 10.0

        force = compute_wall_forces(pos, radius, walls, stiffness)

        np.testing.assert_array_almost_equal(force, np.zeros(2))


class TestCellList:
    """Tests for cell list neighbor search structure."""

    def test_create_cell_list_basic(self):
        """Test cell list creation for neighbor search."""
        positions = np.array([
            [1.0, 1.0],
            [1.5, 1.5],
            [25.0, 25.0]
        ])
        box_size = 100.0
        cell_size = 10.0
        n_particles = 3

        head, list_next, n_cells = create_cell_list(positions, box_size, cell_size, n_particles)

        assert n_cells == 10  # 100 / 10
        assert head.shape == (10, 10)
        assert list_next.shape == (3,)

        # First two particles should be in same cell (0, 0)
        assert head[0, 0] != -1  # Cell has particles
        # Third particle in cell (2, 2)
        assert head[2, 2] != -1

    def test_create_cell_list_single_particle(self):
        """Test cell list with single particle."""
        positions = np.array([[5.0, 5.0]])
        box_size = 100.0
        cell_size = 10.0
        n_particles = 1

        head, list_next, n_cells = create_cell_list(positions, box_size, cell_size, n_particles)

        assert n_cells == 10
        # Only one cell should have a particle
        assert head[0, 0] == 0
        assert list_next[0] == -1

    def test_create_cell_list_all_in_one_cell(self):
        """Test cell list with all particles in one cell."""
        positions = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0]
        ])
        box_size = 100.0
        cell_size = 10.0
        n_particles = 3

        head, list_next, n_cells = create_cell_list(positions, box_size, cell_size, n_particles)

        # All particles in cell (0, 0)
        assert head[0, 0] != -1
        # Should form a linked list
        particle_count = 0
        current = head[0, 0]
        while current != -1:
            particle_count += 1
            current = list_next[current]
        assert particle_count == 3
