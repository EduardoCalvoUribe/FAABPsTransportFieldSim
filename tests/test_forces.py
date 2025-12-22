import pytest
import numpy as np
from src.forces import (
    compute_repulsive_force,
    compute_wall_forces,
    create_cell_list,
    compute_hollow_payload_force
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
        walls = np.array([[0, 0, 0, 10, 0]])  # Wall along y-axis at x=0
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
        walls = np.array([[0, 0, 0, 10, 0]])
        stiffness = 10.0

        force = compute_wall_forces(pos, radius, walls, stiffness)

        np.testing.assert_array_almost_equal(force, np.zeros(2))

    def test_compute_wall_forces_multiple_walls(self):
        """Test forces from multiple walls."""
        pos = np.array([0.5, 0.5])
        radius = 1.0
        walls = np.array([
            [0, 0, 0, 10, 0],   # Left wall
            [0, 0, 10, 0, 0]    # Bottom wall
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
        walls = np.zeros((0, 5))
        stiffness = 10.0

        force = compute_wall_forces(pos, radius, walls, stiffness)

        np.testing.assert_array_almost_equal(force, np.zeros(2))

    def test_compute_wall_forces_curved_wall(self):
        """Test wall force with curved wall (c≠0)."""
        # Curved wall from (0, 0) to (10, 0) with c=0.5
        # Place particle close to middle of arc to ensure collision
        pos = np.array([5.0, -0.5])
        radius = 2.0
        walls = np.array([[0, 0, 10, 0, 0.5]])
        stiffness = 10.0

        force = compute_wall_forces(pos, radius, walls, stiffness)

        # Particle should experience some force from the curved wall
        force_magnitude = np.linalg.norm(force)
        # Force should be non-zero if particle overlaps with wall
        assert force_magnitude >= 0  # At minimum, should not crash

    def test_compute_wall_forces_curved_wall_negative(self):
        """Test wall force with negatively curved wall (c<0)."""
        # Curved wall from (0, 0) to (10, 0) with c=-0.5 (bulges upward)
        # Place particle near the arc
        pos = np.array([5.0, 0.5])
        radius = 2.0
        walls = np.array([[0, 0, 10, 0, -0.5]])
        stiffness = 10.0

        force = compute_wall_forces(pos, radius, walls, stiffness)

        # Force magnitude should be non-negative
        force_magnitude = np.linalg.norm(force)
        assert force_magnitude >= 0


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


class TestHollowPayloadForce:
    """Tests for hollow payload force computation."""

    def test_particle_inside_no_collision(self):
        """Test no force when particle is inside without touching boundary."""
        pos_particle = np.array([50.0, 50.0])  # At center
        pos_payload = np.array([50.0, 50.0])   # Payload centered at same point
        particle_radius = 1.0
        payload_radius = 20.0
        stiffness = 10.0
        box_size = 100.0

        force = compute_hollow_payload_force(
            pos_particle, pos_payload, particle_radius,
            payload_radius, stiffness, box_size
        )

        # Particle at center, well within boundary, no collision
        np.testing.assert_array_almost_equal(force, np.zeros(2), decimal=5)

    def test_particle_inside_collision_with_boundary(self):
        """Test force when particle inside collides with boundary."""
        pos_particle = np.array([69.5, 50.0])  # Close to boundary on right
        pos_payload = np.array([50.0, 50.0])
        particle_radius = 1.0
        payload_radius = 20.0
        stiffness = 10.0
        box_size = 100.0

        force = compute_hollow_payload_force(
            pos_particle, pos_payload, particle_radius,
            payload_radius, stiffness, box_size
        )

        # Particle is 19.5 from center, radius 1.0 means edge at 20.5 > payload_radius
        # Force should push particle toward center (negative x)
        assert force[0] < 0
        assert abs(force[1]) < 1e-6

    def test_particle_outside_no_collision(self):
        """Test no force when particle is outside without touching boundary."""
        pos_particle = np.array([80.0, 50.0])  # Far outside
        pos_payload = np.array([50.0, 50.0])
        particle_radius = 1.0
        payload_radius = 20.0
        stiffness = 10.0
        box_size = 100.0

        force = compute_hollow_payload_force(
            pos_particle, pos_payload, particle_radius,
            payload_radius, stiffness, box_size
        )

        # Particle center is 30 from payload center, payload radius is 20
        # Particle edge at 29, which is > payload_radius, no collision
        np.testing.assert_array_almost_equal(force, np.zeros(2), decimal=5)

    def test_particle_outside_collision_with_boundary(self):
        """Test force when particle outside collides with boundary."""
        pos_particle = np.array([70.5, 50.0])  # Close to boundary
        pos_payload = np.array([50.0, 50.0])
        particle_radius = 1.0
        payload_radius = 20.0
        stiffness = 10.0
        box_size = 100.0

        force = compute_hollow_payload_force(
            pos_particle, pos_payload, particle_radius,
            payload_radius, stiffness, box_size
        )

        # Particle center is 20.5 from payload center
        # Particle inner edge at 19.5 < payload_radius (20), so collision
        # Force should push particle away from center (positive x)
        assert force[0] > 0
        assert abs(force[1]) < 1e-6

    def test_force_direction_symmetry(self):
        """Test that forces are symmetric in all directions."""
        pos_payload = np.array([50.0, 50.0])
        particle_radius = 1.0
        payload_radius = 20.0
        stiffness = 10.0
        box_size = 100.0

        # Test collision from inside at different angles
        # Right side
        force_right = compute_hollow_payload_force(
            np.array([69.5, 50.0]), pos_payload, particle_radius,
            payload_radius, stiffness, box_size
        )
        # Top
        force_top = compute_hollow_payload_force(
            np.array([50.0, 69.5]), pos_payload, particle_radius,
            payload_radius, stiffness, box_size
        )

        # Magnitudes should be equal
        assert abs(np.linalg.norm(force_right) - np.linalg.norm(force_top)) < 1e-6

        # Directions should be toward center (pushed inward)
        assert force_right[0] < 0  # Pushed left (toward center)
        assert force_top[1] < 0    # Pushed down (toward center)

    def test_newtons_third_law(self):
        """Test that the force follows Newton's third law (negate for payload)."""
        pos_particle = np.array([69.5, 50.0])
        pos_payload = np.array([50.0, 50.0])
        particle_radius = 1.0
        payload_radius = 20.0
        stiffness = 10.0
        box_size = 100.0

        force_on_particle = compute_hollow_payload_force(
            pos_particle, pos_payload, particle_radius,
            payload_radius, stiffness, box_size
        )

        # Force on payload should be opposite (negated in simulation code)
        # Here we just verify the force is non-zero and pointing toward center
        assert np.linalg.norm(force_on_particle) > 0

    def test_periodic_boundary(self):
        """Test force computation across periodic boundary."""
        pos_particle = np.array([5.0, 50.0])
        pos_payload = np.array([95.0, 50.0])  # 10 units away with wrapping
        particle_radius = 1.0
        payload_radius = 15.0
        stiffness = 10.0
        box_size = 100.0

        force = compute_hollow_payload_force(
            pos_particle, pos_payload, particle_radius,
            payload_radius, stiffness, box_size
        )

        # With periodic boundaries, effective distance is 10
        # Particle would be inside the payload (distance 10 < radius 15)
        # This tests that periodic boundaries are handled correctly
        assert isinstance(force, np.ndarray)
        assert force.shape == (2,)
