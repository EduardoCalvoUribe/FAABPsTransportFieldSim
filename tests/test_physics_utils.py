import pytest
import numpy as np
from src.physics_utils import (
    normalize,
    line_segments_intersect,
    point_to_segment_distance,
    line_intersects_any_wall,
    particles_separated_by_wall,
    particles_separated_by_wall_periodic,
    compute_minimum_distance
)


class TestNormalize:
    """Tests for vector normalization."""

    def test_normalize_unit_vector(self):
        """Test normalizing already unit vector."""
        v = np.array([1.0, 0.0])
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_normalize_nonunit_vector(self):
        """Test normalizing a non-unit vector."""
        v = np.array([3.0, 4.0])
        result = normalize(v)
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_zero_vector(self):
        """Test normalizing zero vector returns zero."""
        v = np.array([0.0, 0.0])
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)


class TestLineSegmentIntersection:
    """Tests for line segment intersection detection."""

    def test_line_segments_intersect_crossing(self):
        """Test line segments that cross."""
        # X-shaped intersection
        result = line_segments_intersect(0, 0, 10, 10, 0, 10, 10, 0)
        assert result == True

    def test_line_segments_intersect_no_intersection(self):
        """Test line segments that don't intersect."""
        result = line_segments_intersect(0, 0, 1, 1, 2, 2, 3, 3)
        assert result == False

    def test_line_segments_intersect_parallel(self):
        """Test parallel line segments."""
        result = line_segments_intersect(0, 0, 1, 0, 0, 1, 1, 1)
        assert result == False

    def test_line_segments_touching_at_endpoint(self):
        """Test line segments that touch at an endpoint."""
        result = line_segments_intersect(0, 0, 5, 5, 5, 5, 10, 10)
        assert result == True


class TestPointToSegmentDistance:
    """Tests for point to line segment distance calculation."""

    def test_point_to_segment_distance_perpendicular(self):
        """Test distance when point projects onto segment."""
        # Point (0, 1) to segment (0, 0) - (2, 0)
        dist, cx, cy = point_to_segment_distance(0, 1, 0, 0, 2, 0)
        assert abs(dist - 1.0) < 1e-6
        assert abs(cx - 0.0) < 1e-6
        assert abs(cy - 0.0) < 1e-6

    def test_point_to_segment_distance_endpoint(self):
        """Test distance to segment endpoint."""
        # Point (3, 0) to segment (0, 0) - (2, 0)
        dist, cx, cy = point_to_segment_distance(3, 0, 0, 0, 2, 0)
        assert abs(dist - 1.0) < 1e-6
        assert abs(cx - 2.0) < 1e-6

    def test_point_to_segment_distance_on_segment(self):
        """Test distance when point is on the segment."""
        # Point (1, 0) on segment (0, 0) - (2, 0)
        dist, cx, cy = point_to_segment_distance(1, 0, 0, 0, 2, 0)
        assert abs(dist) < 1e-6
        assert abs(cx - 1.0) < 1e-6
        assert abs(cy - 0.0) < 1e-6


class TestPeriodicBoundaries:
    """Tests for periodic boundary distance calculations."""

    def test_compute_minimum_distance_no_wrap(self):
        """Test minimum distance without wrapping."""
        pos_i = np.array([10.0, 10.0])
        pos_j = np.array([15.0, 10.0])
        box_size = 100.0
        result = compute_minimum_distance(pos_i, pos_j, box_size)
        expected = np.array([5.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_minimum_distance_with_wrap(self):
        """Test minimum distance with periodic wrapping."""
        pos_i = np.array([5.0, 5.0])
        pos_j = np.array([95.0, 5.0])
        box_size = 100.0
        result = compute_minimum_distance(pos_i, pos_j, box_size)
        # Should wrap: distance is -10, not +90
        expected = np.array([-10.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_minimum_distance_diagonal_wrap(self):
        """Test minimum distance with wrapping in both dimensions."""
        pos_i = np.array([5.0, 5.0])
        pos_j = np.array([95.0, 95.0])
        box_size = 100.0
        result = compute_minimum_distance(pos_i, pos_j, box_size)
        # Should wrap in both directions
        expected = np.array([-10.0, -10.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestWallIntersections:
    """Tests for wall intersection detection."""

    def test_line_intersects_any_wall(self):
        """Test checking line intersection with walls."""
        walls = np.array([
            [0, 0, 10, 10],
            [0, 10, 10, 0]
        ])
        # Line through middle should hit both walls
        assert line_intersects_any_wall(5, 0, 5, 10, walls) == True
        # Line outside should miss
        assert line_intersects_any_wall(20, 0, 20, 10, walls) == False

    def test_line_intersects_no_walls(self):
        """Test with no walls present."""
        walls = np.zeros((0, 4))
        assert line_intersects_any_wall(0, 0, 10, 10, walls) == False

    def test_particles_separated_by_wall(self):
        """Test if wall blocks particles."""
        walls = np.array([[5, 0, 5, 10]])
        pos_i = np.array([0.0, 5.0])
        pos_j = np.array([10.0, 5.0])
        # Wall at x=5 blocks particles at x=0 and x=10
        assert particles_separated_by_wall(pos_i, pos_j, walls) == True

    def test_particles_not_separated_by_wall(self):
        """Test particles not separated by wall."""
        walls = np.array([[5, 0, 5, 10]])
        pos_i = np.array([0.0, 5.0])
        pos_j = np.array([4.0, 5.0])
        # Both particles on same side of wall
        assert particles_separated_by_wall(pos_i, pos_j, walls) == False

    def test_particles_separated_by_wall_periodic(self):
        """Test wall blocking with periodic boundaries."""
        walls = np.array([[50, 0, 50, 100]])  # Wall at x=50
        pos_i = np.array([5.0, 50.0])
        pos_j = np.array([95.0, 50.0])
        box_size = 100.0
        # Periodic shortest path wraps around, should not be blocked
        # (distance is 10 wrapping left, not 90 going right through wall)
        result = particles_separated_by_wall_periodic(pos_i, pos_j, walls, box_size)
        assert result == False
