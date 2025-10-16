import numpy as np
import matplotlib.pyplot as plt


def uniform_vector_field(field_size: int, direction: float) -> np.ndarray:
    """
    Generate a uniform vector field where all vectors point in the same direction.

    Args:
        field_size: Size of the square field (field_size x field_size)
        direction: Angle in radians (0 = right, �/2 = up, � = left, 3�/2 = down)

    Returns:
        Array of shape (field_size, field_size, 2) with unit vectors
    """
    # Create unit vector from angle
    vector = np.array([np.cos(direction), np.sin(direction)])

    # Create field filled with this vector
    field = np.tile(vector, (field_size, field_size, 1))

    return field


def radial_vector_field(field_size: int, target: np.ndarray) -> np.ndarray:
    """
    Generate a radial vector field where all vectors point toward a target coordinate.

    Args:
        field_size: Size of the square field (field_size x field_size)
        target: numpy array [x, y] coordinate that all vectors point toward

    Returns:
        Array of shape (field_size, field_size, 2) with unit vectors pointing toward target
    """
    target_x, target_y = target[0], target[1]
    assert 0 <= target_x < field_size, f"target x={target_x} must be within [0, {field_size})"
    assert 0 <= target_y < field_size, f"target y={target_y} must be within [0, {field_size})"

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(field_size), np.arange(field_size), indexing='ij')

    # Calculate vectors pointing toward target
    dx = target_x - x_coords
    dy = target_y - y_coords

    # Calculate magnitudes
    magnitudes = np.sqrt(dx**2 + dy**2)

    # Normalize to unit vectors (avoid division by zero)
    field = np.zeros((field_size, field_size, 2))
    mask = magnitudes > 0
    field[mask, 0] = dx[mask] / magnitudes[mask]
    field[mask, 1] = dy[mask] / magnitudes[mask]

    return field


def limit_cycle_vector_field(field_size: int, cycle_radius: float = None, rad_mult: float = 3.0) -> np.ndarray:
    """
    Generate a vector field with a stable limit cycle (clockwise rotation).

    Points inside and outside the cycle are attracted to it, while points on
    the cycle rotate smoothly around the center.

    Args:
        field_size: Size of the square field (field_size x field_size)
        cycle_radius: Radius of the limit cycle. If None, defaults to field_size / 4

    Returns:
        Array of shape (field_size, field_size, 2) representing the vector field
    """
    if cycle_radius is None:
        cycle_radius = field_size / 4

    # Center of the field
    center = field_size / 2

    # Create coordinate grids (centered at middle of field)
    y_coords, x_coords = np.meshgrid(np.arange(field_size), np.arange(field_size), indexing='ij')

    # Calculate distance from center
    dx = x_coords - center
    dy = y_coords - center
    r = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero at center
    r = np.maximum(r, 1e-6)

    # Tangential component (clockwise rotation)
    # Tangent vector perpendicular to radial direction (clockwise)
    tangent_x = -dy / r
    tangent_y = dx / r

    # Radial component (attraction/repulsion toward limit cycle)
    # Stronger radial force makes vectors fall more sharply into the cycle
    radial_strength = rad_mult * (cycle_radius - r) / cycle_radius

    radial_x = (dx / r) * radial_strength
    radial_y = (dy / r) * radial_strength

    # Combine tangential and radial components
    field = np.zeros((field_size, field_size, 2))
    field[:, :, 0] = tangent_x + radial_x
    field[:, :, 1] = tangent_y + radial_y

    # Normalize to unit vectors
    magnitudes = np.sqrt(field[:, :, 0]**2 + field[:, :, 1]**2)
    magnitudes = np.maximum(magnitudes, 1e-6)
    field[:, :, 0] /= magnitudes
    field[:, :, 1] /= magnitudes

    return field


def visualize_vector_field(field: np.ndarray, step: int = 10):
    """
    Visualize a vector field using matplotlib quiver plot.

    Args:
        field: Array of shape (field_size, field_size, 2) representing the vector field
        step: Spacing between displayed vectors (higher = sparser display)
    """
    field_size = field.shape[0]

    # Create coordinate grids with specified step
    y, x = np.meshgrid(
        np.arange(0, field_size, step),
        np.arange(0, field_size, step),
        indexing='ij'
    )

    # Sample the vector field at the specified step
    u = field[::step, ::step, 0]
    v = field[::step, ::step, 1]

    # Create the plot
    plt.figure(figsize=(10, 10))
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=0.1)
    plt.xlim(0, field_size)
    plt.ylim(0, field_size)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.title('Vector Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():
    """Generate and display examples of each vector field type."""
    field_size = 350

    # Uniform vector field (45 degrees)
    print("Generating uniform vector field...")
    uniform_field = uniform_vector_field(field_size, np.pi / 2)
    visualize_vector_field(uniform_field, step=10)

    # # Radial vector field pointing to center
    # print("Generating radial vector field...")
    # radial_field = radial_vector_field(field_size, np.array([field_size / 2, field_size / 2]))
    # visualize_vector_field(radial_field, step=10)

    # Limit cycle vector field
    # print("Generating limit cycle vector field...")
    # cycle_field = limit_cycle_vector_field(field_size, cycle_radius=field_size / 4, rad_mult=5.0)
    # visualize_vector_field(cycle_field, step=10)


if __name__ == "__main__":
    main()
