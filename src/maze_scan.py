"""Maze scanning utilities for converting maze images to 2D arrays."""

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import os


def get_image_filepath(filename: str, images_folder: str = None) -> str:
    """Resolve full path to image file. Defaults to project's /images folder."""
    if images_folder is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        images_folder = os.path.join(project_root, 'images')

    if not os.path.isabs(filename):
        filepath = os.path.join(images_folder, filename)
    else:
        filepath = filename

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")

    return filepath


def load_image(filepath: str, output_size: tuple = None) -> Image.Image:
    """Load image from file, optionally resize."""
    image = Image.open(filepath)
    if output_size:
        image = image.resize(output_size, Image.Resampling.LANCZOS)
    return image


def image_to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to grayscale numpy array (0-255). Transparent pixels become white."""
    # Handle transparency by compositing onto white background
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
        image = background
    grayscale = image.convert('L')
    return np.array(grayscale, dtype=np.float64)


def normalize_to_binary_range(array: np.ndarray) -> np.ndarray:
    """Normalize to 0-1 range, inverted: white=0, black=1."""
    return 1.0 - (array / 255.0)


def apply_gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur with given sigma."""
    if sigma > 0:
        return gaussian_filter(array, sigma=sigma)
    return array


def min_max_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize array so min=0 and max=1."""
    arr_min = array.min()
    arr_max = array.max()
    if arr_max - arr_min == 0:
        return np.zeros_like(array)
    return (array - arr_min) / (arr_max - arr_min)


def threshold_maze(array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert to binary: values >= threshold become 1, else 0."""
    return (array >= threshold).astype(np.float64)


def load_maze_from_image(
    filename: str,
    sigma: float = 2.0,
    output_size: tuple = None,
    images_folder: str = None,
    upscale_factor: int = 1
) -> np.ndarray:
    """Load image, convert to monochrome, apply Gaussian blur. Returns 2D array (0=white, 1=black)."""
    filepath = get_image_filepath(filename, images_folder)
    image = load_image(filepath, output_size)

    # Upscale before processing for smoother blur
    if upscale_factor > 1:
        new_size = (image.width * upscale_factor, image.height * upscale_factor)
        image = image.resize(new_size, Image.Resampling.NEAREST)

    grayscale = image_to_grayscale(image)
    normalized = normalize_to_binary_range(grayscale)
    blurred = apply_gaussian_blur(normalized, sigma)
    result = min_max_normalize(blurred)
    return result


# Alias for backwards compatibility
load_maze_from_svg = load_maze_from_image


def compute_maze_gradients(maze_array: np.ndarray) -> tuple:
    """Compute x and y gradients of maze array. Gradient points toward walls (high values)."""
    grad_y, grad_x = np.gradient(maze_array)
    return grad_x.astype(np.float64), grad_y.astype(np.float64)
