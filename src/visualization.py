import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle, Wedge
import time

from .circles import parametric_curve


#####################################################
# Animation and visualization functions             #
#####################################################

def create_payload_animation(positions, orientations, velocities, payload_positions, params,
                            curvity_values, output_file='visualizations/payload_animation_00.mp4',
                            color_neg1=(1.0, 0.0, 0.0), color_0=(0.5, 0.5, 0.5), color_pos1=(0.0, 0.0, 1.0)):
    """Create an animation of the payload transport simulation.

    Particles are colored by their fixed curvity values.

    Args:
        color_neg1: RGB tuple for curvity = -1 (default: red)
        color_0: RGB tuple for curvity = 0 (default: gray)
        color_pos1: RGB tuple for curvity = +1 (default: blue)
    """

    print("Creating animation...")

    start_time = time.time()

    # Extract parameters
    box_size = params['box_size']
    payload_radius = params['payload_radius']
    n_particles = params['n_particles']
    walls = params.get('walls', np.zeros((0, 5), dtype=np.float64))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set axis limits
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title('FAABP Cooperative Transport Simulation')
    ax.grid(True, alpha=0.3)

    # Color mapping function using parametrized colors
    def get_particle_color_based_on_curvity(curvity_value):
        """Map curvity value to RGB color with smooth gradient.
        Uses the parametrized colors for -1, 0, and +1 curvity values."""
        # Clamp curvity to [-1, 1] range
        c = np.clip(curvity_value, -1, 1)

        if c < 0:
            # Interpolate from color_neg1 to color_0
            t = (c + 1)  # Map [-1, 0] to [0, 1]
            r = color_neg1[0] + t * (color_0[0] - color_neg1[0])
            g = color_neg1[1] + t * (color_0[1] - color_neg1[1])
            b = color_neg1[2] + t * (color_0[2] - color_neg1[2])
        else:
            # Interpolate from color_0 to color_pos1
            t = c  # Map [0, 1] to [0, 1]
            r = color_0[0] + t * (color_pos1[0] - color_0[0])
            g = color_0[1] + t * (color_pos1[1] - color_0[1])
            b = color_0[2] + t * (color_pos1[2] - color_0[2])

        return (r, g, b)

    # Initialize particle colors based on curvity
    particle_colors = [get_particle_color_based_on_curvity(curvity_values[0, i]) for i in range(n_particles)]

    scatter = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        s=np.pi * (params['particle_radius'] * 4)**2,  # Area of circle (scaled up for visibility)
        c=particle_colors,
        alpha=0.7
    )

    # Create payload
    payload = Circle(
        (payload_positions[0, 0], payload_positions[0, 1]),
        radius=payload_radius,
        color='gray',
        alpha=0.7
    )
    ax.add_patch(payload)

    # Draw walls
    wall_lines = []
    for i in range(walls.shape[0]):
        x1, y1, x2, y2, c = walls[i, 0], walls[i, 1], walls[i, 2], walls[i, 3], walls[i, 4]

        if abs(c) < 1e-10:
            # Straight wall
            line, = ax.plot(
                [x1, x2],  # x-coordinates: [x1, x2]
                [y1, y2],  # y-coordinates: [y1, y2]
                color='black',
                linewidth=3,
                solid_capstyle='round',
                zorder=10  # Draw on top of particles
            )
        else:
            # Curved wall
            curve_x, curve_y = parametric_curve((x1, y1), (x2, y2), c, num_points=50)
            line, = ax.plot(
                curve_x, curve_y,
                color='black',
                linewidth=3,
                solid_capstyle='round',
                zorder=10  # Draw on top of particles
            )
        wall_lines.append(line)

    # Create payload trajectory
    trajectory, = ax.plot(
        payload_positions[0:1, 0],
        payload_positions[0:1, 1],
        'k--',
        alpha=0.5,
        linewidth=1.0
    )

    # Add parameters text
    params_text = ax.text(-0.02, -0.065, f'n_particles: {n_particles}, particle radius: {params["particle_radius"][0]}, payload radius: {payload_radius}', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    params_text_2 = ax.text(-0.02, -0.093, f'orientational noise: {params["rot_diffusion"][0]}, particle mobility: {params["mobility"][0]}, payload mobility: {params["payload_mobility"]}', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')

    # Add time counter
    time_text = ax.text(0.02, 0.98, 'Frame: 0', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')

    def init():
        """Initialize the animation."""
        artists = [scatter, payload, trajectory, time_text, params_text, params_text_2]
        # Add wall lines (they don't change, but include for completeness)
        artists.extend(wall_lines)
        return artists

    def update(frame):
        """Update the animation for each frame."""
        # Update time counter
        time_text.set_text(f'Frame: {frame}')

        # Report progress periodically
        if frame % 50 == 0:
            print(f"Progress: Frame {frame}")

        # Update payload
        payload.center = (payload_positions[frame, 0], payload_positions[frame, 1])

        # Update payload trajectory
        trajectory_end = min(frame + 1, len(payload_positions))
        trajectory.set_data(
            payload_positions[:trajectory_end, 0],
            payload_positions[:trajectory_end, 1]
        )

        # Particle positions & colors update
        scatter.set_offsets(positions[frame])
        # Color by curvity
        scatter.set_color([get_particle_color_based_on_curvity(cv) for cv in curvity_values[frame]])

        artists = [scatter, payload, trajectory, time_text]
        return artists

    # Create animation
    n_frames = positions.shape[0]

    sim_seconds_per_real_second = 75 # Increase frame skip for fewer frames to render if its too slow
    target_fps = 15

    # Calculate frame skip to maintain consistent sim-time to real-time ratio
    skip = max(1, int(sim_seconds_per_real_second / target_fps))

    # Create sequence of frames to include
    frames = range(0, n_frames, skip)
    print(f"Number of frames: {n_frames}")

    plt.rcParams['savefig.dpi'] = 170  # Lower dpi for faster rendering

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=True,
        interval=120  # Increased from 50
    )

    # Try FFMpeg first, fall back to Pillow for GIF if FFMpeg unavailable
    try:
        writer = FFMpegWriter(
            fps=target_fps,
            bitrate=8000,
            codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p', '-crf', '18']
        )
        anim.save(output_file, writer=writer)
    except FileNotFoundError:
        print("FFMpeg not found, falling back to GIF output...")
        gif_file = output_file.rsplit('.', 1)[0] + '.gif'
        writer = PillowWriter(fps=target_fps)
        anim.save(gif_file, writer=writer)
        output_file = gif_file

    plt.close()

    end_time = time.time()

    print(f"Animation saved as '{output_file}'")
    print(f"Animation creation time: {end_time - start_time:.2f} seconds")


def create_hollow_payload_animation(positions, orientations, velocities, payload_positions, params,
                                     curvity_values, output_file='visualizations/hollow_payload_animation_00.mp4',
                                     color_neg1=(1.0, 0.0, 0.0), color_0=(0.5, 0.5, 0.5), color_pos1=(0.0, 0.0, 1.0)):
    """Create an animation of the hollow payload transport simulation.

    The hollow payload is rendered as a circle boundary.

    Args:
        color_neg1: RGB tuple for curvity = -1 (default: red)
        color_0: RGB tuple for curvity = 0 (default: gray)
        color_pos1: RGB tuple for curvity = +1 (default: blue)
    """

    print("Creating hollow payload animation...")

    start_time = time.time()

    # Extract parameters
    box_size = params['box_size']
    payload_radius = params['payload_radius']
    n_particles = params['n_particles']
    walls = params.get('walls', np.zeros((0, 5), dtype=np.float64))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set axis limits
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title('FAABP Hollow Payload Transport Simulation')
    ax.grid(True, alpha=0.3)

    # Color mapping function using parametrized colors
    def get_particle_color_based_on_curvity(curvity_value):
        """Map curvity value to RGB color with smooth gradient."""
        c = np.clip(curvity_value, -1, 1)

        if c < 0:
            t = (c + 1)
            r = color_neg1[0] + t * (color_0[0] - color_neg1[0])
            g = color_neg1[1] + t * (color_0[1] - color_neg1[1])
            b = color_neg1[2] + t * (color_0[2] - color_neg1[2])
        else:
            t = c
            r = color_0[0] + t * (color_pos1[0] - color_0[0])
            g = color_0[1] + t * (color_pos1[1] - color_0[1])
            b = color_0[2] + t * (color_pos1[2] - color_0[2])

        return (r, g, b)

    # Initialize particle colors based on curvity
    particle_colors = [get_particle_color_based_on_curvity(curvity_values[0, i]) for i in range(n_particles)]

    scatter = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        s=np.pi * (params['particle_radius'] * 1)**2,  # Scaled up for visibility
        c=particle_colors,
        alpha=0.7,
        zorder=5  # Render particles on top of payload
    )

    # Create hollow payload as a circle boundary (no fill, just edge)
    payload_circle = Circle(
        (payload_positions[0, 0], payload_positions[0, 1]),
        radius=payload_radius,
        fill=False,
        edgecolor='darkgray',
        linewidth=2.5
    )
    ax.add_patch(payload_circle)

    # Draw walls
    wall_lines = []
    for i in range(walls.shape[0]):
        x1, y1, x2, y2, c = walls[i, 0], walls[i, 1], walls[i, 2], walls[i, 3], walls[i, 4]

        if abs(c) < 1e-10:
            line, = ax.plot(
                [x1, x2],
                [y1, y2],
                color='black',
                linewidth=3,
                solid_capstyle='round',
                zorder=10
            )
        else:
            curve_x, curve_y = parametric_curve((x1, y1), (x2, y2), c, num_points=50)
            line, = ax.plot(
                curve_x, curve_y,
                color='black',
                linewidth=3,
                solid_capstyle='round',
                zorder=10
            )
        wall_lines.append(line)

    # Create payload trajectory
    trajectory, = ax.plot(
        payload_positions[0:1, 0],
        payload_positions[0:1, 1],
        'k--',
        alpha=0.5,
        linewidth=1.0
    )

    # Add parameters text
    params_text = ax.text(-0.02, -0.065,
                          f'n_particles: {n_particles}, particle radius: {params["particle_radius"][0]}, '
                          f'payload radius: {payload_radius}',
                          transform=ax.transAxes, fontsize=12, verticalalignment='top')
    params_text_2 = ax.text(-0.02, -0.093,
                            f'orientational noise: {params["rot_diffusion"][0]}, '
                            f'particle mobility: {params["mobility"][0]}, payload mobility: {params["payload_mobility"]}',
                            transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # Add time counter
    time_text = ax.text(0.02, 0.98, 'Frame: 0', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    def init():
        """Initialize the animation."""
        artists = [scatter, payload_circle, trajectory, time_text, params_text, params_text_2]
        artists.extend(wall_lines)
        return artists

    def update(frame):
        """Update the animation for each frame."""
        time_text.set_text(f'Frame: {frame}')

        if frame % 50 == 0:
            print(f"Progress: Frame {frame}")

        # Update hollow payload position
        payload_circle.center = (payload_positions[frame, 0], payload_positions[frame, 1])

        # Update payload trajectory
        trajectory_end = min(frame + 1, len(payload_positions))
        trajectory.set_data(
            payload_positions[:trajectory_end, 0],
            payload_positions[:trajectory_end, 1]
        )

        # Particle positions & colors update
        scatter.set_offsets(positions[frame])
        scatter.set_color([get_particle_color_based_on_curvity(cv) for cv in curvity_values[frame]])

        artists = [scatter, payload_circle, trajectory, time_text]
        return artists

    # Create animation
    n_frames = positions.shape[0]

    sim_seconds_per_real_second = 75
    target_fps = 15

    skip = max(1, int(sim_seconds_per_real_second / target_fps))
    frames = range(0, n_frames, skip)
    print(f"Number of frames: {n_frames}")

    plt.rcParams['savefig.dpi'] = 170

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=True,
        interval=120
    )

    # Try FFMpeg first, fall back to Pillow for GIF if FFMpeg unavailable
    try:
        writer = FFMpegWriter(
            fps=target_fps,
            bitrate=8000,
            codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p', '-crf', '18']
        )
        anim.save(output_file, writer=writer)
    except FileNotFoundError:
        print("FFMpeg not found, falling back to GIF output...")
        gif_file = output_file.rsplit('.', 1)[0] + '.gif'
        writer = PillowWriter(fps=target_fps)
        anim.save(gif_file, writer=writer)
        output_file = gif_file

    plt.close()

    end_time = time.time()

    print(f"Animation saved as '{output_file}'")
    print(f"Animation creation time: {end_time - start_time:.2f} seconds")


def create_multi_hollow_payload_animation(
    positions, orientations, velocities,
    small_payload_positions, large_payload_positions,
    params, curvity_values,
    output_file='visualizations/multi_hollow_payload_animation.mp4',
    color_neg1=(1.0, 0.0, 0.0), color_0=(0.5, 0.5, 0.5), color_pos1=(0.0, 0.0, 1.0)
):
    """Create an animation of the multi-hollow payload transport simulation.

    10 small hollow payloads inside 1 large enclosing hollow payload.

    Args:
        positions: (n_frames, n_particles, 2) particle positions
        small_payload_positions: (n_frames, n_small_payloads, 2) small payload positions
        large_payload_positions: (n_frames, 2) large payload positions
        params: simulation parameters dict
        curvity_values: (n_frames, n_particles) curvity values
        output_file: output video file path
    """
    print("Creating multi-hollow payload animation...")

    start_time_anim = time.time()

    # Extract parameters
    box_size = params['box_size']
    small_payload_radii = params['small_payload_radii']
    large_payload_radius = params['large_payload_radius']
    n_particles = params['n_particles']
    n_small_payloads = params['n_small_payloads']
    walls = params.get('walls', np.zeros((0, 5), dtype=np.float64))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title('FAABP Multi-Hollow Payload Transport')
    ax.grid(True, alpha=0.3)

    # Color mapping function
    def get_particle_color_based_on_curvity(curvity_value):
        c = np.clip(curvity_value, -1, 1)
        if c < 0:
            t = (c + 1)
            r = color_neg1[0] + t * (color_0[0] - color_neg1[0])
            g = color_neg1[1] + t * (color_0[1] - color_neg1[1])
            b = color_neg1[2] + t * (color_0[2] - color_neg1[2])
        else:
            t = c
            r = color_0[0] + t * (color_pos1[0] - color_0[0])
            g = color_0[1] + t * (color_pos1[1] - color_0[1])
            b = color_0[2] + t * (color_pos1[2] - color_0[2])
        return (r, g, b)

    # Initialize particle colors
    particle_colors = [get_particle_color_based_on_curvity(curvity_values[0, i])
                       for i in range(n_particles)]

    # Get particle radius for scatter size
    particle_r = params['particle_radius'][0] if hasattr(params['particle_radius'], '__len__') else params['particle_radius']

    scatter = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        s=np.pi * (particle_r * 1)**2,
        c=particle_colors,
        alpha=0.7,
        zorder=5
    )

    # Create small payload circles (hollow, blue)
    small_payload_circles = []
    for i in range(n_small_payloads):
        circle = Circle(
            (small_payload_positions[0, i, 0], small_payload_positions[0, i, 1]),
            radius=small_payload_radii[i],
            fill=False,
            edgecolor='blue',
            linewidth=2.0,
            zorder=3
        )
        ax.add_patch(circle)
        small_payload_circles.append(circle)

    # Create large payload circle (hollow, dark red, thicker)
    large_payload_circle = Circle(
        (large_payload_positions[0, 0], large_payload_positions[0, 1]),
        radius=large_payload_radius,
        fill=False,
        edgecolor='darkred',
        linewidth=3.0,
        zorder=2
    )
    ax.add_patch(large_payload_circle)

    # Draw walls
    wall_lines = []
    for i in range(walls.shape[0]):
        x1, y1, x2, y2, c = walls[i, 0], walls[i, 1], walls[i, 2], walls[i, 3], walls[i, 4]

        if abs(c) < 1e-10:
            line, = ax.plot([x1, x2], [y1, y2], color='black', linewidth=3,
                           solid_capstyle='round', zorder=10)
        else:
            curve_x, curve_y = parametric_curve((x1, y1), (x2, y2), c, num_points=50)
            line, = ax.plot(curve_x, curve_y, color='black', linewidth=3,
                           solid_capstyle='round', zorder=10)
        wall_lines.append(line)

    # Large payload trajectory
    trajectory, = ax.plot(
        large_payload_positions[0:1, 0],
        large_payload_positions[0:1, 1],
        'r--',
        alpha=0.5,
        linewidth=1.0
    )

    # Add time counter
    time_text = ax.text(0.02, 0.98, 'Frame: 0', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')

    # Add parameters text
    params_text = ax.text(-0.02, -0.065,
                          f'n_particles: {n_particles}, n_small_payloads: {n_small_payloads}, '
                          f'small_r: {small_payload_radii[0]:.1f}, large_r: {large_payload_radius:.1f}',
                          transform=ax.transAxes, fontsize=12, verticalalignment='top')

    def init():
        artists = [scatter, large_payload_circle, trajectory, time_text, params_text]
        artists.extend(small_payload_circles)
        artists.extend(wall_lines)
        return artists

    def update(frame):
        time_text.set_text(f'Frame: {frame}')

        if frame % 50 == 0:
            print(f"Progress: Frame {frame}")

        # Update particles
        scatter.set_offsets(positions[frame])
        scatter.set_color([get_particle_color_based_on_curvity(cv)
                           for cv in curvity_values[frame]])

        # Update small payloads
        for i in range(n_small_payloads):
            small_payload_circles[i].center = (
                small_payload_positions[frame, i, 0],
                small_payload_positions[frame, i, 1]
            )

        # Update large payload
        large_payload_circle.center = (
            large_payload_positions[frame, 0],
            large_payload_positions[frame, 1]
        )

        # Update trajectory
        trajectory_end = min(frame + 1, len(large_payload_positions))
        trajectory.set_data(
            large_payload_positions[:trajectory_end, 0],
            large_payload_positions[:trajectory_end, 1]
        )

        artists = [scatter, large_payload_circle, trajectory, time_text]
        artists.extend(small_payload_circles)
        return artists

    # Create animation
    n_frames = positions.shape[0]

    sim_seconds_per_real_second = 75
    target_fps = 15

    skip = max(1, int(sim_seconds_per_real_second / target_fps))
    frames = range(0, n_frames, skip)
    print(f"Number of frames: {n_frames}")

    plt.rcParams['savefig.dpi'] = 170

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=True,
        interval=120
    )

    # Try FFMpeg first, fall back to Pillow for GIF
    try:
        writer = FFMpegWriter(
            fps=target_fps,
            bitrate=8000,
            codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p', '-crf', '18']
        )
        anim.save(output_file, writer=writer)
    except FileNotFoundError:
        print("FFMpeg not found, falling back to GIF output...")
        gif_file = output_file.rsplit('.', 1)[0] + '.gif'
        writer = PillowWriter(fps=target_fps)
        anim.save(gif_file, writer=writer)
        output_file = gif_file

    plt.close()

    end_time_anim = time.time()

    print(f"Animation saved as '{output_file}'")
    print(f"Animation creation time: {end_time_anim - start_time_anim:.2f} seconds")
