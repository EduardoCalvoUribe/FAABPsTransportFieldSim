import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle
import time


#####################################################
# Animation and visualization functions             #
#####################################################

def create_payload_animation(positions, orientations, velocities, payload_positions, params,
                            curvity_values, output_file='visualizations/payload_animation_00.mp4',
                            show_vectors=False, polarity=None, particle_scores=None):
    """Create an animation of the payload transport simulation.

    Args:
        show_vectors: If True, display the polarity vectors as arrows attached to particles
        polarity: Array of polarity vectors over time (n_frames, n_particles, 2)
        particle_scores: Array of particle scores over time (n_frames, n_particles). If provided, colors particles by score instead of curvity.
    """

    print("Creating animation...")

    start_time = time.time()

    # Extract parameters
    box_size = params['box_size']
    payload_radius = params['payload_radius']
    n_particles = params['n_particles']
    goal_position = params['goal_position']
    walls = params.get('walls', np.zeros((0, 4), dtype=np.float64))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set axis limits
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title('FAABP Cooperative Transport Simulation')
    ax.grid(True, alpha=0.3)

    # Color mapping functions
    # STANDARD: Color mapping: curvity -1 (dark blue) -> 0 (gray) -> +1 (red)
    def get_particle_color_based_on_curvity(curvity_value):
        """Map curvity value to RGB color with smooth gradient.
        -1: dark blue, 0: gray, +1: red"""
        # Clamp curvity to [-1, 1] range
        c = np.clip(curvity_value, -1, 1)

        if c < 0:
            # Interpolate from dark blue (0, 0, 0.5) to gray (0.5, 0.5, 0.5)
            t = (c + 1)  # Map [-1, 0] to [0, 1]
            r = 0.0 + t * 0.5
            g = 0.0 + t * 0.5
            b = 0.5 + t * 0.0
        else:
            # Interpolate from gray (0.5, 0.5, 0.5) to red (1, 0, 0)
            t = c  # Map [0, 1] to [0, 1]
            r = 0.5 + t * 0.5
            g = 0.5 - t * 0.5
            b = 0.5 - t * 0.5

        return (r, g, b)

    # DEBUG: Color mapping: score 0 (blue) -> 999+ (orange)
    def get_particle_color_based_on_score(score_value):
        """Map score value to RGB color with linear gradient.
        0: blue (0, 0, 1), 20+: orange (1, 0.5, 0)"""
        # Clamp score to [0, 50] range
        s = np.clip(score_value, 0, 10)

        # Linear interpolation from blue to orange
        t = s / 10.0  # Map [0, 50] to [0, 1]

        r = 0.0 + t * 1.0  # 0 -> 1
        g = 0.0 + t * 0.5  # 0 -> 0.5
        b = 1.0 - t * 1.0  # 1 -> 0

        return (r, g, b)

    # Initialize particle colors
    if particle_scores is not None:
        # Color by score
        particle_colors = [get_particle_color_based_on_score(particle_scores[0, i]) for i in range(n_particles)]
    else:
        # Color by curvity (fallback)
        particle_colors = [get_particle_color_based_on_curvity(curvity_values[0, i]) for i in range(n_particles)]

    scatter = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        s=np.pi * (params['particle_radius'] * 2)**2,  # Area of circle
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

    # Create goal visualization
    goal, = ax.plot(goal_position[0], goal_position[1], 'g*', markersize=15, markeredgewidth=1.5, markeredgecolor='darkgreen')
    # Green star marker for goal point

    # Draw walls
    wall_lines = []
    for i in range(walls.shape[0]):
        line, = ax.plot(
            [walls[i, 0], walls[i, 2]],  # x-coordinates: [x1, x2]
            [walls[i, 1], walls[i, 3]],  # y-coordinates: [y1, y2]
            color='black',
            linewidth=4,
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

    # Create quiver plot for polarity vectors if enabled
    quiver = None
    if show_vectors and polarity is not None:
        # Scale arrows to be visible - multiply vectors by a scaling factor
        arrow_length = 8.0  # Length multiplier for visibility
        quiver = ax.quiver(
            positions[0, :, 0],
            positions[0, :, 1],
            polarity[0, :, 0] * arrow_length,
            polarity[0, :, 1] * arrow_length,
            angles='xy',
            scale_units='xy',
            scale=1,
            color='darkblue',
            alpha=0.3,
            width=0.004,
            headwidth=3.5,
            headlength=4.5
        )

    def init():
        """Initialize the animation."""
        artists = [scatter, payload, trajectory, time_text, params_text, params_text_2, goal]
        if quiver is not None:
            artists.append(quiver)
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
        # Color by score if available, otherwise by curvity
        if particle_scores is not None:
            scatter.set_color([get_particle_color_based_on_score(score) for score in particle_scores[frame]])
        else:
            scatter.set_color([get_particle_color_based_on_curvity(cv) for cv in curvity_values[frame]])

        # Update polarity vectors if enabled
        if quiver is not None and polarity is not None:
            arrow_length = 8.0  # Same as initialization
            quiver.set_offsets(positions[frame])
            quiver.set_UVC(polarity[frame, :, 0] * arrow_length,
                          polarity[frame, :, 1] * arrow_length)

        artists = [scatter, payload, trajectory, time_text]
        if quiver is not None:
            artists.append(quiver)
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

    #writer = PillowWriter(fps=target_fps) # for gifs, but its slower
    writer = FFMpegWriter(
        fps=target_fps,
        bitrate=8000,
        codec='libx264',
        extra_args=['-pix_fmt', 'yuv420p', '-crf', '18']
    ) # mp4 with high quality settings

    anim.save(output_file, writer=writer)
    plt.close()

    end_time = time.time()

    print(f"Animation saved as '{output_file}'")
    print(f"Animation creation time: {end_time - start_time:.2f} seconds")
