import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle
import colorsys
import os
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

    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("visualizations", exist_ok=True)
        output_file = f"visualizations/payload_animation_{timestamp}.mp4"

    print("Creating animation...")

    start_time = time.time()

    # Extract parameters
    box_size = params['box_size']
    payload_radius = params['payload_radius']
    n_particles = params['n_particles']
    goal_position = params['goal_position']
    walls = params.get('walls', np.zeros((0, 5), dtype=np.float64))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set axis limits
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])

    # --- Inset axes: zoomed payload view locked to payload (top-left corner) ---
    # Use fig.add_axes (top-level) so the inset renders on top of all main-ax artists
    zoom_half = 40
    _ap = ax.get_position()
    inset_ax = fig.add_axes([
        _ap.x0 + 0.02 * _ap.width,
        _ap.y0 + 0.73 * _ap.height,
        0.25 * _ap.width,
        0.25 * _ap.height,
    ])
    inset_ax.set_xlim(-zoom_half, zoom_half)
    inset_ax.set_ylim(-zoom_half, zoom_half)
    inset_ax.set_aspect('equal')
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title('Payload zoom', fontsize=7, pad=2)
    for spine in inset_ax.spines.values():
        spine.set_linewidth(1.5)
    inset_ax.set_facecolor('white')

    # Initial relative positions for inset (periodic-corrected, payload at origin)
    px0, py0 = payload_positions[0, 0], payload_positions[0, 1]
    rel_x0 = positions[0, :, 0] - px0
    rel_y0 = positions[0, :, 1] - py0
    rel_x0 -= box_size * np.round(rel_x0 / box_size)
    rel_y0 -= box_size * np.round(rel_y0 / box_size)

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

    # Color mapping: rainbow cycling every 50 score units, starting at purple
    def get_particle_color_based_on_score(score_value):
        """Map score value to RGB using a rainbow colormap, looping every 50 steps.
        score 0: purple, cycles ROYGBIV, loops back to purple at score=50."""
        hue = (0.75 + (score_value % 50) / 50.0) % 1.0
        return colorsys.hsv_to_rgb(hue, 1.0, 1.0)

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
        s=np.pi * (params['particle_radius'] * 1)**2,  # Area of circle #BIG, particle size (* 3)
        c=particle_colors,
        alpha=0.7
    )
    inset_scatter = inset_ax.scatter(
        rel_x0, rel_y0,
        s=np.pi * (params['particle_radius'] * 2)**2,
        c=particle_colors,
        alpha=0.7
    )

    # Create payload: dark green outer disc + green inner disc to produce a thick interior ring
    border_fraction = 0.13  # dark green ring is 13 % of payload_radius wide
    payload_border = Circle(
        (payload_positions[0, 0], payload_positions[0, 1]),
        radius=payload_radius,
        facecolor='darkgreen',
        edgecolor='none',
        alpha=0.85
    )
    ax.add_patch(payload_border)
    payload = Circle(
        (payload_positions[0, 0], payload_positions[0, 1]),
        radius=payload_radius * (1 - border_fraction),
        facecolor='green',
        edgecolor='none',
        alpha=0.85
    )
    ax.add_patch(payload)

    # Inset payload circles (payload is always at origin in the inset frame)
    inset_payload_border = Circle(
        (0, 0), radius=payload_radius, facecolor='darkgreen', edgecolor='none', alpha=0.85
    )
    inset_ax.add_patch(inset_payload_border)
    inset_payload_patch = Circle(
        (0, 0), radius=payload_radius * (1 - border_fraction), facecolor='green', edgecolor='none', alpha=0.85
    )
    inset_ax.add_patch(inset_payload_patch)

    # Create goal visualization
    goal_marker_size = 22.5
    goal, = ax.plot(goal_position[0], goal_position[1], 'g*', markersize=goal_marker_size, markeredgewidth=1.5, markeredgecolor='darkgreen')
    # Green circle whose radius places its edge at the star tips (markersize/2 pts → data units)
    goal_circle_radius = (goal_marker_size / 2) * box_size / (10 * 72) * 1.45
    goal_circle = Circle(
        (goal_position[0], goal_position[1]),
        radius=goal_circle_radius,
        facecolor='none',
        edgecolor='green',
        linewidth=2,
        zorder=5
    )
    ax.add_patch(goal_circle)

    # Inset goal marker (position updated each frame relative to payload)
    gx0_rel = goal_position[0] - px0
    gy0_rel = goal_position[1] - py0
    gx0_rel -= box_size * np.round(gx0_rel / box_size)
    gy0_rel -= box_size * np.round(gy0_rel / box_size)
    inset_goal, = inset_ax.plot(gx0_rel, gy0_rel, 'g*', markersize=8, markeredgewidth=1, markeredgecolor='darkgreen', zorder=5)

    # Draw walls (straight or curved)
    def _arc_plot_points(x1, y1, x2, y2, c, n_pts=64):
        """Return (xs, ys) arrays to draw a wall arc. Falls back to segment when c≈0."""
        if abs(c) < 1e-9:
            return [x1, x2], [y1, y2]
        chx, chy = x2 - x1, y2 - y1
        chord_len = np.sqrt(chx**2 + chy**2)
        if chord_len < 1e-10:
            return [x1, x2], [y1, y2]
        R = chord_len / (2.0 * abs(c))
        h = np.sqrt(max(R**2 - (chord_len / 2)**2, 0.0))
        cw_perp = np.array([chy, -chx]) / chord_len
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        sc = 1.0 if c > 0 else -1.0
        center = mid + sc * h * cw_perp
        theta1 = np.arctan2(y1 - center[1], x1 - center[0])
        theta2 = np.arctan2(y2 - center[1], x2 - center[0])
        span_ccw = (theta2 - theta1) % (2 * np.pi)
        if span_ccw <= np.pi:
            thetas = np.linspace(theta1, theta1 + span_ccw, n_pts)
        else:
            span_cw = 2 * np.pi - span_ccw
            thetas = np.linspace(theta1, theta1 - span_cw, n_pts)
        return (center[0] + R * np.cos(thetas)).tolist(), (center[1] + R * np.sin(thetas)).tolist()

    wall_lines = []
    for i in range(walls.shape[0]):
        xs, ys = _arc_plot_points(
            walls[i, 0], walls[i, 1], walls[i, 2], walls[i, 3], walls[i, 4]
        )
        line, = ax.plot(
            xs, ys,
            color='black',
            linewidth=2, #BIG
            solid_capstyle='round',
            zorder=10
        )
        wall_lines.append(line)

    def _inset_wall_data(wall_idx, px, py):
        """Return (xs, ys) for a wall segment in payload-relative inset coordinates."""
        mx = (walls[wall_idx, 0] + walls[wall_idx, 2]) / 2
        my = (walls[wall_idx, 1] + walls[wall_idx, 3]) / 2
        shift_x = -box_size * np.round((mx - px) / box_size)
        shift_y = -box_size * np.round((my - py) / box_size)
        wx1 = walls[wall_idx, 0] - px + shift_x
        wy1 = walls[wall_idx, 1] - py + shift_y
        wx2 = walls[wall_idx, 2] - px + shift_x
        wy2 = walls[wall_idx, 3] - py + shift_y
        return _arc_plot_points(wx1, wy1, wx2, wy2, walls[wall_idx, 4])

    inset_wall_lines = []
    for i in range(walls.shape[0]):
        xs, ys = _inset_wall_data(i, px0, py0)
        line, = inset_ax.plot(xs, ys, color='black', linewidth=1.5, solid_capstyle='round', zorder=10)
        inset_wall_lines.append(line)

    # Create payload trajectory
    trajectory, = ax.plot(
        payload_positions[0:1, 0],
        payload_positions[0:1, 1],
        # '--',
        color="#31DC13",
        alpha=0.8,
        linewidth=2.5
    )
    inset_trajectory, = inset_ax.plot(
        [0], [0],
        color="#31DC13",
        alpha=0.8,
        linewidth=2.5,
        zorder=3
    )

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
        artists = [scatter, payload_border, payload, trajectory, goal]
        if quiver is not None:
            artists.append(quiver)
        artists.extend(wall_lines)
        artists.extend([inset_scatter, inset_goal, inset_trajectory])
        artists.extend(inset_wall_lines)
        return artists

    def update(frame):
        """Update the animation for each frame."""
        if frame % 50 == 0:
            print(f"Progress: Frame {frame}")

        px, py = payload_positions[frame, 0], payload_positions[frame, 1]

        # Update payload
        payload_border.center = (px, py)
        payload.center = (px, py)

        # Update payload trajectory
        trajectory_end = min(frame + 1, len(payload_positions))
        trajectory.set_data(
            payload_positions[:trajectory_end, 0],
            payload_positions[:trajectory_end, 1]
        )

        # Update inset trajectory (payload-relative positions with periodic correction)
        traj_x = payload_positions[:trajectory_end, 0] - px
        traj_y = payload_positions[:trajectory_end, 1] - py
        traj_x -= box_size * np.round(traj_x / box_size)
        traj_y -= box_size * np.round(traj_y / box_size)
        inset_trajectory.set_data(traj_x, traj_y)

        # Particle positions & colors update
        scatter.set_offsets(positions[frame])
        if particle_scores is not None:
            colors = [get_particle_color_based_on_score(score) for score in particle_scores[frame]]
        else:
            colors = [get_particle_color_based_on_curvity(cv) for cv in curvity_values[frame]]
        scatter.set_color(colors)

        # Update polarity vectors if enabled
        if quiver is not None and polarity is not None:
            arrow_length = 8.0
            quiver.set_offsets(positions[frame])
            quiver.set_UVC(polarity[frame, :, 0] * arrow_length,
                          polarity[frame, :, 1] * arrow_length)

        # --- Update inset (zoomed payload view, payload fixed at origin) ---
        rel_x = positions[frame, :, 0] - px
        rel_y = positions[frame, :, 1] - py
        rel_x -= box_size * np.round(rel_x / box_size)
        rel_y -= box_size * np.round(rel_y / box_size)
        inset_scatter.set_offsets(np.column_stack([rel_x, rel_y]))
        inset_scatter.set_color(colors)

        gx_rel = goal_position[0] - px
        gy_rel = goal_position[1] - py
        gx_rel -= box_size * np.round(gx_rel / box_size)
        gy_rel -= box_size * np.round(gy_rel / box_size)
        inset_goal.set_data([gx_rel], [gy_rel])

        for i, line in enumerate(inset_wall_lines):
            xs, ys = _inset_wall_data(i, px, py)
            line.set_data(xs, ys)

        artists = [scatter, payload_border, payload, trajectory]
        if quiver is not None:
            artists.append(quiver)
        artists.extend([inset_scatter, inset_goal, inset_trajectory])
        artists.extend(inset_wall_lines)
        return artists

    # Create animation
    n_frames = positions.shape[0]

    sim_seconds_per_real_second = 150 #75 # Increase frame skip for fewer frames to render if its too slow
    target_fps = 15

    # Calculate frame skip to maintain consistent sim-time to real-time ratio
    skip = max(1, int(sim_seconds_per_real_second / target_fps))

    # Create sequence of frames to include
    frames = range(0, n_frames, skip)
    print(f"Number of frames: {n_frames}")

    plt.rcParams['savefig.dpi'] = 150  # Lower dpi for faster rendering

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=False,
        interval=120  # Increased from 50
    )

    # writer = PillowWriter(fps=target_fps) # for gifs, but its slower
    writer = FFMpegWriter(
        fps=target_fps,
        bitrate=8000,
        codec='libx264',
        extra_args=['-pix_fmt', 'yuv420p', '-crf', '18']
    ) # mp4 with high quality settings (requires FFmpeg installation)

    anim.save(output_file, writer=writer)
    plt.close()

    end_time = time.time()

    print(f"Animation saved as '{output_file}'")
    print(f"Animation creation time: {end_time - start_time:.2f} seconds")
