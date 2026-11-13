"""Helpers shared by the fig4 plotting scripts (b/, d/) and code/payload_dotproduct.py."""
from matplotlib.lines import Line2D


def unravel_trajectory(trajectory, box_size=350):
    """Unravel a trajectory that was created with periodic boundary conditions.
    
    Args:
        trajectory: numpy array of shape (n_steps, 2) containing x,y coordinates
        box_size: float, the size of the periodic box
        
    Returns:
        numpy array of shape (n_steps, 2) containing the unraveled trajectory
    """
    # Create a copy to avoid modifying the original
    unraveled = trajectory.copy()
    
    # Track jump locations for both x and y coordinates
    jump_pos_x = []
    jump_pos_y = []
    
    # Go through trajectory in pairs
    for i in range(len(trajectory) - 1):
        # Check x coordinate
        if abs(trajectory[i+1, 0] - trajectory[i, 0]) > box_size/2:
            if trajectory[i, 0] > trajectory[i+1, 0]:
                jump_pos_x.append((i, 1))  # Jumped at upper boundary
            else:
                jump_pos_x.append((i, -1))  # Jumped at lower boundary
                
        # Check y coordinate
        if abs(trajectory[i+1, 1] - trajectory[i, 1]) > box_size/2:
            if trajectory[i, 1] > trajectory[i+1, 1]:
                jump_pos_y.append((i, 1))  # Jumped at upper boundary
            else:
                jump_pos_y.append((i, -1))  # Jumped at lower boundary
    
    # Update values for x coordinate
    for i, direction in jump_pos_x:
        unraveled[i+1:, 0] += direction * box_size
        
    # Update values for y coordinate
    for i, direction in jump_pos_y:
        unraveled[i+1:, 1] += direction * box_size
        
    return unraveled


def mark_goal_start_box(ax, box_size, goal_position):
    """Draw the the goal (green star) and the payload start (red dot)."""
    ax.plot(*goal_position, marker='*', color='green', markersize=22,
            markeredgecolor='darkgreen', linestyle='none', zorder=5)
    ax.plot(box_size / 4, box_size / 4, marker='o', color='red', markersize=3.6,
            linestyle='none', zorder=5)
    handles = [
        Line2D([], [], marker='o', color='red', markersize=3.6, linestyle='none', label='payload start'),
        Line2D([], [], marker='*', color='green', markeredgecolor='darkgreen', markersize=14,
               linestyle='none', label='target'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9)
