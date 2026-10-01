import numpy as np

PATH = r"D:/PostThesis/data/snell_4000_20_1m_short100.npz"
THRESHOLD = 25

data = np.load(PATH, allow_pickle=True)

payload_pos = data["payload_positions"][-1]  # (2,)
goal_pos = data["goal_position"]  # (2,)

dist = np.linalg.norm(payload_pos - goal_pos)
reached = dist <= THRESHOLD

print(f"Payload final position : {payload_pos}")
print(f"Goal position          : {goal_pos}")
print(f"Distance to goal       : {dist:.2f}")
print(f"Goal reached (≤{THRESHOLD})    : {reached}")
