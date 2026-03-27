import re
import numpy as np
import matplotlib.pyplot as plt

RESULTS_FILE = "optimization_results_final.txt"

# Parse

generation_steps = {}   # gen -> [steps, ...]
generation_genes = {}   # gen -> {param -> [values, ...]}

with open(RESULTS_FILE) as f:
    content = f.read()

individual_pattern = re.compile(
    r"\[Gen (\d+), Individual \d+/\d+\]\s+"
    r"Gene: max_c=([\d.eE+\-]+), min_c=([\d.eE+\-]+), mid_c=([\d.eE+\-]+), rot_diff=([\d.eE+\-]+)\s+"
    r"Steps taken: (\d+)"
)

for m in individual_pattern.finditer(content):
    gen = int(m.group(1))
    max_c = float(m.group(2))
    min_c = float(m.group(3))
    mid_c = float(m.group(4))
    rot_diff = float(m.group(5))
    steps = int(m.group(6))

    if gen not in generation_steps:
        generation_steps[gen] = []
        generation_genes[gen] = {"max_c": [], "min_c": [], "mid_c": [], "rot_diff": []}

    generation_steps[gen].append(steps)
    generation_genes[gen]["max_c"].append(max_c)
    generation_genes[gen]["min_c"].append(min_c)
    generation_genes[gen]["mid_c"].append(mid_c)
    generation_genes[gen]["rot_diff"].append(rot_diff)

generations = sorted(generation_steps.keys())

# Plot 1

fig, ax = plt.subplots(figsize=(9, 5))

for gen in generations:
    steps = generation_steps[gen]
    ax.scatter([gen] * len(steps), steps, color="steelblue", alpha=0.5, s=30, zorder=2)

means = [np.mean(generation_steps[g]) for g in generations]
ax.plot(generations, means, color="tomato", linewidth=2, marker="o", markersize=5, label="Mean", zorder=3)

ax.set_xlabel("Generation")
ax.set_ylabel("Steps taken")
ax.set_title("Steps taken per generation")
ax.set_xticks(generations)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("stats_steps.png", dpi=150)
plt.close(fig)
print("Saved stats_steps.png")

# Plot 2

params = ["max_c", "min_c", "mid_c", "rot_diff"]
labels = ["max_curvity", "min_curvity", "mid_curvity", "rot_diffusion"]
colors = ["steelblue", "seagreen", "darkorange", "mediumpurple"]

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
axes = axes.flatten()

for i, (param, label, color) in enumerate(zip(params, labels, colors)):
    ax = axes[i]
    for gen in generations:
        vals = generation_genes[gen][param]
        ax.scatter([gen] * len(vals), vals, color=color, alpha=0.5, s=25, zorder=2)

    means_p = [np.mean(generation_genes[g][param]) for g in generations]
    ax.plot(generations, means_p, color="black", linewidth=1.5, marker="o", markersize=4, zorder=3)

    ax.set_title(label)
    ax.set_xticks(generations)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if i >= 2:
        ax.set_xlabel("Generation")

fig.suptitle("Parameter evolution per generation", fontsize=13)
fig.tight_layout()
fig.savefig("stats_params.png", dpi=150)
plt.close(fig)
print("Saved stats_params.png")
