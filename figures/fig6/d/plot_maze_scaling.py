import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DT = 0.01
V0 = 5.0
R  = 1.0

# One list of trial values per maze size.
# Add more entries here as you collect the remaining runs.
raw_data = {
    4: [13911, 16804, 22685, 18771, 14038],
    25:  [49050, 50387, 40305, 29497, 41081], # 34680
    100: [103190, 118197, 113178, 128281, 117503], # 138170
    400: [720429, 503865, 631420, 567655, 392383], # 635820
    900: [2196953, 2222580, 3315073, 3452369, 5826395], # 2645760
    # 1600: [],
}
data = {s: [v * DT * V0 / R for v in vals] for s, vals in raw_data.items()}

sizes = np.array(sorted(data.keys()))
means = np.array([np.mean(data[s]) for s in sizes])
stds  = np.array([np.std(data[s], ddof=1) if len(data[s]) > 1 else 0 for s in sizes])

coeffs = np.polyfit(np.log10(sizes), np.log10(means), 1)
exponent, intercept = coeffs
x_fit = np.logspace(np.log10(sizes[0] * 0.9), np.log10(sizes[-1] * 1.1), 200)
y_fit = 10**intercept * x_fit**exponent

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x_fit, y_fit, '--', color='steelblue', alpha=0.5,
        label=f'$t^*$ ∝ $N^{{{exponent:.2f}}}$')
ax.errorbar(sizes, means, yerr=stds, fmt='o', markersize=8,
            color='steelblue', capsize=4,) # label='mean ± std'

offsets = [(8, 0)] * (len(sizes) - 1) + [(-55, 0)]
for x, y, (dx, dy) in zip(sizes, means, offsets):
    ax.annotate(f'{y:,.0f}', (x, y), textcoords='offset points',
                xytext=(dx, dy), va='center', fontsize=9)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(sizes)
ax.set_xticklabels([str(s) for s in sizes])
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xlabel('Maze Area')
ax.set_ylabel(r'Time to Reach Goal  ($t^* = t\, v_0 / r$)')
ax.set_title(r'Time to Reach Goal vs Maze Area ($t^* = t\, v_0 / r$)')
ax.legend(fontsize=9)
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 1.15)
plt.tight_layout()
plt.savefig('maze_scaling_plot.png', dpi=150)
plt.show()
