import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
DT = 0.01   # simulation timestep
V0 = 5.0    # particle self-propulsion speed
R  = 1.0    # particle radius  (used for t* normalisation)

# ── Solution path lengths (Wilson maze, seed=42, grid-graph shortest path) ────
# L = (path steps) × cell_size,  cell_size = 60 sim-units for all mazes
#   G2 :  2 steps × 60 =  120  (no shortcut possible for this maze/seed)
#   G5 :  8 steps × 60 =  480  (no shortcut possible for this maze/seed)
#   G10: 22 steps × 60 = 1320  (perfect-maze path; shortcut path = 18 steps = 1080)
# These are grid-graph distances (90° turns through cell centres), so they are
# slight over-estimates of the true continuous-space minimum path length.
SOLUTION_PATH_LENGTH_G2  = 120
SOLUTION_PATH_LENGTH_G5  = 480
SOLUTION_PATH_LENGTH_G10 = 1320
SOLUTION_PATH_LENGTH_G20 = 3480

# ── Raw data: timestep counts until payload reaches goal (6 trials each) ──────

# G10: 10×10 maze,  BOX_SIZE=600, cell_size=60
#      standard density = 10 particles/cell → 1000 particles
raw_data_G10 = {
    250:  [5519793, 3381306, 3605883, 2833803, 3633287, 5657263],
    375:  [1880131, 3576550, 1587558, 1369109, 3058293, 4506999],
    500:  [ 566790,  918203, 1244020,  687763, 1208222, 2043445],
    750:  [ 242971,  222693,  153045,  297731,  152479,  180043],
    1000: [ 100820,  122467,   78636,  203445,  129445,  123624],
    1500: [  75609,   72991,   81809,   81524,   83502,   78288],
    2000: [  71130,   77406,   73567,   72289,   76382,   69359],
    3000: [  62350,   62126,   59198,   57977,   56475,   58962],
    4000: [  57240,   54731,   58528,   57270,   52122,   59228],
    6000: [  50595,   49309,   50497,   51422,   50632,   52898],
    8000: [  44453,   46173,   49381,   44456,   47405,   47027],
    12000:[  43123,   41675,   40352,   40783,   40728,   42436],
    16000:[  38974,   37039,   39203,   40113,   38399,   38790],
}

# G2: 2×2 maze,  BOX_SIZE=120, cell_size=60
#     standard density = 10 particles/cell → 40 particles
raw_data_G2 = {
    10:  [118992,  77210,  92313,  84731, 140393,  99544],
    15:  [ 59616,  46500,  67184,  60472,  34476,  93497],
    20:  [ 63351,  26251,  54660,  88575,  65210,  45998],
    30:  [ 22464,  23001,  18587,  28130,  22531,  13803],
    40:  [ 19756,  14725,  13085,  19380,  14158,  18226],
    60:  [ 10403,  10915,  12587,  14767,   9736,  10139],
    80:  [ 10855,  10349,  13091,  11591,   9803,  12398],
    120: [  7949,   8129,   8807,   8906,   7526,   8399],
    160: [  8042,   8164,   6505,   8328,   6184,   7185],
    240: [  6354,   5735,   6800,   5705,   6658,   6636],
    320: [  5283,   5282,   5216,   5688,   5617,   5326],
    480: [  4977,   5031,   4865,   4650,   4778,   4828],
    640: [  4624,   4925,   4442,   4634,   4342,   4602],
}

# G5: 5×5 maze,  BOX_SIZE=300, cell_size=60
#     standard density = 10 particles/cell → 250 particles
raw_data_G5 = {
    62:   [142728, 159452, 150018, 254972, 220608, 152005],
    93:   [ 68686, 124235, 182495,  99374,  91575,  86389],
    125:  [ 77762,  53795,  61093,  60040,  55896,  95943],
    187:  [ 39898,  64237,  57284,  35696,  57821,  40557],
    250:  [ 40878,  42566,  46660,  30874,  36512,  51368],
    375:  [ 30423,  28819,  29273,  32785,  41627,  32511],
    500:  [ 29266,  23940,  23880,  30282,  23505,  24194],
    750:  [ 19078,  20546,  19060,  28372,  24378,  21583],
    1000: [ 17071,  17681,  21426,  20531,  17910,  17794],
    1500: [ 14951,  14981,  17583,  17255,  15662,  14964],
    2000: [ 14537,  14438,  14980,  14198,  16772,  15617],
    3000: [ 13819,  13588,  14663,  14854,  13917,  13758],
    4000: [ 12900,  13522,  13267,  13643,  13549,  13238],
}

raw_data_G20 = {
    1000: [],
    1500: [],
    2000: [19541395, 92618792, 27770609, ],
    3000: [4657624, 3183999, 2799642, 1955401, 3753049],
    4000: [1018780, 573265, 479694, 610135, 511359, 456902],
    6000: [280473, 230620, 250236, 225484, 282068, 265540],
    8000: [201029, 222876, 193153, 208037, 197507, 235316],
    12000: [176475, 175202, 185155, 187104, 183561, 169831],
    16000: [175821, 150102, 168580, 157257, 161462, 162494],
    24000: [142597, 127971, 137943, 144042, 145535, 143657],
    32000: [128591, 130097, 126913, 132098, 130487, 128929],
    48000: [111909, 116625, 119237, 119601, 116896, 117021],
    64000: [106549, 109376, 112956, 109524, 111136, 109107],
}

# ── Dataset registry ──────────────────────────────────────────────────────────
# ylim: (lo, hi) for the individual t* plot, or None for auto
DATASETS = [
    dict(
        raw       = raw_data_G2,
        grid_size = 2,
        L         = SOLUTION_PATH_LENGTH_G2,
        color     = '#FFA07A',
        label     = '2×2 maze',
        outfile   = 'swarm_scaling_plot_G2.png',
        title     = ('Time to Reach Goal vs Swarm Size\n'
                     r'($t^* = t\,v_0/r$,  2×2 maze, seed=42, 6 trials each)'),
        ylim      = (1e2, 1e4),
    ),
    dict(
        raw       = raw_data_G5,
        grid_size = 5,
        L         = SOLUTION_PATH_LENGTH_G5,
        color     = '#66CDAA',
        label     = '5×5 maze',
        outfile   = 'swarm_scaling_plot_G5.png',
        title     = ('Time to Reach Goal vs Swarm Size\n'
                     r'($t^* = t\,v_0/r$,  5×5 maze, seed=42, 6 trials each)'),
        ylim      = None,   # auto: data spans ~600–13 000 in t*
    ),
    dict(
        raw       = raw_data_G10,
        grid_size = 10,
        L         = SOLUTION_PATH_LENGTH_G10,
        color     = 'steelblue',
        label     = '10×10 maze',
        outfile   = 'swarm_scaling_plot_G10.png',
        title     = ('Time to Reach Goal vs Swarm Size\n'
                     r'($t^* = t\,v_0/r$,  10×10 maze, seed=42, 6 trials each)'),
        ylim      = (1e3, 1e6),
    ),
        dict(
        raw       = raw_data_G20,
        grid_size = 20,
        L         = SOLUTION_PATH_LENGTH_G20,
        color     = '#7B68EE',
        label     = '20×20 maze',
        outfile   = 'swarm_scaling_plot_G20.png',
        title     = ('Time to Reach Goal vs Swarm Size\n'
                     r'($t^* = t\,v_0/r$,  20×20 maze, seed=42, 6 trials each)'),
        ylim      = None #(1e3, 1e6),
    ),
]


# ── Helper: raw dict → (sizes, means_t*, stds_t*) ────────────────────────────
def process(raw):
    """Convert raw timestep counts to dimensionless t* = steps × DT × V0 / R."""
    data  = {s: [v * DT * V0 / R for v in vals] for s, vals in raw.items()}
    sizes = np.array(sorted(data.keys()))
    means = np.array([np.mean(data[s]) for s in sizes])
    stds  = np.array([np.std(data[s], ddof=1) if len(data[s]) > 1 else 0
                      for s in sizes])
    return sizes, means, stds


# ── Individual plot: t* vs absolute swarm size ────────────────────────────────
def plot_individual(ds):
    sizes, means, stds = process(ds['raw'])
    mask = means > 0
    sx, sy, se = sizes[mask], means[mask], stds[mask]

    t_star_opt = ds['L'] / R   # theoretical optimum in t* units = L/r

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(sx, sy, yerr=se, fmt='o', markersize=8,
                color=ds['color'], capsize=4)

    for x, y in zip(sx, sy):
        ax.annotate(f'{y:,.0f}', (x, y), textcoords='offset points',
                    xytext=(8, 0), va='center', fontsize=9)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(sizes[0] * 0.7, sizes[-1] * 1.4)
    if ds['ylim'] is not None:
        ax.set_ylim(*ds['ylim'])
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.axhline(t_star_opt, color=ds['color'], linestyle=':', linewidth=1.5,
               label=f'theoretical optimum ($L/r = {t_star_opt:.0f}$)')

    ax.set_xlabel('Swarm Size (number of particles)')
    ax.set_ylabel(r'Time to Reach Goal  ($t^* = t\, v_0 / r$)')
    ax.set_title(ds['title'])
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(ds['outfile'], dpi=150)
    plt.show()
    print(f"Saved {ds['outfile']}")


# ── Combined normalised plot: t** vs particle density ────────────────────────
# X-axis: density = N / G²  (particles per cell)
# Y-axis: t** = t* / (L/r) = t·v₀/L  (overhead factor; t**=1 is theoretical optimum)
# All three mazes span roughly the same density range (~2.5–160 particles/cell).
def plot_combined():
    fig, ax = plt.subplots(figsize=(6, 6))

    for ds in DATASETS:
        sizes, means, stds = process(ds['raw'])
        mask = means > 0
        sx, sy, se = sizes[mask], means[mask], stds[mask]

        n_cells  = ds['grid_size'] ** 2
        density  = sx / n_cells          # particles per cell  (x-axis)
        t2_mean  = sy / (ds['L'] / R)   # t** = t* / (L/r)   (y-axis)
        t2_err   = se / (ds['L'] / R)

        ax.errorbar(density, t2_mean, yerr=t2_err,
                    fmt='o', markersize=7, capsize=4,
                    color=ds['color'], label=ds['label'])

    # Theoretical optimum: t** = 1
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5,
               label=r'theoretical optimum ($t^{**} = 1$)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Particle Density (particles per cell)')
    ax.set_ylabel(r'Normalised Time  ($t^{**} = t\, v_0 / L$)')
    ax.set_title('Swarm Size Scaling across Maze Sizes\n'
                 r'$t^{**} = t\,v_0/L$;  density $= N/G^2$;  optimum at $t^{**}=1$')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    plt.tight_layout()
    plt.savefig('swarm_scaling_plot_combined_g20.png', dpi=150)
    plt.show()
    print("Saved swarm_scaling_plot_combined_g20.png")


# ── Run ───────────────────────────────────────────────────────────────────────
for ds in DATASETS:
    plot_individual(ds)

plot_combined()
