import pickle
import matplotlib.pyplot as plt
import numpy as np

norm = [
    ["M2 x",            -0.000166,   0.000166],  # meters
    ["M2 y",            -0.000166,   0.000166],
    ["M2 z",            -1.35e-05,   9.14e-06],
    ["M2 thx",          -2.49e-05,   2.49e-05],  # radians
    ["M2 thy",          -2.49e-05,   2.49e-05],
    ["camera x",        -0.000817,   0.000817],  # meters
    ["camera y",        -0.000817,   0.000817],
    ["camera z",        -1.31e-05,    8.8e-06],
    ["camera thx",      -5.92e-05,   5.92e-05],  # radians
    ["camera thy",      -5.92e-05,   5.92e-05],
    ["M1M3 bend 1",       -0.0567,     0.0567],  # units unclear...
    ["M1M3 bend 2",       -0.0568,     0.0567],
    ["M1M3 bend 3",       -0.0555,     0.0792],
    ["M1M3 bend 4",       -0.0608,     0.0607],
    ["M1M3 bend 5",         -0.06,       0.06],
    ["M1M3 bend 6",       -0.0575,     0.0575],
    ["M1M3 bend 7",       -0.0576,     0.0576],
    ["M1M3 bend 8",       -0.0636,     0.0636],
    ["M1M3 bend 9",       -0.0633,     0.0634],
    ["M1M3 bend 10",      -0.0605,     0.0605],
    ["M1M3 bend 11",      -0.0605,     0.0611],
    ["M1M3 bend 12",       -0.171,       0.12],
    ["M1M3 bend 13",      -0.0657,     0.0658],
    ["M1M3 bend 14",      -0.0659,     0.0659],
    ["M1M3 bend 15",       -0.101,      0.101],
    ["M1M3 bend 16",       -0.101,      0.101],
    ["M1M3 bend 17",      -0.0587,     0.0587],
    ["M1M3 bend 18",      -0.0598,     0.0598],
    ["M1M3 bend 19",      -0.0695,     0.0696],
    ["M1M3 bend 20",      -0.0696,     0.0699]
]


from wfTel import LSSTFactory
with open("chips.pkl", 'rb') as f:
    chips = pickle.load(f)
factory = LSSTFactory('i', chips=chips)
reference_telescope = factory.make_visit_telescope()

xs = np.linspace(-1.75, 1.75, 15)
xs, ys = np.meshgrid(xs, xs)
w = np.hypot(xs, ys) <= 1.75
xs = xs[w]
ys = ys[w]

fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(10, 10))
axes = axes.ravel()

for j, (title, minus, plus) in enumerate(norm):
    dof = [0]*30
    dof[j] = plus
    visit_telescope = factory.make_visit_telescope(dof=dof)

    good = np.ones(len(xs), dtype=bool)
    fpxy = np.zeros((len(xs), 2), dtype=float)
    for i, (x, y) in enumerate(zip(xs, ys)):
        try:
            fpxy[i] = visit_telescope.get_fp(
                np.deg2rad(x), np.deg2rad(y), type='spot'
            )
            fpxy[i] -= reference_telescope.get_fp(
                np.deg2rad(x), np.deg2rad(y), type='spot'
            )
        except ValueError:
            good[i] = False

    Q = axes[j].quiver(
        xs[good],
        ys[good],
        (fpxy[good,0]-np.mean(fpxy[good,0]))/10e-6,
        (fpxy[good,1]-np.mean(fpxy[good,1]))/10e-6,
        scale=0.5
    )
    if j == 0:
        axes[j].quiverkey(
            Q,
            0.5, 0.95,
            0.1, "0.1 arcsec", labelpos='E', coordinates='figure'
        )

    axes[j].set_title(title)
    axes[j].set_aspect('equal')
    axes[j].get_xaxis().set_ticks([])
    axes[j].get_yaxis().set_ticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Distortion modes.png")
plt.show()
