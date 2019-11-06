import pickle
import numpy as np
import matplotlib.pyplot as plt

from wfTel import LSSTFactory

def addAxes(rect, fig):
    ax = fig.add_axes(rect)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

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
    ["M1M3 bend 1",       -0.0567,     0.0567],  # not sure?
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

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--chipRMSHeight", default=0.0, type=float)
    parser.add_argument("--chipRMSTilt", default=0.0, type=float)
    parser.add_argument("--M2_shift", default=(0.0, 0.0, 0.0), nargs=3, type=float)
    parser.add_argument("--M2_tilt", default=(0.0, 0.0), nargs=2, type=float)
    parser.add_argument("--camera_shift", default=(0.0, 0.0, 0.0), nargs=3, type=float)
    parser.add_argument("--camera_tilt", default=(0.0, 0.0), nargs=2, type=float)
    parser.add_argument("--showWF", action='store_true')
    parser.add_argument("--showSpots", action='store_true')
    parser.add_argument("--showZernikes", action='store_true')
    parser.add_argument("--normalized", action='store_true')
    parser.add_argument("--band", choices=['u', 'g', 'r', 'i', 'z', 'y'], default='i')
    args = parser.parse_args()

    with open("chips.pkl", 'rb') as f:
        chips = pickle.load(f)

    if args.chipRMSHeight != 0.0:
        for k, v in chips.items():
            v['zernikes'] = np.random.normal(size=2)*np.array([0, args.chipRMSHeight])
    if args.chipRMSTilt != 0.0:
        for k, v in chips.items():
            if 'zernikes' in v:
                v['zernikes'] = np.concatenate([
                    v['zernikes'],
                    np.random.normal(size=2)*args.chipRMSTilt
                ])
            else:
                v['zernikes'] = np.random.normal(size=4)*args.chipRMSTilt
                v['zernikes'] *= [0,0,1,1]
    factory = LSSTFactory(args.band, chips=chips)
    visit_telescope = factory.make_visit_telescope(
        M2_shift = args.M2_shift,
        M2_tilt = args.M2_tilt,
        camera_shift = args.camera_shift,
        camera_tilt = args.camera_tilt,
    )

    if args.showSpots:
        fig = plt.figure(figsize=(10, 10))
        lefts = []
        for k, chip in chips.items():
            rect = np.array([
                chip['left'],
                chip['bottom'],
                chip['right'] - chip['left'],
                chip['top'] - chip['bottom'],
            ])
            rect *= 1.3
            rect += [0.5, 0.5, 0, 0]
            ax = addAxes(rect, fig)
            spotx, spoty = visit_telescope.get_spot(
                np.deg2rad(chip['x']/0.3*1.75),
                np.deg2rad(chip['y']/0.3*1.75),
                naz=90, nrad=30
            )
            ax.scatter(spotx, spoty, s=0.1, alpha=0.1)
            ax.set_xlim(-15e-6, 15e-6)
            ax.set_ylim(-15e-6, 15e-6)
            lefts.append(chip['left'])
        plt.show()
