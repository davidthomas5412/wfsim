import pickle

import numpy as np
import matplotlib.pyplot as plt
import batoid

from wfTel import LSSTFactory

def addAxes(rect, fig, theta=0.0, showFrame=True):
    center = rect[0]+0.5*rect[2]-0.5, rect[1]+0.5*rect[3]-0.5
    if theta != 0.0:
        sth, cth = np.sin(theta), np.cos(theta)
        newCenter = np.dot(
            np.array([[cth, -sth], [sth, cth]]),
            center
        )
        rect[0] = newCenter[0]-0.5*rect[2]+0.5
        rect[1] = newCenter[1]-0.5*rect[3]+0.5

    ax = fig.add_axes(rect)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    if not showFrame:
        ax.axis('off')
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

# chip_rms_height of 4e-6 and chip_rms_tilt of 2e-6 seem about right.


def main(args):
    with open("chips.pkl", 'rb') as f:
        chips = pickle.load(f)

    reference_rng = np.random.RandomState(args.reference_seed)
    if args.chip_rms_height != 0.0:
        for k, chip in chips.items():
            chip['zernikes'] = reference_rng.normal(size=2)
            chip['zernikes'] *= np.array([0, args.chip_rms_height])
    if args.chip_rms_tilt != 0.0:
        for k, chip in chips.items():
            if 'zernikes' not in chip:
                chip['zernikes'] = np.zeros(2)
            chip['zernikes'] = np.concatenate([
                chip['zernikes'],
                reference_rng.normal(size=2)*args.chip_rms_tilt
            ])
    factory = LSSTFactory(args.band, chips=chips)

    M1M3_bend = [0]*20
    for ibend, value in args.M1M3:
        M1M3_bend[int(ibend)] = float(value)

    visit_telescope = factory.make_visit_telescope(
        M2_shift=args.M2_shift,
        M2_tilt=args.M2_tilt,
        camera_shift=args.camera_shift,
        camera_tilt=args.camera_tilt,
        M1M3_bend=M1M3_bend,
        rotation=args.rotation
    )
    # ref has rotation, chip gaps, but no perturbations
    reference_telescope = factory.make_visit_telescope(rotation=args.rotation)

    star_rng = np.random.RandomState(args.star_seed)
    focalRadius = 2.05 # degrees to get to very corner of science array
    th = star_rng.uniform(0, 2*np.pi, size=args.nstar)
    ph = np.sqrt(star_rng.uniform(0, focalRadius**2, size=args.nstar))
    thxs = ph*np.cos(th)  # positions in degrees
    thys = ph*np.sin(th)

    srot, crot = np.sin(args.rotation), np.cos(args.rotation)
    rot = np.array([[crot, -srot], [srot, crot]])

    if args.show_spot:
        fig = plt.figure(figsize=(10, 10))
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
            field_x, field_y = np.dot(
                rot,
                [chip['cam_field_x'], chip['cam_field_y']]
            )

            spotx, spoty = visit_telescope.get_spot(
                field_x, field_y,
                naz=90, nrad=30, reference='chief'
            )
            ax.scatter(spotx, spoty, s=0.1, alpha=0.1)
            ax.set_xlim(-15e-6, 15e-6)
            ax.set_ylim(-15e-6, 15e-6)
        plt.show()

    if args.show_zernike:
        # plots Zernikes measured in CCD frame.
        zs = np.empty((args.nstar, args.jmax+1), dtype=float)
        good = np.ones(len(thxs), dtype=bool)
        for i, (thx, thy) in enumerate(zip(thxs, thys)):
            try:
                zs[i,:] = visit_telescope.get_zernike(
                    np.deg2rad(thx), np.deg2rad(thy),
                    jmax=args.jmax, rings=args.rings, reference='chief'
                )
            except ValueError:
                good[i] = False
        fig = plt.figure(figsize=(13, 8))
        batoid.plotUtils.zernikePyramid(
            thxs[good], thys[good], zs[good,4:].T, s=1, fig=fig
        )
        plt.show()

    if args.show_zernike_resid:
        zs = np.zeros((args.nstar, args.jmax+1), dtype=float)
        good = np.ones(len(thxs), dtype=bool)
        for i, (thx, thy) in enumerate(zip(thxs, thys)):
            try:
                zs[i] = visit_telescope.get_zernike(
                    np.deg2rad(thx), np.deg2rad(thy),
                    jmax=args.jmax, rings=args.rings, reference='chief'
                )
                zs[i] -= reference_telescope.get_zernike(
                    np.deg2rad(thx), np.deg2rad(thy),
                    jmax=args.jmax, rings=args.rings, reference='chief'
                )
            except ValueError:
                good[i] = False
                continue

        fig = plt.figure(figsize=(13, 8))
        batoid.plotUtils.zernikePyramid(
            thxs[good], thys[good], zs[good].T[4:], s=1, fig=fig
        )
        plt.show()

    if args.show_wavefront:
        fig = plt.figure(figsize=(10, 10))
        for k, chip in chips.items():
            rect = np.array([
                chip['left'],
                chip['bottom'],
                chip['right'] - chip['left'],
                chip['top'] - chip['bottom'],
            ])
            rect *= 1.3
            rect += [0.5, 0.5, 0, 0]
            ax = addAxes(rect, fig, theta=args.rotation, showFrame=False)
            field_x, field_y = np.dot(
                rot,
                [chip['cam_field_x'], chip['cam_field_y']]
            )
            wf = visit_telescope.get_wavefront(
                field_x, field_y,
                nx=64
            ).array
            ax.imshow(wf, vmin=-1, vmax=1, cmap='Spectral_r')
        plt.show()

    if args.show_wavefront_resid:
        fig = plt.figure(figsize=(10, 10))
        for k, chip in chips.items():
            rect = np.array([
                chip['left'],
                chip['bottom'],
                chip['right'] - chip['left'],
                chip['top'] - chip['bottom'],
            ])
            rect *= 1.3
            rect += [0.5, 0.5, 0, 0]
            ax = addAxes(rect, fig, theta=args.rotation, showFrame=False)
            field_x, field_y = np.dot(
                rot,
                [chip['cam_field_x'], chip['cam_field_y']]
            )
            wf = visit_telescope.get_wavefront(
                field_x, field_y,
                nx=64
            ).array
            wf -= reference_telescope.get_wavefront(
                field_x, field_y,
                nx=64
            ).array
            ax.imshow(wf, vmin=-0.2, vmax=0.2, cmap='Spectral_r')
        plt.show()

    if args.double_zernike:
        dzs = batoid.analysis.doubleZernike(
            visit_telescope.actual_telescope, np.deg2rad(1.75), 750e-9, rings=10,
            reference='chief', jmax=args.jmax, kmax=args.kmax, eps=0.61
        )
        dzs = dzs[:,4:]
        asort = np.argsort(np.abs(dzs).ravel())[::-1]
        focal_idx, pupil_idx = np.unravel_index(asort[:50], dzs.shape)
        cumsum = 0.0
        print()
        print("fid pid      val      rms")
        for fid, pid in zip(focal_idx, pupil_idx):
            val = dzs[fid, pid]
            cumsum += val**2
            print("{:3d} {:3d} {:8.4f} {:8.4f}".format(fid, pid+4, val, np.sqrt(cumsum)))
        print("sum sqr dz {:8.4f}".format(np.sqrt(np.sum(dzs**2))))

    if args.double_zernike_resid:
        dzs = batoid.analysis.doubleZernike(
            visit_telescope.actual_telescope, np.deg2rad(1.75), 750e-9, rings=10,
            reference='chief', jmax=args.jmax, kmax=args.kmax, eps=0.61
        )
        dzs -= batoid.analysis.doubleZernike(
            reference_telescope.actual_telescope, np.deg2rad(1.75), 750e-9, rings=10,
            reference='chief', jmax=args.jmax, kmax=args.kmax, eps=0.61
        )
        dzs = dzs[:,4:]
        asort = np.argsort(np.abs(dzs).ravel())[::-1]
        focal_idx, pupil_idx = np.unravel_index(asort[:20], dzs.shape)
        cumsum = 0.0
        print()
        print("fid pid      val      rms")
        for fid, pid in zip(focal_idx, pupil_idx):
            val = dzs[fid, pid]
            cumsum += val**2
            print("{:3d} {:3d} {:8.4f} {:8.4f}".format(fid, pid+4, val, np.sqrt(cumsum)))
        print("sum sqr dz {:8.4f}".format(np.sqrt(np.sum(dzs**2))))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--chip_rms_height", default=0.0, type=float
    )
    parser.add_argument(
        "--chip_rms_tilt", default=0.0, type=float
    )
    parser.add_argument(
        "--M2_shift", default=(0.0, 0.0, 0.0), nargs=3, type=float
    )
    parser.add_argument(
        "--M2_tilt", default=(0.0, 0.0), nargs=2, type=float
    )
    parser.add_argument(
        "--camera_shift", default=(0.0, 0.0, 0.0), nargs=3, type=float
    )
    parser.add_argument(
        "--camera_tilt", default=(0.0, 0.0), nargs=2, type=float
    )
    parser.add_argument(
        "--M1M3", action='append', nargs=2, default=list(),
    )
    parser.add_argument(
        "--rotation", default=0.0, type=float
    )
    parser.add_argument("--show_wavefront", action='store_true')
    parser.add_argument("--show_wavefront_resid", action='store_true')
    parser.add_argument("--show_spot", action='store_true')
    parser.add_argument("--show_zernike", action='store_true')
    parser.add_argument("--show_zernike_resid", action='store_true')
    parser.add_argument("--double_zernike", action='store_true')
    parser.add_argument("--double_zernike_resid", action='store_true')
    parser.add_argument(
        "--band", choices=['u', 'g', 'r', 'i', 'z', 'y'], default='i'
    )
    parser.add_argument("--jmax", default=28, type=int)
    parser.add_argument("--kmax", default=28, type=int)
    parser.add_argument("--rings", default=10, type=int)
    parser.add_argument("--nstar", default=1000, type=int)
    parser.add_argument("--star_seed", default=57721, type=int)
    parser.add_argument("--reference_seed", default=5772, type=int)
    args = parser.parse_args()

    main(args)
