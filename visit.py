import pickle

import numpy as np
import matplotlib.pyplot as plt
import batoid

from wfTel import LSSTFactory


norm = {
    "M2 x":            [-0.000166,   0.000166],  # meters
    "M2 y":            [-0.000166,   0.000166],
    "M2 z":            [-1.35e-05,   9.14e-06],
    "M2 thx":          [-2.49e-05,   2.49e-05],  # radians
    "M2 thy":          [-2.49e-05,   2.49e-05],
    "camera x":        [-0.000817,   0.000817],  # meters
    "camera y":        [-0.000817,   0.000817],
    "camera z":        [-1.31e-05,    8.8e-06],
    "camera thx":      [-5.92e-05,   5.92e-05],  # radians
    "camera thy":      [-5.92e-05,   5.92e-05],
    "M1M3 bend 1":     [  -0.0567,     0.0567],  # not sure?
    "M1M3 bend 2":     [  -0.0568,     0.0567],
    "M1M3 bend 3":     [  -0.0555,     0.0792],
    "M1M3 bend 4":     [  -0.0608,     0.0607],
    "M1M3 bend 5":     [    -0.06,       0.06],
    "M1M3 bend 6":     [  -0.0575,     0.0575],
    "M1M3 bend 7":     [  -0.0576,     0.0576],
    "M1M3 bend 8":     [  -0.0636,     0.0636],
    "M1M3 bend 9":     [  -0.0633,     0.0634],
    "M1M3 bend 10":    [  -0.0605,     0.0605],
    "M1M3 bend 11":    [  -0.0605,     0.0611],
    "M1M3 bend 12":    [   -0.171,       0.12],
    "M1M3 bend 13":    [  -0.0657,     0.0658],
    "M1M3 bend 14":    [  -0.0659,     0.0659],
    "M1M3 bend 15":    [   -0.101,      0.101],
    "M1M3 bend 16":    [   -0.101,      0.101],
    "M1M3 bend 17":    [  -0.0587,     0.0587],
    "M1M3 bend 18":    [  -0.0598,     0.0598],
    "M1M3 bend 19":    [  -0.0695,     0.0696],
    "M1M3 bend 20":    [  -0.0696,     0.0699]
}

def visit(args):
    if args.amplitude is not None:
        args.M2_amplitude = args.amplitude
        args.camera_amplitude = args.amplitude
        args.M1M3_amplitude = args.amplitude

    with open(args.reference_file, 'rb') as fd:
        reference = pickle.load(fd)

    factory = LSSTFactory(reference['args'].band, chips=reference['chips'])
    visit_rng = np.random.RandomState(args.visit_seed)

    if args.M2_amplitude != 0.0:
        M2_dx = visit_rng.normal()
        M2_dx *= np.ptp(norm['M2 x'])/2
        M2_dx += np.mean(norm['M2 x'])
        M2_dy = visit_rng.normal()
        M2_dy *= np.ptp(norm['M2 y'])/2
        M2_dy += np.mean(norm['M2 y'])
        M2_dz = visit_rng.normal()
        M2_dz *= np.ptp(norm['M2 z'])/2
        M2_dz += np.mean(norm['M2 z'])
        M2_shift = np.array([M2_dx, M2_dy, M2_dz])
        M2_shift *= args.M2_amplitude

        M2_thx = visit_rng.normal()
        M2_thx *= np.ptp(norm['M2 thx'])/2
        M2_thx += np.mean(norm['M2 thx'])
        M2_thy = visit_rng.normal()
        M2_thy *= np.ptp(norm['M2 thy'])/2
        M2_thy += np.mean(norm['M2 thy'])
        M2_tilt = np.array([M2_thx, M2_thy])
        M2_tilt *= args.M2_amplitude
    else:
        M2_shift = (0, 0, 0)
        M2_tilt = (0, 0)

    if args.camera_amplitude != 0.0:
        camera_dx = visit_rng.normal()
        camera_dx *= np.ptp(norm['camera x'])/2
        camera_dx += np.mean(norm['camera x'])
        camera_dy = visit_rng.normal()
        camera_dy *= np.ptp(norm['camera y'])/2
        camera_dy += np.mean(norm['camera y'])
        camera_dz = visit_rng.normal()
        camera_dz *= np.ptp(norm['camera z'])/2
        camera_dz += np.mean(norm['camera z'])
        camera_shift = np.array([camera_dx, camera_dy, camera_dz])
        camera_shift *= args.camera_amplitude

        camera_thx = visit_rng.normal()
        camera_thx *= np.ptp(norm['camera thx'])/2
        camera_thx += np.mean(norm['camera thx'])
        camera_thy = visit_rng.normal()
        camera_thy *= np.ptp(norm['camera thy'])/2
        camera_thy += np.mean(norm['camera thy'])
        camera_tilt = np.array([camera_thx, camera_thy])
        camera_tilt *= args.camera_amplitude
    else:
        camera_shift = (0, 0, 0)
        camera_tilt = (0, 0)

    if args.M1M3_through10_amplitude != 0.0:
        M1M3_bend = []
        for i in range(1, 11):
            val = visit_rng.normal()
            val *= np.ptp(norm[f'M1M3 bend {i}'])/2
            val += np.mean(norm[f'M1M3 bend {i}'])
            M1M3_bend.append(val)
        for i in range(11, 21):
            M1M3_bend.append(0.0)
        M1M3_bend = np.array(M1M3_bend)*args.M1M3_amplitude
    else:
        M1M3_bend = np.zeros(20)

    if args.M1M3_amplitude != 0.0:
        M1M3_bend = []
        for i in range(1, 21):
            val = visit_rng.normal()
            val *= np.ptp(norm[f'M1M3 bend {i}'])/2
            val += np.mean(norm[f'M1M3 bend {i}'])
            M1M3_bend.append(val)
        M1M3_bend = np.array(M1M3_bend)*args.M1M3_amplitude
    else:
        M1M3_bend = np.zeros(20)

    if args.rotation is not None:
        rotation = args.rotation
    elif args.rot_seed is not None:
        rot_rng = np.random.RandomState(args.rot_seed)
        rotation = rot_rng.uniform(-np.pi/2, np.pi/2)
    else:
        rotation = 0.0

    visit_telescope = factory.make_visit_telescope(
        M2_shift=M2_shift,
        M2_tilt=M2_tilt,
        camera_shift=camera_shift,
        camera_tilt=camera_tilt,
        M1M3_bend=M1M3_bend,
        rotation=rotation
    )

    # Either use same star xs and ys as reference, or if star_seed is set,
    # generate new stars.
    if args.star_seed is not None:
        star_rng = np.random.RandomState(args.star_seed)
        focalRadius = 1.825  # degrees
        th = star_rng.uniform(0, 2*np.pi, size=args.nstar)
        ph = np.sqrt(star_rng.uniform(0, focalRadius**2, size=args.nstar))
        thxs = ph*np.cos(th)  # positions in degrees
        thys = ph*np.sin(th)
        jmax = reference['args'].jmax
        zs = np.zeros((args.nstar, jmax+1), dtype=float)
        ccds = np.zeros(args.nstar, dtype='<U7')
        ccdxs = np.zeros_like(thxs)
        ccdys = np.zeros_like(thxs)
        good = np.ones(len(thxs), dtype=bool)
    else:
        thxs = reference['thxs']
        thys = reference['thys']
        zs = np.zeros_like(reference['zs'])
        ccds = reference['ccds']
        ccdxs = reference['ccdxs']
        ccdys = reference['ccdys']
        good = np.ones_like(xs, dtype=bool)

    for i, (thx, thy) in enumerate(zip(thxs, thys)):
        try:
            ccds[i] = visit_telescope.get_chip(np.deg2rad(thx), np.deg2rad(thy))
            zs[i] = visit_telescope.get_zernike(
                np.deg2rad(thx), np.deg2rad(thy),
                jmax=reference['args'].jmax, rings=10, reference='chief'
            )
            ccdxs[i], ccdys[i] = visit_telescope.get_fp(
                np.deg2rad(thx), np.deg2rad(thy),
                type='chip'
            )
        except ValueError:
            good[i] = False
            continue

    if args.plot:
        fig = plt.figure(figsize=(13, 8))
        batoid.plotUtils.zernikePyramid(
            thxs[good], thys[good], zs[good].T[4:], s=1, fig=fig
        )
        plt.show()

    if args.plot_resid:
        fig = plt.figure(figsize=(13, 8))
        batoid.plotUtils.zernikePyramid(
            thxs[good], thys[good],
            (zs-reference['zs'])[good].T[4:], s=1, fig=fig,
            vmin=-0.2, vmax=0.2
        )
        plt.show()

    if args.visit_file is not None:
        with open(args.visit_file, 'wb') as fd:
            pickle.dump(
                dict(
                    thxs=thxs,
                    thys=thys,
                    zs=zs,
                    ccds=ccds,
                    ccdxs=ccdxs,
                    ccdys=ccdys,
                    good=good,
                    M2_shift=M2_shift,
                    M2_tilt=M2_tilt,
                    camera_shift=camera_shift,
                    camera_tilt=camera_tilt,
                    M1M3_bend=M1M3_bend,
                    rotation=rotation,
                    args=args
                ),
                fd
            )


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--reference_file", default='reference.pkl', type=str)
    parser.add_argument("--M2_amplitude", default=0.0, type=float)
    parser.add_argument("--camera_amplitude", default=0.0, type=float)
    parser.add_argument("--M1M3_through10_amplitude", default=0.0, type=float)
    parser.add_argument("--M1M3_amplitude", default=0.0, type=float)
    parser.add_argument("--amplitude", default=None, type=float)
    parser.add_argument("--visit_seed", default=0, type=int)
    parser.add_argument("--visit_file", default=None, type=str)
    parser.add_argument("--nstar", default=10000, type=int)
    parser.add_argument("--star_seed", default=None, type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rotation", default=None, type=float)
    group.add_argument("--rot_seed", default=None, type=int)
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--plot_resid", action='store_true')
    args = parser.parse_args()

    visit(args)
