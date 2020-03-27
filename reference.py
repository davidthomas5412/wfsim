import pickle

import numpy as np
import matplotlib.pyplot as plt
import batoid

from wfTel import LSSTFactory


def reference(args):
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
    reference_telescope = factory.make_visit_telescope()  # has chip gaps

    star_rng = np.random.RandomState(args.star_seed)
    focalRadius = 1.825  # degrees
    th = star_rng.uniform(0, 2*np.pi, size=args.nstar)
    ph = np.sqrt(star_rng.uniform(0, focalRadius**2, size=args.nstar))
    thxs = ph*np.cos(th)  # positions in degrees
    thys = ph*np.sin(th)

    zs = np.zeros((args.nstar, args.jmax+1), dtype=float)
    ccds = np.zeros(args.nstar, dtype='<U7')
    ccdxs = np.zeros_like(thxs)
    ccdys = np.zeros_like(thys)
    good = np.ones(len(thxs), dtype=bool)
    for i, (thx, thy) in enumerate(zip(thxs, thys)):
        try:
            ccds[i] = reference_telescope.get_chip(
                np.deg2rad(thx), np.deg2rad(thy)
            )
            zs[i] = reference_telescope.get_zernike(
                np.deg2rad(thx), np.deg2rad(thy),
                jmax=args.jmax, rings=10, reference='chief'
            )
            ccdxs[i], ccdys[i] = reference_telescope.get_fp(
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

    with open(args.reference_file, 'wb') as fd:
        pickle.dump(
            dict(
                thxs=thxs,
                thys=thys,
                ccdxs=ccdxs,
                ccdys=ccdys,
                zs=zs,
                ccds=ccds,
                good=good,
                chips=chips,
                args=args
            ),
            fd
        )


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
        "--band", choices=['u', 'g', 'r', 'i', 'z', 'y'], default='i'
    )
    parser.add_argument("--jmax", default=28, type=int)
    parser.add_argument("--nstar", default=10000, type=int)
    parser.add_argument("--star_seed", default=57721, type=int)
    parser.add_argument("--reference_seed", default=5772, type=int)
    parser.add_argument("--reference_file", default='reference.pkl', type=str)
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()

    reference(args)
