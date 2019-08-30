import os

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplots

import batoid
import galsim


def Zpyramid(xs, ys, zs, figsize=(13, 8), vmin=-1, vmax=1, vdim=True,
             s=5, title=None, filename=None, fontsize=7, **kwargs):
    jmax = zs.shape[0]+3
    nmax, _ = galsim.zernike.noll_to_zern(jmax)

    nrow = nmax - 1
    ncol = nrow + 2
    gridspec = GridSpec(nrow, ncol)

    def shift(pos, amt):
        return [pos.x0+amt, pos.y0, pos.width, pos.height]

    def shiftAxes(axes, amt):
        for ax in axes:
            ax.set_position(shift(ax.get_position(), amt))

    fig = plt.figure(figsize=figsize, **kwargs)
    axes = {}
    shiftLeft = []
    shiftRight = []
    for j in range(4, jmax+1):
        n, m = galsim.zernike.noll_to_zern(j)
        if n%2 == 0:
            row, col = n-2, m//2 + ncol//2
        else:
            row, col = n-2, (m-1)//2 + ncol//2
        subplotspec = gridspec.new_subplotspec((row, col))
        axes[j] = fig.add_subplot(subplotspec)
        axes[j].set_aspect('equal')
        if nrow%2==0 and n%2==0:
            shiftLeft.append(axes[j])
        if nrow%2==1 and n%2==1:
            shiftRight.append(axes[j])

    cbar = {}
    for j, ax in axes.items():
        n, _ = galsim.zernike.noll_to_zern(j)
        ax.set_title("Z{}".format(j), fontsize=fontsize)
        if vdim:
            _vmin = vmin/n
            _vmax = vmax/n
        else:
            _vmin = vmin
            _vmax = vmax
        scat = ax.scatter(xs, ys, c=zs[j-4], s=s, linewidths=0.5, cmap='Spectral_r',
                          rasterized=True, vmin=_vmin, vmax=_vmax)
        cbar[j] = fig.colorbar(scat, ax=ax)
        cbar[j].ax.tick_params(labelsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, x=0.1)

    fig.tight_layout()
    amt = 0.5*(axes[4].get_position().x0 - axes[5].get_position().x0)
    shiftAxes(shiftLeft, -amt)
    shiftAxes(shiftRight, amt)

    shiftAxes([cbar[j].ax for j in cbar.keys() if axes[j] in shiftLeft], -amt)
    shiftAxes([cbar[j].ax for j in cbar.keys() if axes[j] in shiftRight], amt)

    if filename:
        fig.savefig(filename)

    fig.show()


def rot(thx, thy):
    return np.dot(batoid.RotX(thx), batoid.RotY(thy))


def getChips(chipAmp, fixedRNG):
    camRad = 0.315
    chipXGap = 0.001
    # Need to fit in 15 CCDs and 14 gaps
    chipWidth = (2*camRad - 14*chipXGap)/15
    # Fill square
    leftEdges = -camRad + np.arange(15)*(chipWidth+chipXGap)
    rightEdges = leftEdges + chipWidth
    bottomEdges = leftEdges
    topEdges = rightEdges
    # Now arbitrarily number the edges
    chips = {}
    k = 1
    for i in range(15):
        for j in range(15):
            if i < 3 or i > 11:
                if j < 3 or j > 11:
                    continue
            # Randomly generate perturbed chip surface.
            surface = batoid.Zernike([0]+1e-5*chipAmp*fixedRNG.uniform(-0.5, 0.5, size=3))

            chips[k] = dict(
                leftEdge = leftEdges[i],
                rightEdge = rightEdges[i],
                bottomEdge = bottomEdges[j],
                topEdge = topEdges[j],
                center = (0.5*(leftEdges[i]+rightEdges[i]), 0.5*(topEdges[j]+bottomEdges[j]), 0.0),
                surface = surface
            )
            k += 1

    return chips


def getChip(x, y, telescope, chips):
    dirCos = batoid.utils.fieldToDirCos(np.deg2rad(x), np.deg2rad(y))
    dirCos = dirCos[0:2] + (-dirCos[2],)
    ray = batoid.Ray.fromPupil(
        np.deg2rad(x), np.deg2rad(y),
        telescope.dist, 625e-9,
        dirCos=dirCos, medium=telescope.inMedium,
        interface=telescope.entrancePupil
    )
    telescope.traceInPlace(ray)
    for k, chip in chips.items():
        if (    ray.x > chip['leftEdge'] and
                ray.x < chip['rightEdge'] and
                ray.y > chip['bottomEdge'] and
                ray.y < chip['topEdge']):
            return k
    raise ValueError("Can't find chip")


def main(args):
    visitRNG = np.random.RandomState(args.visitSeed)
    fixedRNG = np.random.RandomState(args.fixedSeed)
    starRNG = np.random.RandomState(args.starSeed)
    # Start with circular field of view
    focalRadius = 1.75 # degrees
    th = starRNG.uniform(0, 2*np.pi, size=args.nstar)
    ph = np.sqrt(starRNG.uniform(0, focalRadius**2, size=args.nstar))
    xs = ph*np.cos(th) # positions in degrees
    ys = ph*np.sin(th)

    dof = 0
    if args.M2Rigid != 0.0:
        dof += 5
    if args.cameraRigid != 0.0:
        dof += 5
    if args.M1M3Bend != 0.0:
        dof += 20
    if args.M2Bend != 0.0:
        dof += 20

    LSST_r_fn = os.path.join(batoid.datadir, "LSST", "LSST_r.yaml")
    config = yaml.safe_load(open(LSST_r_fn))
    fiducial_telescope = batoid.parse.parse_optic(config['opticalSystem'])
    config = yaml.safe_load(open(LSST_r_fn))
    telescope = batoid.parse.parse_optic(config['opticalSystem'])
    # Rigid body perturbations
    # These are rough values to introduce shifts of ~1 wave Z4 across focal plane
    if args.M2Rigid != 0.0:
        M2Dx = 0.6e-3
        M2Dz = 0.040e-3 * 0.3
        M2Tilt = np.deg2rad(0.25/60)
        M2Shift = args.M2Rigid*visitRNG.normal(scale=(M2Dx, M2Dx, M2Dz))
        M2Tilt = args.M2Rigid*visitRNG.normal(scale=(M2Tilt, M2Tilt))
        # Shrink amplitudes by sqrt(dof) and amplify by overall amplitude factor.
        M2Shift *= args.amplitude/np.sqrt(dof)
        M2Tilt *= args.amplitude/np.sqrt(dof)
        print(f"M2Shift = {M2Shift/1e-6} microns")
        print(f"M2Tilt = {np.rad2deg(M2Tilt)*60} arcmin")
        telescope = (telescope
                     .withGloballyShiftedOptic("LSST.M2", M2Shift)
                     .withLocallyRotatedOptic("LSST.M2", rot(*M2Tilt))
                    )
    if args.cameraRigid != 0.0:
        cameraDx = 2.99e-3
        cameraDz = 0.039e-3 * 0.3
        cameraTilt = np.deg2rad(0.75/60) * 0.3
        cameraShift = args.cameraRigid*visitRNG.normal(scale=(cameraDx, cameraDx, cameraDz))
        cameraTilt = args.cameraRigid*visitRNG.normal(scale=(cameraTilt, cameraTilt))
        # Shrink amplitudes by sqrt(dof) and amplify by overall amplitude factor.
        cameraShift *= args.amplitude/np.sqrt(dof)
        cameraTilt *= args.amplitude/np.sqrt(dof)
        print(f"cameraShift = {cameraShift/1e-6} microns")
        print(f"cameraTilt = {np.rad2deg(cameraTilt)*60} arcmin")
        telescope = (telescope
                     .withGloballyShiftedOptic("LSST.LSSTCamera", cameraShift)
                     .withLocallyRotatedOptic("LSST.LSSTCamera", rot(*cameraTilt))
                    )

    # Figure.  Use Zernikes...
    if args.mirrorFigure != 0.0:
        coefM1 = [0]*22 + (1e-8*args.mirrorFigure*fixedRNG.uniform(-0.5, 0.5, size=33)).tolist()
        M1surface = batoid.Sum([
            telescope.itemDict['LSST.M1'].surface,
            batoid.Zernike(coefM1, R_outer=4.18)
        ])
        coefM2 = [0]*22 + (1e-8*args.mirrorFigure*fixedRNG.uniform(-0.5, 0.5, size=33)).tolist()
        M2surface = batoid.Sum([
            telescope.itemDict['LSST.M2'].surface,
            batoid.Zernike(coefM2, R_outer=1.71)
        ])
        coefM3 = [0]*22 + (1e-8*args.mirrorFigure*fixedRNG.uniform(-0.5, 0.5, size=33)).tolist()
        M3surface = batoid.Sum([
            telescope.itemDict['LSST.M3'].surface,
            batoid.Zernike(coefM3, R_outer=2.508)
        ])
        telescope = (telescope
            .withSurface('LSST.M1', M1surface)
            .withSurface('LSST.M2', M2surface)
            .withSurface('LSST.M3', M3surface)
        )

    if args.cameraFigure != 0.0:
        coef = [0]*2+(1e-7*args.cameraFigure*fixedRNG.uniform(-0.5, 0.5, size=55)).tolist()
        L1surface = batoid.Sum([
            telescope.itemDict['LSST.LSSTCamera.L1.L1_entrance'].surface,
            batoid.Zernike(coef, R_outer=0.775)
        ])
        telescope = telescope.withSurface('LSST.LSSTCamera.L1.L1_entrance', L1surface)

    if args.rotation is None:
        args.rotation = visitRNG.uniform(-np.pi/2, np.pi/2)
    telescope = telescope.withLocallyRotatedOptic(
        'LSST.LSSTCamera', batoid.RotZ(args.rotation)
    )
    # if args.wfShow:
    #     wf = batoid.analysis.wavefront(
    #         telescope, 0.0, 0.0, 625e-9, nx=256
    #     )
    #     plt.imshow(wf.array)
    #     plt.colorbar()
    #     plt.show()
    #     return

    if args.noGQ:
        telescope.clearObscuration()
        m1 = telescope.itemDict['LSST.M1']
        m1.obscuration = batoid.ObscNegation(batoid.ObscCircle(4.18))

    if args.chipAmp != 0.0:
        telescopes = {}
        chips = getChips(args.chipAmp, fixedRNG)
        for k, v in chips.items():
            telescopes[k] = (
                telescope
                .withSurface('LSST.LSSTCamera.Detector', v['surface'])
                .withGloballyShiftedOptic('LSST.LSSTCamera.Detector', v['center'])
            )

    zs = np.zeros((args.nstar, args.jmax+1), dtype=float)
    markBad = np.zeros(len(xs), dtype=bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if args.chipAmp != 0.0:
            try:
                k = getChip(x, y, telescope, chips)
            except ValueError:
                markBad[i] = True
                continue
            zTelescope = telescopes[k]
        else:
            zTelescope = telescope
        if args.noGQ:
            zs[i] = batoid.analysis.zernike(
                zTelescope, np.deg2rad(x), np.deg2rad(y), 625e-9,
                jmax=args.jmax, nx=64,
                reference='chief'
            )
        else:
            zs[i] = batoid.analysis.zernikeGQ(
                zTelescope, np.deg2rad(x), np.deg2rad(y), 625e-9,
                jmax=args.jmax, nrings=args.nrings, nspokes=args.nrings*2+1,
                reference='chief'
            )
    zs += args.zNoise*starRNG.normal(size=zs.shape)

    xs = xs[~markBad]
    ys = ys[~markBad]
    zs = zs[~markBad]

    if args.outFile is not None:
        import pickle
        with open(args.outFile, 'wb') as f:
            pickle.dump({"xs": xs, "ys":ys, "zs":zs, "args":args}, f)

    if args.show:
        Zpyramid(xs, ys, zs.T[4:], s=1)
        plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--nstar", default=1000, type=int)
    parser.add_argument("--visitSeed", default=None, type=int)
    parser.add_argument("--fixedSeed", default=577, type=int)
    parser.add_argument("--starSeed", default=None, type=int)
    parser.add_argument("--zNoise", default=0.01, type=float)

    parser.add_argument("--M2Rigid", default=1.0, type=float)
    parser.add_argument("--cameraRigid", default=1.0, type=float)

    parser.add_argument("--M1M3Bend", default=0.0, type=float)
    parser.add_argument("--M2Bend", default=0.0, type=float)

    parser.add_argument("--mirrorFigure", default=0.0, type=float)
    parser.add_argument("--cameraFigure", default=0.0, type=float)

    parser.add_argument("--chipAmp", default=0.0, type=float)

    parser.add_argument("--amplitude", default=1.0, type=float)

    parser.add_argument("--rotation", default=None, type=float)

    parser.add_argument("--jmax", default=28, type=int)
    parser.add_argument("--nrings", default=10, type=int)
    parser.add_argument("--outFile", default=None, type=str)
    parser.add_argument("--show", action='store_true')

    parser.add_argument("--noGQ", action='store_true')

    args = parser.parse_args()
    main(args)


# Goal
# "simulate" ~50 fields of donut data.  derive reference wavefront and degrees of freedom
# simulate ~50 fields of PSFs.  Use dof and reference and forward model to determine PSF.
