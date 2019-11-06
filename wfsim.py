import pickle

import numpy as np
import batoid
from galsim.utilities import lazy_property


def addAxes(rect, fig):
    ax = fig.add_axes(rect)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def rot(thx, thy):
    return np.dot(batoid.RotX(thx), batoid.RotY(thy))


def makeChips(chipAmp, fixedRNG):
    """
    chipAmp = amplitude of random perturbations
    fixedRNG = random number generator to use
    """
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
            surface = batoid.Zernike(
                [0]+1e-5*chipAmp*fixedRNG.uniform(-0.5, 0.5, size=3),
                R_outer=chipWidth
            )

            chips[k] = dict(
                leftEdge = leftEdges[i],
                rightEdge = rightEdges[i],
                bottomEdge = bottomEdges[j],
                topEdge = topEdges[j],
                center = (
                    0.5*(leftEdges[i]+rightEdges[i]),
                    0.5*(topEdges[j]+bottomEdges[j]),
                    0.0
                ),
                surface = surface
            )
            k += 1
    return chips


with open("mirrorModes.pkl", 'rb') as f:
    bendingModeDict = pickle.load(f)


class LSSTVisitTelescope:
    def __init__(
        self,
        M2Shift=(0,0,0), M2Tilt=(0,0),
        cameraShift=(0,0,0), cameraTilt=(0,0),
        M1M3Bend=tuple([0]*20), M2Bend=tuple([0]*20),
        M1Figure=tuple([0]*33), M2Figure=tuple([0]*33),
        M3Figure=tuple([0]*33), cameraFigure=tuple([0]*33),
        rotation=0,
        chips=None
    ):
        self.M2Shift = M2Shift
        self.M2Tilt = M2Tilt
        self.cameraShift = cameraShift
        self.cameraTilt = cameraTilt
        self.M1M3Bend = M1M3Bend
        self.M2Bend = M2Bend
        self.M1Figure = M1Figure
        self.M2Figure = M2Figure
        self.M3Figure = M3Figure
        self.cameraFigure = cameraFigure
        self.rotation = rotation
        self.chips = chips

        # process the bending mode coefs
        self.M1grid = np.einsum('a,abc->bc', self.M1M3Bend, bendingModeDict['M1Modes'])
        self.M3grid = np.einsum('a,abc->bc', self.M1M3Bend, bendingModeDict['M3Modes'])
        # and some day M2...

    @lazy_property
    def fiducial_telescope(self):
        return batoid.Optic.fromYaml("LSST_i.yaml")

    @lazy_property
    def telescope(self):
        # Rigid body perturbations
        telescope = (self.fiducial_telescope
            .withGloballyShiftedOptic("LSST.M2", self.M2Shift)
            .withLocallyRotatedOptic("LSST.M2", rot(*self.M2Tilt))
            .withGloballyShiftedOptic("LSST.LSSTCamera", self.cameraShift)
            .withLocallyRotatedOptic("LSST.LSSTCamera", rot(*self.cameraTilt))
        )

        # Figure errors and bending modes
        M1surface = batoid.Sum([
            self.fiducial_telescope['M1'].surface,
            batoid.Zernike(self.M1Figure, R_outer=4.18),
            batoid.Bicubic(
                bendingModeDict['xgrid'],
                bendingModeDict['ygrid'],
                self.M1grid
            )
        ])
        M2surface = batoid.Sum([
            self.fiducial_telescope['M2'].surface,
            batoid.Zernike(self.M2Figure, R_outer=1.71)
            # someday bending modes...
        ])
        M3surface = batoid.Sum([
            self.fiducial_telescope['M3'].surface,
            batoid.Zernike(self.M3Figure, R_outer=2.508),
            batoid.Bicubic(
                bendingModeDict['xgrid'],
                bendingModeDict['ygrid'],
                self.M3grid
            )
        ])
        L1surface = batoid.Sum([
            self.fiducial_telescope['L1_entrance'].surface,
            batoid.Zernike(self.cameraFigure, R_outer=0.775)
        ])
        telescope = (telescope
            .withSurface('LSST.M1', M1surface)
            .withSurface('LSST.M2', M2surface)
            .withSurface('LSST.M3', M3surface)
            .withSurface('LSST.LSSTCamera.L1.L1_entrance', L1surface)
            .withLocallyRotatedOptic('LSST.LSSTCamera', batoid.RotZ(self.rotation))
        )
        return telescope

    def getChip(self, x, y):
        ray = batoid.Ray.fromStop(
            0.0, 0.0, wavelength=750e-9,
            theta_x=x, theta_y=y,
            optic=self.telescope
        )
        self.fiducial_telescope.traceInPlace(ray)
        for k, chip in self.chips.items():
            if (    ray.x > chip['leftEdge'] and
                    ray.x < chip['rightEdge'] and
                    ray.y > chip['bottomEdge'] and
                    ray.y < chip['topEdge']):
                return k
        raise ValueError("Can't find chip")

    @lazy_property
    def chipTelescopes(self):
        out = {}
        for k, v in self.chips.items():
            out[k] = (
                self.telescope
                .withSurface('LSST.LSSTCamera.Detector', v['surface'])
                .withGloballyShiftedOptic('LSST.LSSTCamera.Detector', v['center'])
            )
        return out

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_chipTelescopes', None)
        d.pop('_fiducial_telescope', None)
        d.pop('_telescope', None)

    def getZernikes(self, x, y, use_chip=True, **kwargs):
        if not use_chip:
            telescope = self.telescope
        else:
            telescope = self.chipTelescopes[self.getChip(x, y)]

        return batoid.analysis.zernikeGQ(
            telescope,
            x, y, 750e-9,
            **kwargs
        )

    def getWF(self, x, y, **kwargs):
        return batoid.analysis.wavefront(
            self.chipTelescopes[self.getChip(x, y)],
            x, y, 750e-9,
            **kwargs
        )

    def getSpot(self, x, y, **kwargs):
        rays = batoid.RayVector.asPolar(
            wavelength=750e-9,
            outer=4.18, inner=2.5,
            theta_x=x, theta_y=y,
            optic=self.fiducial_telescope,
            **kwargs
        )
        telescope = self.chipTelescopes[self.getChip(x, y)]
        telescope.traceInPlace(rays)
        w = ~rays.vignetted
        return rays.x[w]-np.mean(rays.x[w]), rays.y[w]-np.mean(rays.y[w])


def main(args):
    visitRNG = np.random.RandomState(args.visitSeed)
    fixedRNG = np.random.RandomState(args.fixedSeed)
    starRNG = np.random.RandomState(args.starSeed)

    nrepeat = 1 if args.nrepeat is None else args.nrepeat
    for _ in range(nrepeat):

        # Rigid body perturbations
        M2Dx = 0.6e-4
        M2Dz = 0.012e-4
        M2Tilt = np.deg2rad(0.25/60)
        M2Shift = args.M2Rigid*visitRNG.normal(scale=(M2Dx, M2Dx, M2Dz))
        M2Tilt = args.M2Rigid*visitRNG.normal(scale=(M2Tilt, M2Tilt))

        cameraDx = 3e-3
        cameraDz = 0.04e-3 * 0.3
        cameraTilt = np.deg2rad(0.75/60) * 0.3
        cameraShift = args.cameraRigid*visitRNG.normal(scale=(cameraDx, cameraDx, cameraDz))
        cameraTilt = args.cameraRigid*visitRNG.normal(scale=(cameraTilt, cameraTilt))

        # Permanent figure errors.  We'll use high-ish order Zernikes for these.
        M1Figure = [0]*22 + (1e-8*args.mirrorFigure*fixedRNG.uniform(-0.5, 0.5, size=34)).tolist()
        M2Figure = [0]*22 + (1e-8*args.mirrorFigure*fixedRNG.uniform(-0.5, 0.5, size=34)).tolist()
        M3Figure = [0]*22 + (1e-8*args.mirrorFigure*fixedRNG.uniform(-0.5, 0.5, size=34)).tolist()
        cameraFigure = [0]*2+(3e-7*args.cameraFigure*fixedRNG.uniform(-0.5, 0.5, size=54)).tolist()

        # Bending modes
        M1M3Bend = 5e-2*args.M1M3Bend*visitRNG.uniform(-0.5, 0.5, size=20)

        # Camera rotator
        if args.rotate:
            rotation = visitRNG.uniform(-np.pi/2, np.pi/2)
        else:
            rotation = 0

        chips = makeChips(args.chipAmp, fixedRNG)

        lvt = LSSTVisitTelescope(
            M1Figure=M1Figure,
            M2Figure=M2Figure,
            M3Figure=M3Figure,
            cameraFigure=cameraFigure,
            M2Shift=M2Shift,
            M2Tilt=M2Tilt,
            cameraShift=cameraShift,
            cameraTilt=cameraTilt,
            M1M3Bend=M1M3Bend,
            rotation=rotation,
            chips=chips
        )


        # Start with circular field of view
        focalRadius = 1.75 # degrees
        th = starRNG.uniform(0, 2*np.pi, size=args.nstar)
        ph = np.sqrt(starRNG.uniform(0, focalRadius**2, size=args.nstar))
        xs = ph*np.cos(th) # positions in degrees
        ys = ph*np.sin(th)

        zs = np.zeros((args.nstar, args.jmax+1), dtype=float)
        markBad = np.zeros(len(xs), dtype=bool)
        for i, (x, y) in enumerate(zip(xs, ys)):
            try:
                zs[i] = lvt.getZernikes(
                    np.deg2rad(x), np.deg2rad(y),
                    jmax=args.jmax, rings=args.rings, reference='chief'
                )
            except ValueError:
                markBad[i] = True
                continue
        zs += args.zNoise*starRNG.normal(size=zs.shape)

        xs = xs[~markBad]
        ys = ys[~markBad]
        zs = zs[~markBad]

        if args.outFile is not None:
            import pickle
            with open(args.outFile, 'wb') as f:
                pickle.dump(
                    dict(
                        args=args,
                        xs=xs, ys=ys, zs=zs,
                        lvt=lvt
                    ),
                f)

        if args.showZernikes:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(13, 8))
            batoid.plotUtils.zernikePyramid(xs, ys, zs.T[4:], s=1, fig=fig)
            plt.show()

        if args.showSpots:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 10))

            for k, chip in lvt.chips.items():
                rect = np.array([
                    chip['bottomEdge'],
                    chip['leftEdge'],
                    chip['topEdge'] - chip['bottomEdge'],
                    chip['rightEdge'] - chip['leftEdge']
                ])
                rect *= 1.3
                rect += [0.5, 0.5, 0, 0]
                ax = addAxes(rect, fig)
                center = chip['center']
                spotx, spoty = lvt.getSpot(
                    np.deg2rad(center[0]/0.3),
                    np.deg2rad(center[1]/0.3),
                    naz=90, nrad=30
                )
                spotx *= 1e6
                spoty *= 1e6
                ax.scatter(spoty, spotx, s=0.05, c='k', alpha=0.05)
                ax.set_xlim(-20, 20)  # 40 micron = 4 pixels
                ax.set_ylim(-20, 20)
            plt.show()

        if args.showWavefront:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 10))

            for k, chip in lvt.chips.items():
                rect = np.array([
                    chip['bottomEdge'],
                    chip['leftEdge'],
                    chip['topEdge'] - chip['bottomEdge'],
                    chip['rightEdge'] - chip['leftEdge']
                ])
                rect *= 1.3
                rect += [0.5, 0.5, 0, 0]
                ax = addAxes(rect, fig)
                center = chip['center']
                wf = lvt.getWF(
                    np.deg2rad(center[0]/0.3),
                    np.deg2rad(center[1]/0.3),
                    nx=63
                )
                ax.imshow(wf.array, vmin=-1, vmax=1, cmap='seismic')
            plt.show()

        if args.reportDZ:
            dzs = batoid.analysis.doubleZernike(
                lvt.telescope, np.deg2rad(1.75), 750e-9, rings=args.rings,
                reference='chief', jmax=args.jmax, kmax=args.jmax, eps=0.61
            )
            print(np.sqrt(np.sum(dzs[:, 4:]**2)))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--nstar", default=1000, type=int)
    parser.add_argument(
        "--visitSeed", default=None, type=int,
        help="Seed affecting rigid body perturbations, bending modes, and "
             "rotation angle"
    )
    parser.add_argument(
        "--fixedSeed", default=577, type=int,
        help="Seed affecting chip-gaps and permanent figure errors"
    )
    parser.add_argument(
        "--starSeed", default=None, type=int,
        help="Seed affecting positions of stars and measurement errors"
    )
    parser.add_argument("--zNoise", default=0.01, type=float)

    parser.add_argument("--M2Rigid", default=0.0, type=float)
    parser.add_argument("--cameraRigid", default=0.0, type=float)

    parser.add_argument("--M1M3Bend", default=0.0, type=float)
    parser.add_argument("--M2Bend", default=0.0, type=float)

    parser.add_argument("--mirrorFigure", default=0.0, type=float)
    parser.add_argument("--cameraFigure", default=0.0, type=float)

    parser.add_argument("--chipAmp", default=0.0, type=float)

    parser.add_argument("--rotate", action='store_true')

    parser.add_argument("--jmax", default=36, type=int)
    parser.add_argument("--rings", default=6, type=int)
    parser.add_argument("--outFile", default=None, type=str)
    parser.add_argument("--showZernikes", action='store_true')
    parser.add_argument("--showWavefront", action='store_true')
    parser.add_argument("--showSpots", action='store_true')
    parser.add_argument("--reportDZ", action='store_true')
    parser.add_argument("--nrepeat", default=None, type=int)

    args = parser.parse_args()
    main(args)


# Goal
# "simulate" ~50 fields of donut data.  derive reference wavefront and degrees of freedom
# simulate ~50 fields of PSFs.  Use dof and reference and forward model to determine PSF.
