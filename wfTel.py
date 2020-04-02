import pickle

import numpy as np

import batoid
from galsim.utilities import lazy_property
from galsim.zernike import zernikeRotMatrix
from galsim import BaseDeviate, GaussianDeviate

wavelength_dict = dict(
    u=365.49,
    g=480.03,
    r=622.20,
    i=754.06,
    z=868.21,
    y=991.66
)

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

with open("mirrorModes.pkl", 'rb') as f:
    bendingModeDict = pickle.load(f)


def rot(thx, thy):
    return np.dot(batoid.RotX(thx), batoid.RotY(thy))


def _fieldToFocal(cam_field_x, cam_field_y, telescope, wavelength):
    ray = batoid.Ray.fromStop(
        0.0, 0.0,
        optic=telescope,
        wavelength=wavelength,
        theta_x=cam_field_x, theta_y=cam_field_y, projection='gnomonic'
    )
    telescope.traceInPlace(ray)
    return ray.x, ray.y


def _focalToCamField(x, y, telescope, wavelength):
    from scipy.optimize import least_squares

    def _resid(args):
        fx1, fy1 = args
        x1, y1 = _fieldToFocal(fx1, fy1, telescope, wavelength)
        return np.array([x1-x, y1-y])

    result = least_squares(_resid, np.array([0.0, 0.0]))
    return result.x[0], result.x[1]


class LSSTFactory:
    """Create LSST telescope factory that can produce telescopes with permanent
    errors included.

    Parameters
    ----------
    band : {'u', 'g', 'r', 'i', 'z', 'y'}
        Which filter to load
    M1_figure_coef : array_like
        Zernike coefficients describing M1 figure error.
    M2_figure_coef : array_like
        Zernike coefficients describing M2 figure error.
    M3_figure_coef : array_like
        Zernike coefficients describing M3 figure error.
    L1_figure_coef : array_like
        Zernike coefficients describing L1 entrance figure error.
    chips : dict
        Keys are detector names
        Values are dicts with key/value pairs:
            left : left edge in meters
            right : right edge in meters
            top : top edge in meters
            bottom : bottom edge in meters
            x, y : chip center in meters
    """
    def __init__(
        self,
        band,
        M1_figure_coef=None,
        M2_figure_coef=None,
        M3_figure_coef=None,
        L1_figure_coef=None,
        chips=None,
    ):
        self.band = band
        self.wavelength = wavelength_dict[band]*1e-9 # nm -> m

        self.M1_figure_coef = M1_figure_coef
        self.M2_figure_coef = M2_figure_coef
        self.M3_figure_coef = M3_figure_coef
        self.L1_figure_coef = L1_figure_coef
        self.chips = chips
        self._fillInChipFields()

    @lazy_property
    def fiducial_telescope(self):
        """Fiducial telescope.  No perturbations.  No chip gaps.
        """
        fn = f"LSST_{self.band}.yaml"
        return batoid.Optic.fromYaml(fn)

    @lazy_property
    def M1_error(self):
        return batoid.Zernike(self.M1_figure_coef, R_outer=4.18)

    @lazy_property
    def M2_error(self):
        return batoid.Zernike(self.M2_figure_coef, R_outer=1.71)

    @lazy_property
    def M3_error(self):
        return batoid.Zernike(self.M3_figure_coef, R_outer=2.508)

    @lazy_property
    def L1_error(self):
        return batoid.Zernike(self.L1_figure_coef, R_outer=0.775)

    @lazy_property
    def chip_error_dict(self):
        """Dictionary of individual chip surfaces.  Indexed by chip names.
        """
        out = {}
        for k, v in self.chips.items():
            if 'zernikes' in v:
                out[k] = batoid.Zernike(
                    v['zernikes'],
                    R_outer=np.hypot(2000, 2036)*10e-6  # roughly the diagonal
                )
            else:
                out[k] = batoid.Plane()
        # Note, this is the centered representation.  We'll shift down below.
        return out

    def _fillInChipFields(self):
        if self.chips:
            for k, chip in self.chips.items():
                chip['cam_field_x'], chip['cam_field_y'] = _focalToCamField(
                    chip['x'], chip['y'],
                    self.fiducial_telescope,
                    self.wavelength
                )

    def make_visit_telescope(
        self,
        M2_shift = (0,0,0),
        M2_tilt = (0,0),
        camera_shift = (0,0,0),
        camera_tilt = (0,0),
        M1M3_bend = None,
        M2_bend = None,
        defocus = 0.0,
        rotation = 0.0,
        dof = None,
        **kwargs
    ):
        """Create a perturbed telescope for a single visit.
        """
        if dof is not None:
            # override!
            M2_shift = dof[0:3]
            M2_tilt = dof[3:5]
            camera_shift = dof[5:8]
            camera_tilt = dof[8:10]
            M1M3_bend = dof[10:30]
            # ignore M2_bend for now.

        return VisitTelescope(
            self,
            M2_shift=M2_shift,
            M2_tilt=M2_tilt,
            camera_shift=camera_shift,
            camera_tilt=camera_tilt,
            M1M3_bend=M1M3_bend,
            M2_bend=M2_bend,
            defocus=defocus,
            rotation=rotation,
            **kwargs
        )


class VisitTelescope:
    """Make telescope instance for single visit.

    Parameters
    ----------
    factory : LSSTFactory
        Factory with permanent errors set.
    M2_shift : 3-tuple; meters
    M2_tilt : 2-tuple; radians
    M2_amplitude : float; sets M2_shift and M2_tilt randomly
    camera_shift : 3-tuple; meters
    camera_tilt : 2-tuple; radians
    camera_amplitude : float; sets M2_shift and M2_tilt randomly
    M1M3_bend : 20-tuple
    M1M3_bend_amplitude : float; sets M1M3_bend randomly
    M2_bend : 20-tuple
    M2_bend_amplitude : float; sets M2_bend randomly
    defocus : float; meters
    rotation : float
        Camera rotation in radians.
    """
    def __init__(
        self,
        factory,
        M2_shift=(0,0,0),
        M2_tilt=(0,0),
        M2_amplitude=None,
        camera_shift=(0,0,0),
        camera_tilt=(0,0),
        camera_amplitude=None,
        M1M3_bend=None,
        M1M3_bend_amplitude=None,
        M2_bend=None,
        M2_bend_amplitude=None,
        defocus=0.0,
        rotation=0,
        rng=None
    ):
        if rng is None:
            rng = BaseDeviate()
        gd = GaussianDeviate(rng)

        if M2_amplitude is not None:
            M2_dx = gd()
            M2_dx *= np.ptp(norm['M2 x'])/2
            M2_dx += np.mean(norm['M2 x'])
            M2_dy = gd()
            M2_dy *= np.ptp(norm['M2 y'])/2
            M2_dy += np.mean(norm['M2 y'])
            M2_dz = gd()
            M2_dz *= np.ptp(norm['M2 z'])/2
            M2_dz += np.mean(norm['M2 z'])
            M2_shift = np.array([M2_dx, M2_dy, M2_dz])
            M2_shift *= M2_amplitude

            M2_thx = gd()
            M2_thx *= np.ptp(norm['M2 thx'])/2
            M2_thx += np.mean(norm['M2 thx'])
            M2_thy = gd()
            M2_thy *= np.ptp(norm['M2 thy'])/2
            M2_thy += np.mean(norm['M2 thy'])
            M2_tilt = np.array([M2_thx, M2_thy])
            M2_tilt *= M2_amplitude

        if camera_amplitude is not None:
            camera_dx = gd()
            camera_dx *= np.ptp(norm['camera x'])/2
            camera_dx += np.mean(norm['camera x'])
            camera_dy = gd()
            camera_dy *= np.ptp(norm['camera y'])/2
            camera_dy += np.mean(norm['camera y'])
            camera_dz = gd()
            camera_dz *= np.ptp(norm['camera z'])/2
            camera_dz += np.mean(norm['camera z'])
            camera_shift = np.array([camera_dx, camera_dy, camera_dz])
            camera_shift *= camera_amplitude

            camera_thx = gd()
            camera_thx *= np.ptp(norm['camera thx'])/2
            camera_thx += np.mean(norm['camera thx'])
            camera_thy = gd()
            camera_thy *= np.ptp(norm['camera thy'])/2
            camera_thy += np.mean(norm['camera thy'])
            camera_tilt = np.array([camera_thx, camera_thy])
            camera_tilt *= camera_amplitude

        if M1M3_bend_amplitude is not None:
            M1M3_bend = []
            for i in range(1, 21):
                val = gd()
                val *= np.ptp(norm[f'M1M3 bend {i}'])/2
                val += np.mean(norm[f'M1M3 bend {i}'])
                M1M3_bend.append(val)
            M1M3_bend = np.array(M1M3_bend)*M1M3_bend_amplitude

        if M2_bend_amplitude is not None:
            M2_bend = []
            for i in range(1, 21):
                val = gd()
                val *= np.ptp(norm[f'M2 bend {i}'])/2
                val += np.mean(norm[f'M2 bend {i}'])
                M2_bend.append(val)
            M2_bend = np.array(M2_bend)*M2_bend_amplitude

        self.factory = factory
        self.M2_shift = M2_shift
        self.M2_tilt = M2_tilt
        self.camera_shift = camera_shift
        self.camera_tilt = camera_tilt
        self.M1M3_bend = M1M3_bend
        self.M2_bend = M2_bend
        self.rotation = rotation
        self.defocus = defocus

        if self.M1M3_bend is not None:
            self.M1_grid = np.einsum(
                'a,abc->bc', self.M1M3_bend, bendingModeDict['M1Modes']
            )
            self.M3_grid = np.einsum(
                'a,abc->bc', self.M1M3_bend, bendingModeDict['M3Modes']
            )

        if self.M2_bend is not None:
            self.M2_grid = np.einsum(
                'a,abc->bc', self.M2_bend, bendingModeDict['M2Modes']
            )

    @lazy_property
    def fiducial_telescope(self):
        """Unperturbed, unrotated, undefocused, no chip gaps"""
        return self.factory.fiducial_telescope

    @lazy_property
    def rotated_fiducial_telescope(self):
        """Unperturbed.  No chip gaps.  But rotator is applied."""
        return (
            self.fiducial_telescope
            .withLocallyRotatedOptic('LSSTCamera', batoid.RotZ(self.rotation))
        )

    @lazy_property
    def actual_telescope(self):
        """Perturbed, rotated, defocused.  No chip gaps."""
        telescope = self.fiducial_telescope

        M1_terms = [telescope['M1'].surface]
        if self.M1M3_bend is not None:
            M1_terms.append(batoid.Bicubic(
                bendingModeDict['xgrid'],
                bendingModeDict['ygrid'],
                self.M1_grid
            ))
        if self.factory.M1_figure_coef is not None:
            M1_terms.append(self.factory.M1_error)
        if len(M1_terms) > 1:
            telescope = telescope.withSurface('M1', batoid.Sum(M1_terms))

        M2_terms = [telescope['M2'].surface]
        if self.M2_bend is not None:
            M2_terms.append(batoid.Bicubic(
                bendingModeDict['xgrid'],
                bendingModeDict['ygrid'],
                self.M2_grid
            ))
        if self.factory.M2_figure_coef is not None:
            M2_terms.append(self.factory.M2_error)
        if len(M2_terms) > 1:
            telescope = telescope.withSurface('M2', batoid.Sum(M2_terms))

        M3_terms = [telescope['M3'].surface]
        if self.M1M3_bend is not None:
            M3_terms.append(batoid.Bicubic(
                bendingModeDict['xgrid'],
                bendingModeDict['ygrid'],
                self.M3_grid
            ))
        if self.factory.M3_figure_coef is not None:
            M3_terms.append(self.factory.M3_error)
        if len(M3_terms) > 1:
            telescope = telescope.withSurface('M3', batoid.Sum(M3_terms))

        L1_terms = [telescope['L1_entrance'].surface]
        if self.factory.L1_figure_coef is not None:
            L1_surface = batoid.Sum([
                telescope['L1_entrance'],
                self.factory.L1_figure_surface
            ])
            telescope = telescope.withSurface('L1_entrance', L1_surface)

        telescope = (telescope
            .withGloballyShiftedOptic('M2', self.M2_shift)
            .withLocallyRotatedOptic('M2', rot(*self.M2_tilt))
            .withGloballyShiftedOptic('LSSTCamera', self.camera_shift)
            .withLocallyRotatedOptic('LSSTCamera', rot(*self.camera_tilt))
            .withLocallyRotatedOptic('LSSTCamera', batoid.RotZ(self.rotation))
            .withGloballyShiftedOptic('LSSTCamera', (0, 0, self.defocus))
        )
        return telescope

    def dz(self, jmax=36, kmax=36, rings=10):
        """Double Zernike coefficients of actual_telescope (rotated, perturbed,
        no chip gaps)
        """
        return batoid.analysis.doubleZernike(
            self.actual_telescope, np.deg2rad(1.75), self.factory.wavelength,
            rings=rings, reference='chief', jmax=jmax, kmax=kmax, eps=0.61
        )

    @lazy_property
    def chip_telescopes(self):
        """Dictionary of individual chip telescopes, which include
        perturbations, the rotator, and chip gaps.
        """
        out = {}
        for k, v in self.factory.chips.items():
            out[k] = (
                self.actual_telescope
                .withSurface('Detector', self.factory.chip_error_dict[k])
                .withLocallyShiftedOptic('Detector', (v['x'], v['y'], 0.0))
            )
        return out

    def get_chip(self, thx, thy):
        """Determine a chip by tracing a chief ray through the
        rotated_fiducial_telescope.  So handles rotation only.  No
        perturbations.  No chip gaps.  Returns chip name.
        """
        ray = batoid.Ray.fromStop(
            0.0, 0.0, wavelength=self.factory.wavelength,
            theta_x=thx, theta_y=thy,
            optic=self.fiducial_telescope
        )
        self.rotated_fiducial_telescope.traceInPlace(ray)
        for k, chip in self.factory.chips.items():
            if (    ray.x > chip['left'] and
                    ray.x < chip['right'] and
                    ray.y > chip['bottom'] and
                    ray.y < chip['top']):
                return k
        raise ValueError("Can't find chip")

    def get_chip_telescope(self, thx, thy):
        """Return perturbed, rotated, gappy telescope for chip at thx, thy"""
        if self.factory.chips is None:
            return self.actual_telescope
        k = self.get_chip(thx, thy)
        return self.chip_telescopes[k]

    def get_zernike(self, thx, thy, **kwargs):
        """Return Zernike as measured in CCD frame.
        """
        telescope = self.get_chip_telescope(thx, thy)
        z = batoid.analysis.zernikeGQ(
            telescope, thx, thy, self.factory.wavelength, eps=0.61, **kwargs
        )
        return np.dot(zernikeRotMatrix(len(z)-1, -self.rotation), z)

    def get_wavefront(self, thx, thy, **kwargs):
        """Return wavefront in telescope frame (entrance pupil coords)"""
        telescope = self.get_chip_telescope(thx, thy)
        return batoid.analysis.wavefront(
            telescope, thx, thy, self.factory.wavelength, **kwargs
        )

    def get_spot(self, thx, thy, naz=300, nrad=50, reference='mean', **kwargs):
        """Return spots.  In CCD coords.
        """
        telescope = self.get_chip_telescope(thx, thy)
        rays = batoid.RayVector.asPolar(
            wavelength=self.factory.wavelength,
            outer=4.18, inner=2.5,
            theta_x=thx, theta_y=thy,
            optic=telescope,
            nrad=nrad, naz=naz,
            **kwargs
        )
        telescope.traceInPlace(rays)
        w = ~rays.vignetted
        if reference == 'mean':
            return rays.x[w]-np.mean(rays.x[w]), rays.y[w]-np.mean(rays.y[w])
        elif reference == 'chief':
            ray = batoid.Ray.fromStop(
                0, 0,
                wavelength=self.factory.wavelength,
                theta_x=thx, theta_y=thy,
                optic=telescope
            )
            telescope.traceInPlace(ray)
            return rays.x[w]-ray.x, rays.y[w]-ray.y
        elif reference == 'None':
            return rays.x[w], rays.y[w]

    def get_fp(self, thx, thy, type='spot'):
        """Detailed FP position.  Trace from a non-vignetted ring on the pupil.
        Or from the chief ray.
        """
        if type == 'chief':
            ray = batoid.Ray.fromStop(
                0.0, 0.0, wavelength=self.factory.wavelength,
                theta_x=thx, theta_y=thy,
                optic=self.fiducial_telescope
            )
            self.actual_telescope.traceInPlace(ray)
            return np.array([ray.x, ray.y])
        elif type == 'spot':
            rays = batoid.RayVector.asSpokes(
                wavelength=self.factory.wavelength,
                outer=3.5, inner=3.2,
                theta_x=thx, theta_y=thy,
                optic=self.fiducial_telescope,
                rings=3, spokes=51
            )
            self.actual_telescope.traceInPlace(rays)
            if np.any(rays.vignetted) or np.any(rays.failed):
                raise ValueError()
            return np.array([np.mean(rays.x), np.mean(rays.y)])
        elif type == 'chip':
            rays = batoid.RayVector.asSpokes(
                wavelength=self.factory.wavelength,
                outer=3.5, inner=3.2,
                theta_x=thx, theta_y=thy,
                optic=self.fiducial_telescope,
                rings=3, spokes=51
            )
            telescope = self.get_chip_telescope(thx, thy)
            telescope.traceInPlace(rays)
            w = ~rays.vignetted
            return np.array([np.mean(rays.x[w]), np.mean(rays.y[w])])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    with open("chips.pkl", 'rb') as f:
        chips = pickle.load(f)
    for k, v in chips.items():
        # only piston for now
        v['zernikes'] = np.random.normal(size=2)*np.array([0, 5e-6])

    factory = LSSTFactory(
        'i',
        chips=chips,
    )

    visit_telescope = factory.make_visit_telescope(rotation=0)

    x, y = np.deg2rad(1.0), np.deg2rad(1.0)
    print(visit_telescope.get_chip(x, y))
