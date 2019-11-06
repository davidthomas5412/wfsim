import pickle

import numpy as np

import batoid
from galsim.utilities import lazy_property


effective_wavelengths = {
    'u':350e-9,
    'g':480e-9,
    'r':620e-9,
    'i':750e-9,
    'z':880e-9,
    'y':975e-9
}


with open("mirrorModes.pkl", 'rb') as f:
    bendingModeDict = pickle.load(f)


def rot(thx, thy):
    return np.dot(batoid.RotX(thx), batoid.RotY(thy))


def _fieldToFocal(field_x, field_y, telescope, wavelength):
    ray = batoid.Ray.fromStop(
        0.0, 0.0,
        optic=telescope,
        wavelength=wavelength,
        theta_x=field_x, theta_y=field_y, projection='gnomonic'
    )
    telescope.traceInPlace(ray)
    return ray.x, ray.y


def _focalToField(x, y, telescope, wavelength):
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
        self.wavelength = effective_wavelengths[band]

        self.M1_figure_coef = M1_figure_coef
        self.M2_figure_coef = M2_figure_coef
        self.M3_figure_coef = M3_figure_coef
        self.L1_figure_coef = L1_figure_coef
        self.chips = chips
        self._fillInChipFields()

    @lazy_property
    def fiducial_telescope(self):
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
        out = {}
        for k, v in self.chips.items():
            if 'zernikes' in v:
                out[k] = batoid.Zernike(
                    v['zernikes'],
                    R_outer=np.hypot(2000, 2036)*10e-6  # roughly the CCD diagonal
                )
            else:
                out[k] = batoid.Plane()
        # Note, this is the centered representation.  We'll shift down below.
        return out

    def _fillInChipFields(self):
        for k, chip in self.chips.items():
            chip['field_x'], chip['field_y'] = _focalToField(
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
        rotation = 0.0,
        dof = None
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
            M2_shift, M2_tilt,
            camera_shift, camera_tilt,
            M1M3_bend, M2_bend,
            rotation
        )


class VisitTelescope:
    """Make telescope instance for single visit.

    Parameters
    ----------
    factory : LSSTFactory
        Factory with permanent errors set.
    M2_shift : 3-tuple; meters
    M2_tilt : 2-tuple; radians
    camera_shift : 3-tuple; meters
    camera_tilt : 2-tuple; radians
    M1M3_bend : 20-tuple
    M2_bend : 20-tuple
    rotation : float
        Camera rotation in radians.
    """
    def __init__(
        self,
        factory,
        M2_shift=(0,0,0),
        M2_tilt=(0,0),
        camera_shift=(0,0,0),
        camera_tilt=(0,0),
        M1M3_bend=None,
        M2_bend=None,
        rotation=0
    ):
        self.factory = factory
        self.M2_shift = M2_shift
        self.M2_tilt = M2_tilt
        self.camera_shift = camera_shift
        self.camera_tilt = camera_tilt
        self.M1M3_bend = M1M3_bend
        self.M2_bend = M2_bend
        self.rotation = rotation

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
        return self.factory.fiducial_telescope

    @lazy_property
    def actual_telescope(self):
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
        )
        return telescope

    def dz(self, jmax=36, kmax=36, rings=10):
        return batoid.analysis.doubleZernike(
            self.actual_telescope, np.deg2rad(1.75), self.factory.wavelength,
            rings=rings, reference='chief', jmax=jmax, kmax=kmax, eps=0.61
        )

    @lazy_property
    def chip_telescopes(self):
        out = {}
        for k, v in self.factory.chips.items():
            out[k] = (
                self.actual_telescope
                .withSurface('Detector', self.factory.chip_error_dict[k])
                .withGloballyShiftedOptic('Detector', (v['x'], v['y'], 0.0))
            )
        return out

    def get_chip(self, x, y):
        ray = batoid.Ray.fromStop(
            0.0, 0.0, wavelength=self.factory.wavelength,
            theta_x=x, theta_y=y,
            optic=self.fiducial_telescope
        )
        self.fiducial_telescope.traceInPlace(ray)
        for k, chip in self.factory.chips.items():
            if (    ray.x > chip['left'] and
                    ray.x < chip['right'] and
                    ray.y > chip['bottom'] and
                    ray.y < chip['top']):
                return self.chip_telescopes[k]
        raise ValueError("Can't find chip")

    def get_zernike(self, x, y, **kwargs):
        telescope = self.get_chip(x, y)
        return batoid.analysis.zernikeGQ(
            telescope, x, y, self.factory.wavelength, eps=0.61, **kwargs
        )

    def get_wavefront(self, x, y, **kwargs):
        telescope = self.get_chip(x, y)
        return batoid.analysis.wavefront(
            telescope, x, y, self.factory.wavelength, **kwargs
        )

    def get_spot(self, x, y, naz=300, nrad=50, **kwargs):
        telescope = self.get_chip(x, y)
        rays = batoid.RayVector.asPolar(
            wavelength=self.factory.wavelength,
            outer=4.18, inner=2.5,
            theta_x=x, theta_y=y,
            optic=telescope,
            nrad=nrad, naz=naz,
            **kwargs
        )
        telescope.traceInPlace(rays)
        w = ~rays.vignetted
        return rays.x[w]-np.mean(rays.x[w]), rays.y[w]-np.mean(rays.y[w])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    with open("chips.pkl", 'rb') as f:
        chips = pickle.load(f)
    for k, v in chips.items():
        # only piston for now
        v['zernikes'] = np.random.normal(size=2)*np.array([0, 5e-6])

    factory = LSSTFactory(
        'i',
        chips=chips
    )

    visit_telescope = factory.make_visit_telescope()

    x, y = np.deg2rad(1.75), np.deg2rad(0)

    print(visit_telescope.dz())
    print()
    print(visit_telescope.get_zernike(x, y))
    print(visit_telescope.get_wf(x, y))
    print(visit_telescope.get_spot(x, y))
