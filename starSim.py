import multiprocessing
import numpy as np
from scipy.optimize import bisect
from astropy.table import Table
from tqdm import tqdm
from astropy.table import Table, vstack
from astroquery.gaia import Gaia

import galsim
import batoid
import pickle

from wfTel import LSSTFactory
from survey import Survey

wavelength_dict = dict(
    u=365.49,
    g=480.03,
    r=622.20,
    i=754.06,
    z=868.21,
    y=991.66
)

class FocalPlane:
    """
    Keeps track of intra and extra-focal chips positions in focal plane.
    """
    def __init__(self):
        chips = pickle.load(open('chips.pkl', 'rb'))
        self.wavefront_sensors = {k: v for (k,v) in chips.items() if 'SW' in k}

class CatalogFactory:
    """
    Queries Gaia catalog.
    """
    def __init__(self, focal_plane):
        self.focal_plane = focal_plane

    def make_catalog(self, boresight, q, mag_cutoff=18, verbose=True):
        """
        The sdss_r_mag relationship comes from 
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
        viewed on 2020/4/7.
        """
        cq, sq = np.cos(q), np.sin(q)
        affine = galsim.AffineTransform(cq, -sq, sq, cq)
        wcs = galsim.TanWCS(
                    affine,
                    boresight,
                    units=galsim.radians
                )

        stack = []
        for name, positions in self.focal_plane.wavefront_sensors.items():
            corners = np.array(positions['corners_field'])
            ra, dec = wcs.toWorld(corners[:,0], corners[:,1], units=galsim.degrees)
            result = CatalogFactory.__chip_table(ra, dec, mag_cutoff, verbose)
            result['name'] = name
            stack.append(result)
        stack = vstack(stack)
        
        # convert magnitudes
        x = stack['phot_bp_mean_mag'] - stack['phot_rp_mean_mag']
        G_minus_r = -0.12879 + 0.24662 * x - 0.027464 * x ** 2 - 0.049465 * x ** 3
        stack['sdss_r_mag'] = stack['phot_g_mean_mag'] - G_minus_r

        return stack

    @staticmethod
    def __chip_table(ra, dec, mag_cutoff, verbose):
        query = f"""SELECT source_id, ra, dec, teff_val, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag FROM gaiadr2.gaia_source
        WHERE phot_g_mean_mag < {mag_cutoff}
        AND 1=CONTAINS(POINT('ICRS',ra,dec), {CatalogFactory.__polygon_string(ra, dec)})
        """
        job = Gaia.launch_job(query=query, verbose=verbose)
        return job.get_results()

    @staticmethod
    def __polygon_string(ra, dec):
        return f"POLYGON('ICRS', {ra[0]}, {dec[0]}," + \
            f"{ra[1]},{dec[1]},{ra[2]}," + \
            f"{dec[2]},{ra[3]},{dec[3]})"

class Survey:
    """
    Stores table of LSST observations.
    
    Notes
    -----
    The raw data is generated from the OpSim baseline_2snapsv1.4_10yrs.db database. The query sequence is:
        .mode csv
        .output survey.csv
        SELECT observationId, fieldRA, fieldDec, airmass, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" ORDER BY random() LIMIT 10000;
        SELECT observationId, fieldRA, fieldDec, airmass, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 10000 LIMIT 100;
        SELECT observationId, fieldRA, fieldDec, airmass, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 20000 LIMIT 100;
        SELECT observationId, fieldRA, fieldDec, airmass, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 30000 LIMIT 100;
        SELECT observationId, fieldRA, fieldDec, airmass, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 40000 LIMIT 100;
        SELECT observationId, fieldRA, fieldDec, airmass, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 50000 LIMIT 100;
        .output stdout
    """
    survey_file = 'survey.csv'

    def __init__(self):
        self.table = Table.read(Survey.survey_file, names=['observationId', 'fieldRA', 'fieldDec', 'airmass',\
         'filter', 'rotTelPos', 'rotSkyPos', 'skyBrightness', 'seeingFwhm500'])

    def get_observation(self, idx):
        row = self.table[idx]
        observation = {
            'boresight': galsim.CelestialCoord(row['fieldRA']*galsim.degrees, row['fieldDec']*galsim.degrees),
            'airmass': row['airmass'],
            'rotTelPos': row['rotTelPos']*galsim.degrees,  # zenith measured CCW from up
            'rotSkyPos': row['rotSkyPos']*galsim.degrees,  # N measured CCW from up
            'rawSeeing': row['seeingFwhm500']*galsim.arcsec,
            'skyBrightness': row['skyBrightness'],
            'band': row['filter'],
            'exptime': 15.0,
            'temperature': 293.15,  # K
            'pressure': 69.328,  # kPa
            'H2O_pressure': 1.067,  # kPa
        }
        return observation

def BBSED(T):
    """(unnormalized) Blackbody SED for temperature T in Kelvin.
    """
    waves_nm = np.arange(330.0, 1120.0, 10.0)
    def planck(t, w):
        # t in K
        # w in m
        c = 2.99792458e8  # speed of light in m/s
        kB = 1.3806488e-23  # Boltzmann's constant J per Kelvin
        h = 6.62607015e-34  # Planck's constant in J s
        return w**(-5) / (np.exp(h*c/(w*kB*t))-1)
    flambda = planck(T, waves_nm*1e-9)
    return galsim.SED(
        galsim.LookupTable(waves_nm, flambda),
        wave_type='nm',
        flux_type='flambda'
    )


def _vkSeeing(r0_500, wavelength, L0):
    # von Karman profile FWHM from Tokovinin fitting formula
    kolm_seeing = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
    r0 = r0_500 * (wavelength/500)**1.2
    arg = 1. - 2.183*(r0/L0)**0.356
    factor = np.sqrt(arg) if arg > 0.0 else 0.0
    return kolm_seeing*factor


def _seeingResid(r0_500, wavelength, L0, targetSeeing):
    return _vkSeeing(r0_500, wavelength, L0) - targetSeeing


def _r0_500(wavelength, L0, targetSeeing):
    """Returns r0_500 to use to get target seeing."""
    r0_500_max = min(1.0, L0*(1./2.183)**(-0.356)*(wavelength/500.)**1.2)
    r0_500_min = 0.01
    return bisect(
        _seeingResid,
        r0_500_min,
        r0_500_max,
        args=(wavelength, L0, targetSeeing)
    )


def makeAtmosphere(
    airmass,
    rawSeeing,
    wavelength,
    rng,
    kcrit=0.2,
    screen_size=819.2,
    screen_scale=0.1,
    nproc=6
):
    targetFWHM = (
        rawSeeing/galsim.arcsec *
        airmass**0.6 *
        (wavelength/500.0)**(-0.3)
    )

    ud = galsim.UniformDeviate(rng)
    gd = galsim.GaussianDeviate(rng)

    # Use values measured from Ellerbroek 2008.
    altitudes = [0.0, 2.58, 5.16, 7.73, 12.89, 15.46]
    # Elevate the ground layer though.  Otherwise, PSFs come out too correlated
    # across the field of view.
    altitudes[0] = 0.2

    # Use weights from Ellerbroek too, but add some random perturbations.
    weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
    weights = [np.abs(w*(1.0 + 0.1*gd())) for w in weights]
    weights = np.clip(weights, 0.01, 0.8)  # keep weights from straying too far.
    weights /= np.sum(weights)  # renormalize

    # Draw outer scale from truncated log normal
    L0 = 0
    while L0 < 10.0 or L0 > 100:
        L0 = np.exp(gd() * 0.6 + np.log(25.0))
    # Given the desired targetFWHM and randomly selected L0, determine
    # appropriate r0_500
    r0_500 = _r0_500(wavelength, L0, targetFWHM)

    # Broadcast common outer scale across all layers
    L0 = [L0]*6

    # Uniformly draw layer speeds between 0 and max_speed.
    maxSpeed = 20.0
    speeds = [ud()*maxSpeed for _ in range(6)]
    # Isotropically draw directions.
    directions = [ud()*360.0*galsim.degrees for _ in range(6)]

    atmKwargs = dict(
        r0_500=r0_500,
        L0=L0,
        speed=speeds,
        direction=directions,
        altitude=altitudes,
        r0_weights=weights,
        rng=rng,
        screen_size=screen_size,
        screen_scale=screen_scale
    )

    ctx = multiprocessing.get_context('fork')
    atm = galsim.Atmosphere(mp_context=ctx, **atmKwargs)

    r0_500 = atm.r0_500_effective
    r0 = r0_500 * (wavelength/500.0)**(6./5)
    kmax = kcrit/r0

    with ctx.Pool(
        nproc,
        initializer=galsim.phase_screens.initWorker,
        initargs=galsim.phase_screens.initWorkerArgs()
    ) as pool:
        atm.instantiate(pool=pool, kmax=kmax, check='phot')

    return atm


class StarSimulator:
    def __init__(
        self,
        observation,  # from OpSim
        telescope,  # batoid.Optic
        rng=None,
    ):
        if rng is None:
            rng = galsim.BaseDeviate()
        self.observation = observation

        self.wavelength = wavelength_dict[observation['band']]
        self.atm = makeAtmosphere(
            observation['airmass'],
            observation['rawSeeing'],
            self.wavelength,
            rng,
        )

        # and pre-cache a 2nd kick
        psf = self.atm.makePSF(self.wavelength, diam=8.36)
        _ = psf.drawImage(nx=1, ny=1, n_photons=1, rng=rng, method='phot')
        self.second_kick = psf.second_kick

        self.bandpass = galsim.Bandpass(
            f"LSST_{observation['band']}.dat", wave_type='nm'
        )
        
        self.telescope = telescope

        # Develop gnomonic projection from ra/dec to field angle using
        # GalSim TanWCS class.
        q = observation['rotTelPos'] - observation['rotSkyPos']
        cq, sq = np.cos(q), np.sin(q)
        affine = galsim.AffineTransform(cq, -sq, sq, cq)
        self.radecToField = galsim.TanWCS(
            affine,
            self.observation['boresight'],
            units=galsim.radians
        )


    def simStar(self, coord, sed, nphoton, rng, return_photons=False):
        fieldAngle = self.radecToField.toImage(coord)
        # Populate pupil
        r_outer = 8.36/2
        # purposely underestimate inner radius a bit.
        # Rays that miss will be marked vignetted.
        r_inner = 8.36/2*0.58
        ud = galsim.UniformDeviate(rng)
        r = np.empty(nphoton)
        ud.generate(r)
        r *= (r_outer**2 - r_inner**2)
        r += r_inner**2
        r = np.sqrt(r)

        th = np.empty(nphoton)
        ud.generate(th)
        th *= 2*np.pi
        u = r*np.cos(th)
        v = r*np.sin(th)

        # uniformly distribute photon times throughout 30s exposure
        t = np.empty(nphoton)
        ud.generate(t)
        t *= self.observation['exptime']

        # evaluate phase gradients at appropriate location/time
        dku, dkv = self.atm.wavefront_gradient(
            u, v, t, (fieldAngle.x*galsim.radians, fieldAngle.y*galsim.radians)
        )  # output is in nm per m.  convert to radians
        dku *= 1.e-9
        dkv *= 1.e-9

        # add in second kick
        pa = galsim.PhotonArray(nphoton)
        self.second_kick._shoot(pa, rng)
        dku += pa.x*(galsim.arcsec/galsim.radians)
        dkv += pa.y*(galsim.arcsec/galsim.radians)

        # assign wavelengths.
        wavelengths = sed.sampleWavelength(nphoton, self.bandpass, rng)

        # Chromatic seeing.  Scale deflections by (lam/500)**(-0.3)
        dku *= (wavelengths/500)**(-0.3)
        dkv *= (wavelengths/500)**(-0.3)

        # DCR.  dkv is aligned along meridian, so only need to shift in this
        # direction (I think)
        base_refraction = galsim.dcr.get_refraction(
            self.wavelength,
            self.observation['zenith'],
            temperature=self.observation['temperature'],
            pressure=self.observation['pressure'],
            H2O_pressure=self.observation['H2O_pressure'],
        )
        refraction = galsim.dcr.get_refraction(
            wavelengths,
            self.observation['zenith'],
            temperature=self.observation['temperature'],
            pressure=self.observation['pressure'],
            H2O_pressure=self.observation['H2O_pressure'],
        )
        refraction -= base_refraction
        dkv += refraction

        # We're through the atmosphere!  Make a structure that batoid can use
        # now.  Note we're going to just do the sum in the tangent plane
        # coordinates.  This isn't perfect, but almost certainly good enough to
        # still be interesting.
        dku += fieldAngle.x
        dkv += fieldAngle.y
        vx, vy, vz = batoid.utils.fieldToDirCos(dku, dkv, projection='gnomonic')

        # Place rays on entrance pupil - the planar cap coincident with the rim
        # of M1.  Eventually may want to back rays up further so that they can
        # be obstructed by struts, e.g..
        x = u
        y = v
        zPupil = self.telescope["M1"].surface.sag(0, 0.5*self.telescope.pupilSize)
        z = np.zeros_like(x)+zPupil
        # Rescale velocities so that they're consistent with the current
        # refractive index.
        n = self.telescope.inMedium.getN(wavelengths)
        vx /= n
        vy /= n
        vz /= n
        rays = batoid.RayVector.fromArrays(
            x, y, z, vx, vy, vz, t=np.zeros_like(x), w=wavelengths*1e-9, flux=1
        )

        self.telescope.traceInPlace(rays)

        # Now we need to refract the beam into the Silicon.
        silicon = batoid.TableMedium.fromTxt("silicon_dispersion.txt")
        self.telescope['Detector'].surface.refractInPlace(
            rays,
            self.telescope['Detector'].inMedium,
            silicon, coordSys=self.telescope['Detector'].coordSys
        )

        # Need to convert to pixels for galsim sensor object
        # Put batoid results back into photons
        # Use the same array.
        pa.x = rays.x/10e-6
        pa.y = rays.y/10e-6
        pa.dxdz = rays.vx/rays.vz
        pa.dydz = rays.vy/rays.vz
        pa.wavelength = wavelengths
        pa.flux = ~rays.vignetted

        # sensor = galsim.Sensor()
        sensor = galsim.SiliconSensor()
        image = galsim.Image(256, 256)  # hard code for now
        image.setCenter(
            int(np.mean(pa.x[~rays.vignetted])),
            int(np.mean(pa.y[~rays.vignetted]))
        )
        sensor.accumulate(pa, image)

        if return_photons:
            return image, pa
        else:
            return image


if __name__ == '__main__':
    idx = 0 # TODO: figure out how to distribute computation
    survey = Survey()
    observation = survey.get_observation(idx)

    # for now we ensure that camera always has same rotation with respect to optical system
    observation['rotSkyPos'] -= observation['rotTelPos']
    observation['rotTelPos'] -= observation['rotTelPos']

    rng = galsim.BaseDeviate(57721)

    # Could put in chip-to-chip information here.  Omit for the moment.
    # Put in defocus and rotation here.
    factory = LSSTFactory(observation['band'])
    visit_telescope = factory.make_visit_telescope(

        # Extreme aberrations
        M2_amplitude = 1.0,
        camera_amplitude = 1.0,
        M1M3_zer_amplitude = 1.0,

        # Moderate aberrations
        # M2_amplitude = 1./sqrt(30),
        # camera_amplitude = 1./sqrt(30),
        # M1M3_bend_amplitude = 1./sqrt(30),

        rng = rng,
        rotation = observation['rotTelPos'].rad,
        defocus = 1.5e-3
    )
    telescope = visit_telescope.actual_telescope

    simulator = StarSimulator(
        observation,
        telescope,
        rng=rng,
    )

    ras = []
    decs = []
    Ts = []
    for _ in range(100):
        dist = 100*galsim.degrees
        while dist > 1.9*galsim.degrees:
            ra = np.random.uniform(28.0, 32.0)
            dec = np.random.uniform(8.0, 12.0)
            coord = galsim.CelestialCoord(
                ra*galsim.degrees, dec*galsim.degrees
            )
            dist = observation['boresight'].distanceTo(coord)
        ras.append(ra)
        decs.append(dec)
        Ts.append(np.random.uniform(4000, 10000))

    import matplotlib.pyplot as plt
    plt.axis()
    plt.ion()
    plt.show()
    for ra, dec, T in zip(tqdm(ras), decs, Ts):
        coord = galsim.CelestialCoord(
            ra*galsim.degrees, dec*galsim.degrees
        )
        sed = BBSED(T)
        nphoton = int(1e6)
        starImage, starPhotons = simulator.simStar(
            coord, sed, nphoton, rng, return_photons=True
        )

        plt.imshow(starImage.array)
        plt.draw()
        plt.pause(0.1)

        field = simulator.radecToField.toImage(coord)
        zernikes = visit_telescope.get_zernike(field.x, field.y, jmax=11)
        for j in range(4, 12):
            print(f"Z{j:<4}  {zernikes[j]:7.3f}")
