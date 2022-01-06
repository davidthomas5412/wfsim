import numpy as np
from numpy.random import randint
from math import sqrt, ceil
from scipy.optimize import bisect
from astropy.table import Table
from tqdm import tqdm
from astropy.table import Table, vstack
from astroquery.gaia import Gaia
from bisect import bisect_left
from galsim.utilities import lazy_property
from time import time

import argparse
import multiprocessing
import glob
import galsim
import batoid
import pickle
import os

from wfTel import LSSTFactory
import glob

BASE = '/labs/khatrilab/scottmk/david/wfsim'

class Flux:
    # precomputed in transmission.py
    __cache = np.load(os.path.join(BASE, 'transmission_cache.npy'))

    @staticmethod
    def nphotons(sdss_mag_r, T, band='r', exptime=15):
        """
        See flux_notes.pdf for details and assumptions.

        Only r-band is supported.
        """
        if band != 'r':
            raise ValueError('Only support r-band for the time being')
        sdss_mag_zero_r = 24.80
        A_lsst = np.pi * (4.18 ** 2 - 2.558 ** 2)
        
        l = min(bisect_left(Flux.__cache[:,0], T), Flux.__cache.shape[0] - 1)
        r = min(l + 1, Flux.__cache.shape[0] - 1)
        if r == l:
            ratio = Flux.__cache[l,1]
        else:
            # interpolate
            ratio = Flux.__cache[l,1] * (Flux.__cache[r,0] - T) + Flux.__cache[r,1] * (T - Flux.__cache[l,0]) 
            ratio /= (Flux.__cache[r,0] - Flux.__cache[l,0])
        
        nphot = A_lsst * exptime * 10 ** ((sdss_mag_zero_r - sdss_mag_r) / 2.5) * ratio
        return int(nphot)

    @staticmethod
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

class SimRecord:
    """
    Metadata for simulations.
    """
    def __init__(self, directory):
        self.idx = 0
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

        self.table = Table(names=['idx', 'observationId', 'sourceId', 'runId', 'fieldx', 'fieldy', 'posx', 'posy', 'parallactic', 'airmass', 'zenith', 'seed', 'chip', 'intensity', 'temperature'],
                           dtype=['i4', 'i8', 'i8', 'i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'i4', 'str', 'i4', 'f4'])

    def write(self, observationId, sourceId, runId, fieldx, fieldy, posx, posy, parallactic, airmass, zenith, seed, chip, intensity, temperature, starImage, zernikes):
        img_path = os.path.join(self.directory, f'{self.idx}.image')
        zer_path = os.path.join(self.directory, f'{self.idx}.zernike')
        np.save(open(img_path, 'wb'), starImage)
        np.save(open(zer_path, 'wb'), zernikes)
        self.table.add_row([self.idx, observationId, sourceId, runId, fieldx, fieldy, posx, posy, parallactic, airmass, zenith, seed, chip, intensity, temperature])
        self.idx += 1

    def flush(self):
        table_path = os.path.join(self.directory, 'record.csv')
        self.table.write(table_path, overwrite=True)

    @staticmethod
    def combine(in_glob, out_dir):
        tables = []
        for f in glob.glob(in_glob):
            if 'record.csv' in f:
                print(f'reading {f}')
                tables.append(Table.read(f))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'record.csv')
        stacked = vstack(tables)
        stacked.write(out_path, overwrite=True)


class Catalog:
    """
    Queries Gaia catalog.
    """
    @staticmethod
    def query(boresight, parallactic, mag_cutoff=18, verbose=True):
        """
        The sdss_r_mag relationship comes from 
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html
        viewed on 2020/4/7.
        """
        # get wavefront sensor dimensions
        chips = pickle.load(open('chips.pkl', 'rb'))
        wavefront_sensors = {k: v for (k,v) in chips.items() if 'SW' in k}

        # sky to field mapping
        cq, sq = np.cos(parallactic), np.sin(parallactic)
        affine = galsim.AffineTransform(cq, -sq, sq, cq)
        wcs = galsim.TanWCS(
                    affine,
                    boresight,
                    units=galsim.radians
                )

        # stack catalog for each of 8 wavefront chips
        stack = []
        for name, positions in wavefront_sensors.items():
            corners = np.array(positions['corners_field'])
            ra, dec = wcs.toWorld(corners[:,0], corners[:,1], units=galsim.degrees)
            result = Catalog.__chip_table(ra, dec, mag_cutoff, verbose)
            mask = np.logical_or(
                np.logical_or(result['phot_g_mean_mag'].mask, result['phot_bp_mean_mag'].mask), 
                result['phot_rp_mean_mag'].mask)
            result = result[(~mask)]
            result['chip'] = name
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
        AND 1=CONTAINS(POINT('ICRS',ra,dec), {Catalog.__polygon_string(ra, dec)})
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
        SELECT observationId, fieldRA, fieldDec, airmass, altitude, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 100000 LIMIT 1000;
        SELECT observationId, fieldRA, fieldDec, airmass, altitude, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 200000 LIMIT 1000;
        SELECT observationId, fieldRA, fieldDec, airmass, altitude, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 300000 LIMIT 1000;
        SELECT observationId, fieldRA, fieldDec, airmass, altitude, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 400000 LIMIT 1000;
        SELECT observationId, fieldRA, fieldDec, airmass, altitude, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 500000 LIMIT 1000;
        SELECT observationId, fieldRA, fieldDec, airmass, altitude, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 610000 LIMIT 100;
        SELECT observationId, fieldRA, fieldDec, airmass, altitude, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 620000 LIMIT 100;
        SELECT observationId, fieldRA, fieldDec, airmass, altitude, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 630000 LIMIT 100;
        SELECT observationId, fieldRA, fieldDec, airmass, altitude, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 640000 LIMIT 100;
        SELECT observationId, fieldRA, fieldDec, airmass, altitude, filter, rotTelPos, rotSkyPos, skyBrightness, seeingFwhm500 FROM SummaryAllProps WHERE filter IS "r" AND observationId > 650000 LIMIT 100;
        .output stdout
    """
    survey_file = os.path.join(BASE, 'survey2.csv')

    def __init__(self):
        self.table = Table.read(Survey.survey_file, names=['observationId', 'fieldRA', 'fieldDec', 'airmass',\
         'altitude', 'filter', 'rotTelPos', 'rotSkyPos', 'skyBrightness', 'seeingFwhm500', 'skyBrightness2'])

    def get_observation(self, idx):
        row = self.table[idx]
        observation = {
            'observationId': row['observationId'],
            'boresight': galsim.CelestialCoord(row['fieldRA']*galsim.degrees, row['fieldDec']*galsim.degrees),
            'airmass': row['airmass'],
            'rotTelPos': row['rotTelPos']*galsim.degrees,  # zenith measured CCW from up
            'rotSkyPos': row['rotSkyPos']*galsim.degrees,  # N measured CCW from up
            'rawSeeing': row['seeingFwhm500']*galsim.arcsec,
            'skyBrightness': row['skyBrightness'],
            'band': row['filter'],
            'zenith': (90 - row['altitude']) * galsim.degrees,
            'exptime': 15.0,
            'temperature': 293.15,  # K
            'pressure': 69.328,  # kPa
            'H2O_pressure': 1.067,  # kPa
        }

        # for now we ensure that camera always has same rotation with respect to optical system
        observation['rotSkyPos'] -= observation['rotTelPos']
        observation['rotTelPos'] -= observation['rotTelPos']
        observation['parallactic'] = observation['rotTelPos'] - observation['rotSkyPos']
        
        return observation

class Atmosphere:
    @staticmethod
    def _vkSeeing(r0_500, wavelength, L0):
        # von Karman profile FWHM from Tokovinin fitting formula
        kolm_seeing = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
        r0 = r0_500 * (wavelength/500)**1.2
        arg = 1. - 2.183*(r0/L0)**0.356
        factor = np.sqrt(arg) if arg > 0.0 else 0.0
        return kolm_seeing*factor

    @staticmethod
    def _seeingResid(r0_500, wavelength, L0, targetSeeing):
        return Atmosphere._vkSeeing(r0_500, wavelength, L0) - targetSeeing

    @staticmethod
    def _r0_500(wavelength, L0, targetSeeing):
        """Returns r0_500 to use to get target seeing."""
        r0_500_max = min(1.0, L0*(1./2.183)**(-0.356)*(wavelength/500.)**1.2)
        r0_500_min = 0.01
        return bisect(
            Atmosphere._seeingResid,
            r0_500_min,
            r0_500_max,
            args=(wavelength, L0, targetSeeing)
        )

    @staticmethod
    def makeAtmosphere(
        airmass,
        rawSeeing,
        wavelength,
        rng,
        kcrit=0.2,
        screen_size=819.2,
        screen_scale=0.1,
        nproc=1
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
        r0_500 = Atmosphere._r0_500(wavelength, L0, targetFWHM)

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

        if nproc <= 1:
            atm.instantiate(kmax=kmax, check='phot')
        else:
            with ctx.Pool(
                nproc,
                initializer=galsim.phase_screens.initWorker,
                initargs=galsim.phase_screens.initWorkerArgs()
            ) as pool:
                atm.instantiate(pool=pool, kmax=kmax, check='phot')

        return atm


class StarSimulator:
    wavelength_dict = dict(
        u=365.49,
        g=480.03,
        r=622.20,
        i=754.06,
        z=868.21,
        y=991.66
    )

    def __init__(
        self,
        observation,  # from OpSim
        rng=None
    ):
        if rng is None:
            rng = galsim.BaseDeviate()
        self.observation = observation
        self.wavelength = StarSimulator.wavelength_dict[observation['band']]
        self.rng = rng

        self.bandpass = galsim.Bandpass(
            f"LSST_{observation['band']}.dat", wave_type='nm'
        )

        # Develop gnomonic projection from ra/dec to field angle using
        # GalSim TanWCS class.
        q = observation['parallactic']
        cq, sq = np.cos(q), np.sin(q)
        affine = galsim.AffineTransform(cq, -sq, sq, cq)
        self.radecToField = galsim.TanWCS(
            affine,
            self.observation['boresight'],
            units=galsim.radians
        )

    def prepare_atmosphere(self):
        # generate atmosphere
        self.atm = Atmosphere.makeAtmosphere(
            self.observation['airmass'],
            self.observation['rawSeeing'],
            self.wavelength,
            self.rng,
        )

        # pre-cache a 2nd kick
        psf = self.atm.makePSF(self.wavelength, diam=8.36)
        _ = psf.drawImage(nx=1, ny=1, n_photons=1, rng=self.rng, method='phot')
        self.second_kick = psf.second_kick

    def simStar(self, telescope, coord, sed, nphoton, defocus, rng):
        if not hasattr(self, 'atm'):
            self.prepare_atmosphere()

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

        # uniformly distribute photon times throughout 15s exposure
        t = np.empty(nphoton)
        ud.generate(t)
        t *= self.observation['exptime']

        start_gradient = time()
        # evaluate phase gradients at appropriate location/time
        dku, dkv = self.atm.wavefront_gradient(
            u, v, t, (fieldAngle.x*galsim.radians, fieldAngle.y*galsim.radians)
        )  # output is in nm per m.  convert to radians
        dku *= 1.e-9
        dkv *= 1.e-9
        finish_gradient = time()
        print(f'Gradient: {finish_gradient - start_gradient}')

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

        start_ray = time()
        # Place rays on entrance pupil - the planar cap coincident with the rim
        # of M1.  Eventually may want to back rays up further so that they can
        # be obstructed by struts, e.g..
        x = u
        y = v
        zPupil = telescope["M1"].surface.sag(0, 0.5*telescope.pupilSize)
        z = np.zeros_like(x)+zPupil
        # Rescale velocities so that they're consistent with the current
        # refractive index.
        n = telescope.inMedium.getN(wavelengths*1e-9)
        vx /= n
        vy /= n
        vz /= n
        rays = batoid.RayVector.fromArrays(
            x, y, z, vx, vy, vz, t=np.zeros_like(x), w=wavelengths*1e-9, flux=1
        )

        # Set wavefront sensor focus.
        telescope = telescope.withLocallyShiftedOptic('Detector', (0.0, 0.0, defocus))

        telescope.traceInPlace(rays)

        # Now we need to refract the beam into the Silicon.
        silicon = batoid.TableMedium.fromTxt("silicon_dispersion.txt")
        telescope['Detector'].surface.refractInPlace(
            rays,
            telescope['Detector'].inMedium,
            silicon, coordSys=telescope['Detector'].coordSys
        )
        finish_ray = time()
        print(f'Ray: {finish_ray - start_ray}')

        # Need to convert to pixels for galsim sensor object
        # Put batoid results back into photons
        # Use the same array.
        pa.x = rays.x/10e-6
        pa.y = rays.y/10e-6
        pa.dxdz = rays.vx/rays.vz
        pa.dydz = rays.vy/rays.vz
        pa.wavelength = wavelengths
        pa.flux = ~rays.vignetted

        # Reset telescope focus to neutral.
        telescope = telescope.withLocallyShiftedOptic('Detector', (0.0, 0.0, -defocus))

        start_sensor = time()
        # sensor = galsim.Sensor()
        sensor = galsim.SiliconSensor(nrecalc=1e6)
        image = galsim.Image(256, 256)  # hard code for now
        image.setCenter(int(np.mean(pa.x[~rays.vignetted])), 
            int(np.mean(pa.y[~rays.vignetted])))
        sensor.accumulate(pa, image)
        finish_sensor = time()
        print(f'Sensor: {finish_sensor - start_sensor}')

        pos_x = np.mean(rays.x[~rays.vignetted])
        pos_y = np.mean(rays.y[~rays.vignetted])

        return image, pos_x, pos_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-obs', type=int, default=0)  # observation index, spans 0-5499
    parser.add_argument('-stars', type=int, default=100) # number of stars to use
    parser.add_argument('-atm', type=int, default=2) # number of atmosphere instances to use
    parser.add_argument('-dir', type=str, default='/scratch/users/dthomas5/twostage') # data directory
    args = parser.parse_args()

    # check arguments
    if args.stars <= 0 or args.atm <= 0 or not os.path.exists(args.dir):
        raise ValueError('Check arguments.')

    # set seeds
    rng = galsim.BaseDeviate(args.obs)
    np.random.seed(args.obs)
    
    # get observation
    survey = Survey()
    observation = survey.get_observation(args.obs)
    
    # query corresponding catalog
    start_catalog = time()
    catalog = Catalog.query(observation['boresight'], observation['parallactic'], mag_cutoff=18, verbose=False)
    catalog.write(os.path.join(args.dir, f'catalogs/observation{args.obs}.csv'), overwrite=True) # save for future
    ncat = len(catalog)
    samples = np.random.choice(ncat, args.stars, replace=(ncat < args.stars)) # draw STARS samples
    finish_catalog = time()
    print(f'Catalog length: {len(catalog)}, time: {finish_catalog-start_catalog}')

    # create simulator, telescope factory, and sim records
    simulator = StarSimulator(observation, rng=rng)
    factory = LSSTFactory(observation['band'])
    sr = SimRecord(os.path.join(args.dir, f'records/observation{args.obs}'))
    
    for i,row in enumerate(catalog[samples]):
        
        # check if we should regenerate the atmosphere
        if i % ceil(args.stars / args.atm) == 0:
            start_atmosphere = time()
            simulator.prepare_atmosphere()
            finish_atmosphere = time()
            print(f'Atmosphere time: {finish_atmosphere - start_atmosphere}')
        
        # generate telescope
        amp = sqrt(5.0 / 50) # 5 * noise for 0.3 wave rms
        telescope = factory.make_visit_telescope(M2_amplitude=amp, camera_amplitude=amp,
            M1M3_zer_amplitude=amp, M2_zer_amplitude=amp, rng=rng).actual_telescope

        # get coordinates, sed, nphotons
        coord = galsim.CelestialCoord(row['ra'] * galsim.degrees, row['dec'] * galsim.degrees)
        T = row['teff_val'] if row['teff_val'] else np.random.uniform(4000, 10000)
        sed = Flux.BBSED(T)
        nphoton = Flux.nphotons(max(row['sdss_r_mag'], 14), T) # cutoff mags greater than 14.
        defocus = -1.5e-3 if 'SW0' in row['chip'] else 1.5e-3 # SW0 -> intrafocal, SW1 -> extrafocal
        
        start_star = time()
        starImage, pos_x, pos_y = simulator.simStar(
            telescope, coord, sed, nphoton, defocus, rng)
        finish_star = time()
        print(f'Star nphoton: {nphoton}, time: {finish_star - start_star}')

        # wavefront
        field = simulator.radecToField.toImage(coord)
        zernikes = batoid.analysis.zernikeGQ(telescope, field.x, field.y, factory.wavelength, eps=0.61, reference='chief', jmax=22)
        
        # write
        sr.write(observation['observationId'], row['source_id'], i, field.x, field.y, 
                pos_x, pos_y, observation['parallactic'].rad, observation['airmass'], 
                observation['zenith'].rad, args.obs, row['chip'], starImage.array.sum(), 
                T, starImage.array, zernikes)
    
    sr.flush()
