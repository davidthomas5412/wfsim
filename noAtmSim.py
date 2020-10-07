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

from starSim import *

class NoAtmSimulator:
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
        self.wavelength = NoAtmSimulator.wavelength_dict[observation['band']]
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
        self.prepare_psf()

    def prepare_psf(self):
        # generate atmosphere
        gd = galsim.GaussianDeviate(self.rng)
        L0 = 0
        while L0 < 10.0 or L0 > 100:
            L0 = np.exp(gd() * 0.6 + np.log(25.0))
        targetFWHM = (
            self.observation['rawSeeing']/galsim.arcsec *
            self.observation['airmass']**0.6 *
            (self.wavelength/500.0)**(-0.3)
        )
        r0_500 = Atmosphere._r0_500(self.wavelength, L0, targetFWHM)
        self.psf = galsim.VonKarman(lam=self.wavelength, r0_500=r0_500)

    def simStar(self, telescope, coord, sed, nphoton, defocus, rng):
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

        # Von Karman PSF
        dku = self.psf.shoot(nphoton).x * (galsim.arcsec/galsim.radians)
        dkv = self.psf.shoot(nphoton).y * (galsim.arcsec/galsim.radians)

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

        dku += fieldAngle.x
        dkv += fieldAngle.y 
        vx, vy, vz = batoid.utils.fieldToDirCos(dku, dkv, projection='gnomonic')

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

        # Need to convert to pixels for galsim sensor object
        # Put batoid results back into photons
        # Use the same array.
        pa = galsim.PhotonArray(nphoton)
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
    parser.add_argument('-dir', type=str, default='/scratch/users/dthomas5/twostage') # data directory
    args = parser.parse_args()

    # check arguments
    if args.stars <= 0 or not os.path.exists(args.dir):
        raise ValueError('Check arguments.')

    # set seeds
    rng = galsim.BaseDeviate(args.obs)
    np.random.seed(args.obs)
    
    # get observation
    survey = Survey()
    observation = survey.get_observation(args.obs)
    
    # query corresponding catalog
    catalog = Table.read(os.path.join(args.dir, f'catalogs/observation{args.obs}.csv')) # save for future
    ncat = len(catalog)
    samples = np.random.choice(ncat, args.stars, replace=(ncat < args.stars)) 
    # draw STARS samples

    # create simulator, telescope factory, and sim records
    simulator = NoAtmSimulator(observation, rng=rng)
    factory = LSSTFactory(observation['band'])
    sr = SimRecord(os.path.join(args.dir, f'records/observation{args.obs}_noatm'))
    
    for i,row in enumerate(catalog[samples]):
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
