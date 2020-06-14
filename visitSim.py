from starSim import *
from wfTel import LSSTFactory

import numpy as np
from math import sqrt
from astropy.table import Table
from time import time

import argparse
import batoid
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-obs', type=int, default=5000)  # visit observation index, spans 5000-5499
    parser.add_argument('-dir', type=str, default='/scratch/users/dthomas5/twostage') # data directory
    parser.add_argument('-start', type=int, required=True) # start catalog index
    parser.add_argument('-end', type=int, required=True) # end catalog index
    args = parser.parse_args()

    # check arguments
    if not os.path.exists(args.dir) or args.obs < 5000 or args.obs >= 5500:
        raise ValueError('Check arguments.')

    # set seeds
    rng = galsim.BaseDeviate(args.obs)
    np.random.seed(args.obs)
    
    # get observation
    survey = Survey()
    observation = survey.get_observation(args.obs)
    
    # query corresponding catalog
    cat_path = os.path.join(args.dir, f'catalogs/observation{args.obs}.csv')
    catalog = Table.read(cat_path) # assumes we already have fetched catalog

    # create simulator, telescope factory, and sim records
    simulator = StarSimulator(observation, rng=rng)
    factory = LSSTFactory(observation['band'])
    sr = SimRecord(os.path.join(args.dir, f'records/visit/observation{args.obs}_{args.start}_{args.end}'))
    
    # atmosphere and telescope are fixed in these simulations
    simulator.prepare_atmosphere()
    amp = sqrt(5.0 / 50) # 5 * noise for 0.3 wave rms
    visit_telescope = factory.make_visit_telescope(M2_amplitude=amp, camera_amplitude=amp,
        M1M3_zer_amplitude=amp, M2_zer_amplitude=amp, rng=rng)
    telescope = visit_telescope.actual_telescope
    
    # save double zernike telescope state
    dz = visit_telescope.dz()
    np.save(open(os.path.join(args.dir, f'records/visit/observation{args.obs}_{args.start}_{args.end}.dz2'), 'wb'), dz)    

    for i,row in enumerate(catalog[args.start:args.end]):
        # get coordinates, sed, nphotons
        coord = galsim.CelestialCoord(row['ra'] * galsim.degrees, row['dec'] * galsim.degrees)
        T = row['teff_val'] if row['teff_val'] else np.random.uniform(4000, 10000)
        sed = Flux.BBSED(T)
        nphoton = Flux.nphotons(max(row['sdss_r_mag'], 14), T) # cutoff mags greater than 14.
        defocus = -1.5e-3 if 'SW0' in row['chip'] else 1.5e-3 # SW0 -> intrafocal, SW1 -> extrafocal

        starImage, pos_x, pos_y = simulator.simStar(
            telescope, coord, sed, nphoton, defocus, rng)

        # wavefront
        field = simulator.radecToField.toImage(coord)
        zernikes = batoid.analysis.zernikeGQ(telescope, field.x, field.y, factory.wavelength, eps=0.61, reference='chief', jmax=22)

        # write
        sr.write(observation['observationId'], row['source_id'], args.start + i, field.x, field.y,
                pos_x, pos_y, observation['parallactic'].rad, observation['airmass'],
                observation['zenith'].rad, args.obs, row['chip'], starImage.array.sum(),
                T, starImage.array, zernikes)

    sr.flush()
