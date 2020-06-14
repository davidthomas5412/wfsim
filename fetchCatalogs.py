from starSim import *
from wfTel import LSSTFactory

import numpy as np
from astropy.table import Table
from time import time

import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', type=str, default='/scratch/users/dthomas5/twostage') # data directory
    parser.add_argument('-start', type=int, default=5000) # start of observation range
    parser.add_argument('-end', type=int, default=5500) # end of observation range
    args = parser.parse_args()

    # check arguments
    if not os.path.exists(args.dir):
        raise ValueError('Check arguments.')

    # get observation
    survey = Survey()

    for obs in range(args.start, args.end):
        observation = survey.get_observation(obs)
        # query corresponding catalog
        catalog = Catalog.query(observation['boresight'], observation['parallactic'], mag_cutoff=18, verbose=False)
        catalog.write(os.path.join(args.dir, f'catalogs/observation{obs}.csv'), overwrite=True) # save for future
        print(time(), obs)
