import multiprocessing
import numpy as np
from astropy.table import Table, vstack
from astroquery.gaia import Gaia

import pickle
import galsim

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


q = 10 * galsim.degrees
boresight = galsim.CelestialCoord(288.047168498236 * galsim.degrees, -39.8148859245856 * galsim.degrees)
fp = FocalPlane()
cf = CatalogFactory(fp)
tab = cf.make_catalog(boresight, q)
tab.write('tab2.csv', overwrite=True)