import galsim
import torch
import numpy as np

from torch.autograd import Variable
from astropy.table import Table
from os.path import join
from torch.utils.data import Dataset, DataLoader

"""
Includes blends, breaks up by chip.
"""

class VisitData(Dataset):
    DATA_DIR = '/labs/khatrilab/scottmk/david/twostage2/aggregate/visit'
    CHIP2NUM = {
        'R00_SW0': 0,
        'R00_SW1': 1,
        'R04_SW0': 2,
        'R04_SW1': 3,
        'R40_SW0': 4,
        'R40_SW1': 5,
        'R44_SW0': 6,
        'R44_SW1': 7,
    }
    NUM2CHIP = {
        0: 'R00_SW0',
        1: 'R00_SW1',
        2: 'R04_SW0',
        3: 'R04_SW1',
        4: 'R40_SW0',
        5: 'R40_SW1',
        6: 'R44_SW0',
        7: 'R44_SW1',
    }

    def __init__(self, obs):
        if obs < 0 or obs >= 499:
            raise ValueError('Check obs.')
        self.vis_dir = join(VisitData.DATA_DIR, f'observation{5000 + obs}')
        self.table = Table.read(join(self.vis_dir, 'record.csv'))
        self.__setup_neighbors()
        self.state = np.load(join(self.vis_dir, 'state.dz'))
        sky = np.loadtxt('/labs/khatrilab/scottmk/david/wfsim/sky.csv')
        self.m_sky = sky[obs % len(sky)]
        self.rng = galsim.BaseDeviate(obs) 

        # average of ITL + E2V sensors from O’Connor 2019
        gain = (0.69 + 0.94) / 2  

        # from https://www.lsst.org/scientists/keynumbers
        plate_scale = 0.2

        # from https://smtn-002.lsst.io/
        m_zero = 28.13

        # 15 second exposure time
        t_exp = 15
        
        # average of ITL + E2V sensors from O’Connor 2019
        self.read_noise = (4.7 + 6.1) / 2
            
        self.sky_level = (t_exp / gain) * 10 ** ((m_zero - self.m_sky) / 2.5) * plate_scale ** 2
        self.noise = np.sqrt(self.sky_level + self.read_noise ** 2)

    def __setup_neighbors(self):
        if not self.table:
            raise RuntimeError()
        neighbors = dict()
        for row in self.table:
            posx = row['posx']
            posy = row['posy']
            sub = self.table[np.sqrt((self.table['posx'] - posx) ** 2 + (self.table['posy'] - posy) ** 2) < 128*1e-5]
            neighbors[row['idx']] = []
            for subrow in sub:
                if subrow['idx'] == row['idx']:
                    continue
                else:
                    neighbors[row['idx']].append(
                        (subrow['idx'], 
                            int((subrow['posx'] - row['posx']) * 1e5), 
                            int((subrow['posy'] - row['posy']) * 1e5)
                        ))
        self.neighbors = neighbors

    def __getitem__(self, idx):
        star = np.load(join(self.vis_dir, f'{idx}.image')).astype('float32')

        neighbors = self.neighbors[idx]
        for idx, pixx, pixy in neighbors:
            neigh = np.load(join(self.vis_dir, f'{idx}.image')).astype('float32')
            star[max(0,pixy):(pixy + 256), max(0,pixx):(pixx + 256)] += neigh[max(0,-pixy):(-pixy + 256), max(0,-pixx):(-pixx + 256)]


        # we use 4-21 inclusive
        wavefront = np.load(join(self.vis_dir, f'{idx}.zernike')).astype('float32')[4:22]
        fx = self.table[idx]['fieldx']
        fy = self.table[idx]['fieldy']
        fcl = ('SW0' in self.table[idx]['chip'])

        self.apply_sky(star)

        output = {'star': torch.from_numpy(star).reshape(1,256,256),
                'fx': torch.FloatTensor([fx]),
                'fy': torch.FloatTensor([fy]),
                'focal': torch.FloatTensor([fcl]),
                'wavefront': torch.from_numpy(wavefront),
                'chip': torch.FloatTensor([VisitData.CHIP2NUM[self.table[idx]['chip']]]),
                'neighbors': torch.FloatTensor([len(neighbors)])                  }
        return output
   


    def apply_sky(self, star):
        """
        Applies sky to star.

        Mutates star.
        """
        noise = galsim.CCDNoise(self.rng, sky_level=self.sky_level, gain=gain, read_noise=self.read_noise)
        img = galsim.Image(star)
        img.addNoise(noise)
        star = img.array

    def __len__(self):
        return len(self.table)

    def snr(self):
        return self.table['intensity'] / self.noise