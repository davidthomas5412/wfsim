import galsim
import torch
import numpy as np

from torch.autograd import Variable
from astropy.table import Table
from os.path import join
from torch.utils.data import Dataset, DataLoader

"""
Ended up implementing most of this in the TestFullModel2.ipynb notebook.
"""

class VisitData(Dataset):
    DATA_DIR = '/labs/khatrilab/scottmk/david/twostage2/aggregate/visit'

    @staticmethod
    def loader(obs, workers=48):
        data = VisitData(obs)
        loader = DataLoader(data, batch_size=16, shuffle=False, num_workers=workers)
        return loader

    def __init__(self, obs):
        if obs < 5000 or obs >= 5499:
            raise ValueError('Check obs.')
        self.vis_dir = join(VisitData.DATA_DIR, f'observation{obs}')
        self.table = Table.read(join(self.vis_dir, 'record.csv'))
        self.state = np.load(join(self.vis_dir, 'state.dz'))
        sky = np.loadtxt('/labs/khatrilab/scottmk/david/wfsim/sky.csv')
        self.m_sky = sky[obs % len(sky)]
        self.rng = galsim.BaseDeviate(obs)  

    def __getitem__(self, idx):
        star = np.load(join(self.vis_dir, f'{idx}.image')).astype('float32')

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
                'chip': self.table[idx]['chip'],
                  }
        return output
   

    def apply_sky(self, star):
        """
        Applies sky to star.

        Mutates star.
        """
        # average of ITL + E2V sensors from O’Connor 2019
        gain = (0.69 + 0.94) / 2  

        # from https://www.lsst.org/scientists/keynumbers
        plate_scale = 0.2

        # from https://smtn-002.lsst.io/
        m_zero = 28.13

        # 15 second exposure time
        t_exp = 15
        
        # average of ITL + E2V sensors from O’Connor 2019
        read_noise = (4.7 + 6.1) / 2
            
        sky_level = (t_exp / gain) * 10 ** ((m_zero - self.m_sky) / 2.5) * plate_scale ** 2
        noise = galsim.CCDNoise(self.rng, sky_level=sky_level, gain=gain, read_noise=read_noise)
        img = galsim.Image(star)
        img.addNoise(noise)
        star = img.array

    def __len__(self):
        return len(self.table)

class ChipVisitData(VisitData):
    DATA_DIR = '/labs/khatrilab/scottmk/david/twostage2/aggregate/visit'

    @staticmethod
    def loader(obs, chip, workers=48):
        data = ChipVisitData(obs, chip)
        loader = DataLoader(data, batch_size=16, shuffle=False, num_workers=workers)
        return loader

    def __init__(self, obs, chip):
        super().__init__(obs)
        self.table = Table.read(join(self.vis_dir, 'record.csv'))
        mask = [chip in c for c in self.table['chip']]
        self.table = self.table[mask]
