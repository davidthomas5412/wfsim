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

class FullModel:
    nzer = 18
    kmax = 21

    def __init__(self, model, true_wavefront=False):
        self.model = model
        self.model.cuda()
        self.true_wavefront = true_wavefront

    def wavefront(self, samples):
        if self.true_wavefront:
            estimate = samples['wavefront']
        else:
            star = Variable(samples['star'].cuda())
            wavefront = Variable(samples['wavefront'].cuda())
            fx = Variable(samples['fx'].cuda())
            fy = Variable(samples['fy'].cuda())
            focal = Variable(samples['focal'].cuda())
            estimate = self.model(star, fx, fy, focal).cpu().detach().numpy()
            
        return estimate

    def state(self, estimates):
        fx = estimates['fx']
        fy = estimates['fy']
        print(fx, fy)
        basis = galsim.zernike.zernikeBasis(FullModel.kmax, fx, fy, R_outer=np.deg2rad(1.90))[1:,1:]
        state = np.zeros((FullModel.nzer-1, FullModel.kmax))
        est = estimates['est']
        for row in range(1,FullModel.nzer):
            state[row-1] = FullModel.regress(basis.T, est[1:,row])
        return state

    def __call__(self, loader):
        n = len(loader.dataset)
        batch_size = loader.batch_size
        estimates = {
            'fx': np.zeros(n),
            'fy': np.zeros(n),
            'est': np.zeros((n, FullModel.nzer))
        }
        for step,samples in enumerate(loader):
            estimate = self.wavefront(samples)
            batch = slice(step * batch_size, (step + 1) * batch_size)
            estimates['fx'][batch] = samples['fx'].numpy().flatten()
            estimates['fy'][batch] = samples['fy'].numpy().flatten()
            estimates['est'][batch] = estimate

        state = self.state(estimates)
        return state

    @staticmethod
    def regress(X, y, lam=0):
        """
        Closed form solution for L2 regularized regression.
        """
        return np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y

class VisitData(Dataset):
    DATA_DIR = '/labs/khatrilab/scottmk/david/twostage2/aggregate/visit'

    @staticmethod
    def loader(obs, workers=48):
        data = VisitData(obs)
        loader = DataLoader(data, batch_size=16, shuffle=False, num_workers=workers)
        return loader

    def __init__(self, obs):
        if obs < 5000 or obs >= 5099:
            raise ValueError('Check obs.')
        self.vis_dir = join(VisitData.DATA_DIR, f'observation{obs}')
        self.table = Table.read(join(self.vis_dir, 'record.csv'))
        self.state = np.load(join(self.vis_dir, 'state.dz'))
        self.sky = np.loadtxt('/labs/khatrilab/scottmk/david/wfsim/sky.csv')
        self.rng = galsim.BaseDeviate()  

    def __getitem__(self, idx):
        star = np.load(join(self.vis_dir, f'{idx}.image')).astype('float32')

        # we use 4-21 inclusive
        wavefront = np.load(join(self.vis_dir, f'{idx}.zernike')).astype('float32')[4:22]
        fx = self.table[idx]['fieldx']
        fy = self.table[idx]['fieldy']
        fcl = ('SW0' in self.table[idx]['chip'])

        self._apply_sky(idx, star)

        output = {'star': torch.from_numpy(star).reshape(1,256,256),
                'fx': torch.FloatTensor([fx]),
                'fy': torch.FloatTensor([fy]),
                'focal': torch.FloatTensor([fcl]),
                'wavefront': torch.from_numpy(wavefront)
                  }
        return output
   

    def _apply_sky(self, idx, star):
        """
        Applies sky to star.

        Mutates star.
        """
        m_sky = self.sky[idx % len(self.sky)]

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
            
        sky_level = (t_exp / gain) * 10 ** ((m_zero - m_sky) / 2.5) * plate_scale ** 2
        noise = galsim.CCDNoise(self.rng, sky_level=sky_level, gain=gain, read_noise=read_noise)
        img = galsim.Image(star)
        img.addNoise(noise)
        star = img.array

    def __len__(self):
        return len(self.table)