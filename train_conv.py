import os
import galsim
import argparse
import numpy as np
from time import time
from astropy.table import Table

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import L1Loss, MSELoss
from torch.utils.data import Dataset, DataLoader

class ConvNet(nn.Module):
    """
    Simple convnet.

    Takes batches of 1x256x256 donut images as input and produces a 1x18
    dimensional wavefront (zernikes 4-21 inclusive).

    """
    def __init__(self):
        super().__init__()
        
        # 1x256x256
        self.star_block = nn.Sequential(
            nn.Conv2d(1,16,3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResBlock(16, 32),
            ResBlock(32, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64), # 64x4x4
            nn.Flatten(), # 1024
            LinearBlock(1024, 128)
        )
        
        self.drop = nn.Dropout(p=0.05)
        
        self.combined_block = nn.Sequential(
            LinearBlock(128+3, 64),
            LinearBlock(64, 18, last=True)
        )

    def forward(self, star, fx, fy, focal):
        star_output = self.drop(self.star_block(star))
        combined_input = torch.cat([star_output, fx, fy, focal], axis=1)
        return self.combined_block(combined_input)

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super().__init__()
        self.last = last
        if last:
            self.net = nn.Linear(in_channels, out_channels)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x0):
        return self.net(x0)

class ResBlock(nn.Module):
    """
    Convnet bulding block.

    Comprised of 3 sequences. The first and second sequence maintain the input tensor
    dimensions. The last downsizes it by 2x2 in the image dimensions. There are
    skip connections around the first and second tensors. 
    
    We also use the 'inception trick' to improve performance. We first convolve
    with 1x1 kernel into 16 channels, then we convovle with a 3x3 kernel into
    the output number of channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1, stride=1, padding=0),
            nn.Conv2d(16, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.second = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1, stride=1, padding=0),
            nn.Conv2d(16, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1, stride=1, padding=0),
            nn.Conv2d(16, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x0):
        x1 = self.first(x0) + x0
        x2 = self.second(x1) + x1
        x3 = self.down(x2)
        return x3


class Data(Dataset):
    """
    star images are 1x256x256
    """

    STAR_TRAIN = 499000
    STAR_DEV = 500
    STAR_TEST = 500
    BLEND_TRAIN = 100004
    BLEND_DEV = 200
    BLEND_TEST = 200

    def __init__(self, mode='train', background=True, data_dir='/labs/khatrilab/scottmk/david/twostage2/aggregate'):
        if mode not in ['train', 'dev', 'test']:
            raise ValueError(f'Incorrect mode {mode}.')

        mode2Len = {
            'train': Data.STAR_TRAIN + Data.BLEND_TRAIN,
            'dev': Data.STAR_DEV + Data.BLEND_DEV,
            'test': Data.STAR_TEST + Data.BLEND_TEST
        }

        self.len = mode2Len[mode]
        self.mode = mode
        self.data_dir = data_dir
        self.star_table = Table.read(os.path.join(data_dir, 'star', 'record.csv'))
        self.blend_table = Table.read(os.path.join(data_dir, 'blend', 'record.csv'))
        self.background = background
        self.sky = np.loadtxt('/labs/khatrilab/scottmk/david/wfsim/sky.csv')
        self.rng = galsim.BaseDeviate()     

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
        return self.len

    def __get_star(self, star_idx):
        star = np.load(os.path.join(self.data_dir, 'star', f'{star_idx}.image')).astype('float32')
        # we use 4-21 inclusive
        wavefront = np.load(os.path.join(self.data_dir, 'star', f'{star_idx}.zernike')).astype('float32')[4:22]
        fx = self.star_table[star_idx]['fieldx']
        fy = self.star_table[star_idx]['fieldy']
        fcl = ('SW0' in self.star_table[star_idx]['chip'])

        if self.background:
            self._apply_sky(star_idx, star)

        output = {'star': torch.from_numpy(star).reshape(1,256,256),
                'fx': torch.FloatTensor([fx]),
                'fy': torch.FloatTensor([fy]),
                'focal': torch.FloatTensor([fcl]),
                'wavefront': torch.from_numpy(wavefront)
                  }
        return output

    def __get_blend(self, blend_idx):
        """
        negative positions
        """
        sub = self.blend_table[self.blend_table['idx'] == blend_idx]
        neighbors = len(sub) - 1
        star = np.load(os.path.join(self.data_dir, 'blend', f'{blend_idx}_0.image')).astype('float32')
        wavefront = np.load(os.path.join(self.data_dir, 'blend', f'{blend_idx}_0.zernike')).astype('float32')[4:22]
        fx = self.blend_table[blend_idx]['fieldx']
        fy = self.blend_table[blend_idx]['fieldy']
        fcl = ('SW0' in self.blend_table[blend_idx]['chip'])

        
        starx = -sub[0]['posx']
        stary = -sub[0]['posy']
        for neighbor in range(1, neighbors + 1):
            neighbor_star = np.load(os.path.join(self.data_dir, 'blend', f'{blend_idx}_{neighbor}.image')).astype('float32')
            neighborx = -sub[neighbor]['posx']
            neighbory = -sub[neighbor]['posy']

            deltax = int(round((neighborx - starx) * 10 ** 5))
            deltay = int(round((neighbory - stary) * 10 ** 5))

            xs = slice(max(deltax, 0), min(256 + deltax, 256))
            ys = slice(max(deltay, 0), min(256 + deltay, 256))
            xn = slice(max(-deltax, 0), min(256 - deltax, 256))
            yn = slice(max(-deltay, 0), min(256 - deltay, 256))
            star[ys,xs] += neighbor_star[yn,xn]

        if self.background:
            self._apply_sky(blend_idx, star)

        output = {'star': torch.from_numpy(star).reshape(1,256,256),
                'fx': torch.FloatTensor([fx]),
                'fy': torch.FloatTensor([fy]),
                'focal': torch.FloatTensor([fcl]),
                'wavefront': torch.from_numpy(wavefront)
                  }
        return output

    def __getitem__(self, idx):
        if idx < 0 or idx > self.len:
            raise ValueError('idx out of bounds.')

        """
        We do the following index mapping:
        (assuming STAR_TRAIN = 499000, STAR_DEV = 500, STAR_TEST = 500,
        BLEND_TRAIN = 99700, BLEND_DEV = 150, BLEND_TEST = 150)

        train 0-499000 -> star 0-499000
        train 499000-598700 -> blend 0-99700
        dev 0-500 -> star 499000-499500
        dev 500-650 -> blend 99700-99850
        test 0-500 -> star 499500-500000
        test 500-650 -> blend 99850-100000
        """
        
        if self.mode == 'train' and idx < Data.STAR_TRAIN:
            return self.__get_star(idx)
        elif self.mode == 'train' and idx >= Data.STAR_TRAIN:
            return self.__get_blend(idx - Data.STAR_TRAIN)
        elif self.mode == 'dev' and idx < Data.STAR_DEV:
            return self.__get_star(idx + Data.STAR_TRAIN)
        elif self.mode == 'dev' and idx >= Data.STAR_DEV:
            return self.__get_blend(idx + Data.BLEND_TRAIN - Data.STAR_DEV)
        elif self.mode == 'test' and idx < Data.STAR_TEST:
            return self.__get_star(idx + Data.STAR_TRAIN + Data.STAR_DEV)
        elif self.mode == 'test' and idx >= Data.STAR_TEST:
            return self.__get_blend(idx + Data.BLEND_TRAIN + Data.BLEND_DEV - Data.STAR_TEST)

def toscalar(var):
    return np.sum(var.data.cpu().numpy())

def train(model, optimizer, criterion, loaders, checkpoint, epochs=10, steps_per_train_loss=None, steps_per_dev_loss=100):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model.train()
    model.cuda()

    best_loss = float('inf')

    for epoch in range(epochs):

        for i, sample in enumerate(loaders['train']):
            star = Variable(sample['star'].cuda())
            wavefront = Variable(sample['wavefront'].cuda())
            fx = Variable(sample['fx'].cuda())
            fy = Variable(sample['fy'].cuda())
            focal = Variable(sample['focal'].cuda())

            # forward pass
            out = model(star, fx, fy, focal)
            loss = criterion(out, wavefront)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps_per_train_loss is not None and i % steps_per_train_loss == 0:
                print(f'time: {time()}, epoch: {epoch}, batch: {i}, train loss: {toscalar(loss)}')

            if steps_per_dev_loss is not None and i % steps_per_dev_loss == 0:
                model.eval()
                dloss_arr = []

                for j, dev_sample in enumerate(loaders['dev']):
                    star = Variable(dev_sample['star'].cuda())
                    wavefront = Variable(dev_sample['wavefront'].cuda())
                    fx = Variable(dev_sample['fx'].cuda())
                    fy = Variable(dev_sample['fy'].cuda())
                    focal = Variable(dev_sample['focal'].cuda())

                    out = model(star, fx, fy, focal)

                    dloss_arr.append(toscalar(criterion(out, wavefront)))
                dloss = np.sum(dloss_arr)

                if dloss < best_loss:
                    best_loss = dloss
                    state = model.state_dict()
                    torch.save(state, checkpoint)

                print(f'time: {time()}, epoch: {epoch}, batch: {i}, mean dev loss: {np.mean(dloss)}, best: {dloss == best_loss}')
                model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-rate', default=0.001, type=float)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-epochs', default=5, type=int)
    parser.add_argument('-train_print', default=200, type=int)
    parser.add_argument('-dev_print', default=200, type=int)
    parser.add_argument('-checkpoint', default='/labs/khatrilab/scottmk/david/twostage2/checkpoints/convnet/model.checkpoint', type=str)
    parser.add_argument('-load', type=str)
    parser.add_argument('-workers', default=40, type=int)
    parser.add_argument('-shuffle', action='store_true')
    args = parser.parse_args()

    model = ConvNet()
    if args.load:
        model.load_state_dict(torch.load(args.load))
    optimizer =  Adam(model.parameters(), lr=args.rate)
    criterion = nn.MSELoss()
    loaders = {
        'train': DataLoader(Data('train'), batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.workers),
        'dev': DataLoader(Data('dev'), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    }
    train(model, optimizer, criterion, loaders, args.checkpoint, steps_per_train_loss=args.train_print, steps_per_dev_loss=args.dev_print, epochs=args.epochs)
