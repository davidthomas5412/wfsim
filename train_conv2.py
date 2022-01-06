import os
import sys
import galsim
import argparse
import numpy as np
from time import time
from astropy.table import Table, vstack

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader

class WaveNet(nn.Module):
    """
    Takes batches of 1x256x256 donut images as input and produces a 1x18
    dimensional wavefront (zernikes 4-21 inclusive).

    """
    def __init__(self, n=1, m=1, d=16):
        super().__init__()
        self.donut = DonutNet(n, d)
        self.meta = MetaNet(m, d)

    def forward(self, donut, fx, fy, focal):
        rep = self.donut(donut)
        comb = torch.cat([rep, fx, fy, focal], axis=1)
        return self.meta(comb)

class DonutNet(nn.Module):
    """
    Sub-net that takes batches of 1x256x256 donut images as input and produces a 
    1xd dimensional representation.
    """
    nblocks = 8

    def __init__(self, n, d):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)])
        for i in range(DonutNet.nblocks):
            inchannels = min(d, 2 ** (i + 3))
            outchannels = min(d, 2 ** (i + 3 + 1))
            self.layers.append(DownBlock(inchannels, outchannels, n))
        self.layers.append(nn.Flatten()) 
        
    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x
        
class MetaNet(nn.Module):
    """
    Sub-net that takes batches of 1xd representations and their corresponding 3 metaparameters 
    as input and produces a 1x18 dimensional wavefront.
    """
    ncoefs = 18
    nmetas = 3

    def __init__(self, m, d):
        super().__init__()
        channels = np.logspace(np.log10(MetaNet.ncoefs), np.log10(d), m + 1).astype('int')[::-1]
        channels[0] = d + MetaNet.nmetas
        channels[-1] = MetaNet.ncoefs
        self.layers = nn.ModuleList()
        for i in range(m - 1):
            self.layers.extend([
                nn.Linear(channels[i], channels[i + 1]),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True)])
            if i == 0:
                self.layers.append(nn.Dropout(0.1))
        # final layer: no BN or ReLU
        self.layers.append(nn.Linear(channels[-2], channels[-1])) 
        
    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x

class DownBlock(nn.Module):
    """
    batchsize x width x height x in_channels -> batchsize x width/2 x height/2 x out_channels
    """
    def __init__(self, in_channels, out_channels, n):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n - 1):
            # same width, height
            self.layers.append(SkipBlock(in_channels))
        # down width, height
        self.layers.extend([nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)])

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x

class SkipBlock(nn.Module):
    """
    skip connection
    """
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True))
    
    def forward(self, x):
        dx = self.layers(x)
        return x + dx

def smart_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(m.bias, -bound, bound)

def naive_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight)
        else:
            init.eye_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

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

    STARS = 499999
    BLENDS = 100404

    def __init__(self, mode='train', background=True, badpix=True, dither=True, fixseed=False, ndev=256, ntest=2048, data_dir='/labs/khatrilab/scottmk/david/twostage2/aggregate'):
        if mode not in ['train', 'dev', 'test']:
            raise ValueError(f'Incorrect mode {mode}.')

        self.mode = mode
        if mode == 'train':
            self.len = Data.STARS + Data.BLENDS - ndev - ntest
        elif mode == 'dev':
            self.len = ndev
        else: 
            self.len = ntest

        self.data_dir = data_dir
        self.star_table = Table.read(os.path.join(data_dir, 'star', 'record.csv'))
        self.blend_table = Table.read(os.path.join(data_dir, 'blend', 'record.csv'))

        self.background = background
        self.badpix = badpix
        self.dither = dither
        self.fixseed = fixseed
        
        self.sky = np.loadtxt('/labs/khatrilab/scottmk/david/wfsim/sky.csv')
        self.rng = galsim.BaseDeviate()
        
        self.rng._seed(0)
        np.random.seed(0)
        holdout = np.random.choice(Data.STARS + Data.BLENDS, ndev+ntest, replace=False)
        dev = holdout[:ndev]
        test = holdout[ndev:]
        
        # make index
        if mode == 'train':
            nindex = Data.STARS + Data.BLENDS - ndev - ntest
            filtered = set(range(Data.STARS + Data.BLENDS)) - set(holdout)
            train = list(filtered)
            self.index = np.array(train, dtype='int')
        elif mode == 'dev':
            self.index = dev.astype('int')
        else:
            self.index = test.astype('int')
        
        self.index.sort()

    def __apply_sky(self, idx, star):
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
        if self.fixseed:
            self.rng._seed(idx)
        noise = galsim.CCDNoise(self.rng, sky_level=sky_level, gain=gain, read_noise=read_noise)
        img = galsim.Image(star)
        img.addNoise(noise)
        star = img.array
        
    def __apply_badpix(self, star):
        # ~2 bad pixel per image
        nbadpix = int(np.round(np.random.exponential(scale=2)))
        x = np.random.choice(256, nbadpix)
        y = np.random.choice(256, nbadpix)
        star[x,y] = 0
        
        # ~1 bad column per image
        nbadcol = int(np.round(np.random.exponential(scale=1)))
        badcols = np.random.choice(256, nbadcol)
        for col in badcols:
            start = np.random.choice([0, np.random.randint(0, 256)])
            end = np.random.choice([0, np.random.randint(0, 256)])
            star[start:end, col] = 0
        
    def __apply_dither(self, star):
        dx,dy = np.random.randint(-5,6,size=2)
        star = np.roll(np.roll(star, dx, 0), dy, 1)

    def __len__(self):
        return self.len
    
    def __get_star(self, star_idx):
        star = np.load(os.path.join(self.data_dir, 'star', f'{star_idx}.image')).astype('float32')
        # we use 4-21 inclusive
        wavefront = np.load(os.path.join(self.data_dir, 'star', f'{star_idx}.zernike')).astype('float32')[4:22]
        fx = self.star_table[star_idx]['fieldx']
        fy = self.star_table[star_idx]['fieldy']
        fcl = ('SW0' in self.star_table[star_idx]['chip'])
        
        return star, fx, fy, fcl, wavefront

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
            
        return star, fx, fy, fcl, wavefront



    def __getitem__(self, idx):
        if idx < 0 or idx > self.len:
            raise ValueError('idx out of bounds.')
        
        ind = self.index[idx]
        if ind <= Data.STARS:
            star, fx, fy, fcl, wavefront = self.__get_star(ind)
        else:
            star, fx, fy, fcl, wavefront = self.__get_blend(ind - Data.STARS)
            
        if self.background:
            self.__apply_sky(ind, star)
            
        if self.badpix:
            self.__apply_badpix(star)
            
        if self.dither:
            self.__apply_dither(star)
        
        output = {'star': torch.from_numpy(star).reshape(1,256,256),
                'fx': torch.FloatTensor([fx]),
                'fy': torch.FloatTensor([fy]),
                'focal': torch.FloatTensor([fcl]),
                'wavefront': torch.from_numpy(wavefront),
                'blend': torch.ByteTensor([ind > Data.STARS])
                  }
        return output

    def get_info(self, idx):
        star_idx = idx[idx <= Data.STARS]
        blend_idx = idx[idx >= Data.STARS] - Data.STARS

        star_table = self.star_table[star_idx]
        blend_table = vstack([self.blend_table[self.blend_table['idx'] == idx] for idx in blend_idx])
        
        return star_table, blend_table        

def toscalar(var):
    return np.sum(var.data.cpu().numpy())

def train(model, optimizer, criterion, loaders, checkpoint, log, device,epochs, scheduler, steps_per_train_loss, steps_per_dev_loss):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model.train()
    model.cuda(device)

    best_loss = float('inf')

    for epoch in range(epochs):
        for i, sample in enumerate(loaders['train']):
            star = Variable(sample['star'].cuda(device))
            wavefront = Variable(sample['wavefront'].cuda(device))
            fx = Variable(sample['fx'].cuda(device))
            fy = Variable(sample['fy'].cuda(device))
            focal = Variable(sample['focal'].cuda(device))

            # forward pass
            out = model(star, fx, fy, focal)
            loss = criterion(out, wavefront)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps_per_train_loss is not None and i % steps_per_train_loss == 0:
                log.write(f'epoch: {epoch}, batch: {i}, train loss: {toscalar(loss)}')

            if steps_per_dev_loss is not None and i % steps_per_dev_loss == 0:
                model.eval()
                dloss_arr = []

                for dev_sample in loaders['dev']:
                    star = Variable(dev_sample['star'].cuda(device))
                    wavefront = Variable(dev_sample['wavefront'].cuda(device))
                    fx = Variable(dev_sample['fx'].cuda(device))
                    fy = Variable(dev_sample['fy'].cuda(device))
                    focal = Variable(dev_sample['focal'].cuda(device))

                    out = model(star, fx, fy, focal)

                    dloss_arr.append(toscalar(criterion(out, wavefront)))
                
                dloss = np.mean(dloss_arr)
                if dloss < best_loss:
                    best_loss = dloss
                    state = model.state_dict()
                    torch.save(state, checkpoint)

                log.write(f'epoch: {epoch}, batch: {i}, mean dev loss: {np.mean(dloss)}, best: {dloss == best_loss}')
                model.train()
            
        scheduler.step()
        log.write(f'lr: {scheduler.get_lr()}')
        log.flush()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        message = f'[{time()}] {message}\n'
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-rate', default=0.001, type=float)
    parser.add_argument('-gamma', default=0.65, type=float)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-epochs', default=2, type=int)
    parser.add_argument('-train_print', default=10, type=int)
    parser.add_argument('-dev_print', default=200, type=int)
    parser.add_argument('-load', type=str)
    parser.add_argument('-workers', default=40, type=int)
    parser.add_argument('-n', default=2, type=int)
    parser.add_argument('-m', default=2, type=int)
    parser.add_argument('-d', default=256, type=int)
    parser.add_argument('-init', default='smart', type=str)
    parser.add_argument('-cuda', default=0, type=int)
    parser.add_argument('-model', default='wave', type=str)
    args = parser.parse_args()

    if args.model == 'wave':
        model = WaveNet(args.n, args.m, args.d)
    else:
        model = ConvNet()

    if args.init == 'naive':
        model.apply(naive_init)
    elif args.init == 'smart':
        model.apply(smart_init)
    else:
        raise RuntimeError()

    if args.load:
        model.load_state_dict(torch.load(args.load))
    
    optimizer =  Adam(model.parameters(), lr=args.rate)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=args.gamma)
    criterion = nn.MSELoss()
    loaders = {
        'train': DataLoader(Data('train'), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True),
        'dev': DataLoader(Data('dev'), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    }

    check_dir = '/labs/khatrilab/scottmk/david/twostage2/checkpoints/wavenet/'
    name = f'{model.__class__.__name__}_{args.n}_{args.m}_{args.d}_{args.init}_{args.epochs}_{args.gamma}'
    check = os.path.join(check_dir, name + '.checkpoint')
    log = Logger(os.path.join(check_dir, name + '.log'))

    log.write(args)
    log.write(f'params: {count_parameters(model)}')
    log.write('start')
    log.write(f'lr: {scheduler.get_lr()}')

    train(model, optimizer, criterion, loaders, check, log, f'cuda:{args.cuda}', args.epochs, scheduler, args.train_print, args.dev_print)
    
    log.write('complete')
    log.log.close()
