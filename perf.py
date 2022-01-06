from time import time
from train_conv2 import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def perf(model, optimizer, criterion, loader, nbatches):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model.train()
    model.cuda()

    for i, sample in enumerate(loader):
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

        if i%10==0:
            print(f'step: {i}, time: {time()}')

        if i == nbatches:
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=1, type=int)
    parser.add_argument('-m', default=1, type=int)
    parser.add_argument('-d', default=64, type=int)
    parser.add_argument('-w', default=40, type=int)
    parser.add_argument('-b', default=64, type=int)
    args = parser.parse_args()

    model = WaveNet(args.n, args.m, args.d)
    print(f'params: {count_parameters(model)}')
    optimizer =  Adam(model.parameters())
    criterion = nn.MSELoss()
    loader = DataLoader(Data('train'), batch_size=args.b, shuffle=True, num_workers=args.w)
    nbatches = 100
    perf(model, optimizer, criterion, loader, nbatches)