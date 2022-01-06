from train_conv2 import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-rate', default=0.00001, type=float)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-epochs', default=4, type=int)
    parser.add_argument('-train_print', default=10, type=int)
    parser.add_argument('-dev_print', default=200, type=int)
    parser.add_argument('-workers', default=40, type=int)
    parser.add_argument('-cuda', default=0, type=int)
    args = parser.parse_args()

    model = ConvNet()
    base = '/labs/khatrilab/scottmk/david/twostage2/checkpoints/wavenet/'
    model.load_state_dict(torch.load(base + 'refine_conv_8.checkpoint'))
    model.eval().cuda('cuda:0')
    device = 'cuda:0'

    criterion = nn.MSELoss()

    for dev_sample in DataLoader(Data('dev'), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True):
        star = Variable(dev_sample['star'].cuda(device))
        wavefront = Variable(dev_sample['wavefront'].cuda(device))
        fx = Variable(dev_sample['fx'].cuda(device))
        fy = Variable(dev_sample['fy'].cuda(device))
        focal = Variable(dev_sample['focal'].cuda(device))

        out = model(star, fx, fy, focal)

        print(toscalar(criterion(out, wavefront)))

    optimizer =  Adam(model.parameters(), lr=args.rate)
    loaders = {
        'train': DataLoader(Data('train'), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True),
        'dev': DataLoader(Data('dev'), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    }

    check_dir = '/labs/khatrilab/scottmk/david/twostage2/checkpoints/wavenet/'
    name = f'refine_refine_conv_8'
    check = os.path.join(check_dir, name + '.checkpoint')
    log = Logger(os.path.join(check_dir, name + '.log'))

    log.write(args)
    log.write(f'params: {count_parameters(model)}')
    log.write('start')

    scheduler = ExponentialLR(optimizer, gamma=1.0)
    train(model, optimizer, criterion, loaders, check, log, f'cuda:{args.cuda}', args.epochs, scheduler, args.train_print, args.dev_print)
    
    log.write('complete')
    log.log.close()
