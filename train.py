"""
从6个节点旋转矩阵(6d)预测15个节点,只需要一个网络,5层SRU
"""

import argparse
import os
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from config import paths
from criterion import  MyLoss1Stage
from datasets import OwnDatasets
from tqdm import tqdm
from net import PoseNet
from visdom import Visdom

parser = argparse.ArgumentParser(description="This is a FDIP of %(prog)s", epilog="This is a epilog of %(prog)s", prefix_chars="-+", fromfile_prefix_chars="@", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-b", "--batch_size",metavar="批次数量", type=int, required=True)
parser.add_argument("-m", "--model", choices=['RNN', 'TCN', 'GCN'], required=True, metavar="模型类型")
parser.add_argument("-f", "--fineturning", action="store_true", help="isFineTurning")
parser.add_argument("-c", "--cuda", action="store_true", help="isCuda")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--save-freq', '-s', default=30, type=int,
                    metavar='N', help='save sample frequency (default: 30)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--visdom', action="store_true", help="visdom")

def train(train_loader, model, criterion, optimizers, epoch, refine=False):
    """
        Run one train epoch
    """
    losses = AverageMeter(refine)

    # switch to train mode
    model.train()

    bar = tqdm(enumerate(train_loader), total = len(train_loader))
    for i, (imu, nn_pose,leaf_jtr, full_jtr, stable, velocity_local, root_ori) in bar:
        imu = imu.transpose(0, 1)
        nn_pose = nn_pose.transpose(0, 1)
        leaf_jtr = leaf_jtr.transpose(0, 1)
        full_jtr = full_jtr.transpose(0, 1)
        if not refine:
            stable = stable.transpose(0, 1)
            velocity_local = velocity_local.transpose(0, 1)
        root_ori = root_ori.transpose(0, 1)

        if args.cuda:
            imu = imu.cuda()
            nn_pose = nn_pose.cuda()
            leaf_jtr = leaf_jtr.cuda()
            full_jtr = full_jtr.cuda()
            if not refine:
                stable = stable.cuda()
                velocity_local = velocity_local.cuda()
            root_ori = root_ori.cuda()
        if args.half:
            imu = imu.half()
            nn_pose = nn_pose.half()
            leaf_jtr = leaf_jtr.half()
            full_jtr = full_jtr.half()
            if not refine:
                stable = stable.half()
                velocity_local = velocity_local.half()
            root_ori = root_ori.half()


        # compute outputn m
        output = model.forward_my((imu, leaf_jtr, full_jtr), refine=refine)            
        target = (nn_pose, stable, velocity_local)
        loss_dict = criterion(output, target, refine)

        bar.set_description(
                f"Train[{epoch}/{args.epochs}] lr={optimizers[0].param_groups[0]['lr']}")
        bar.set_postfix(**{k:v.item() for k,v in loss_dict.items()})
        
        # compute gradient and do Adam step
        [optimizer.zero_grad() for optimizer in optimizers]
        [v.backward() for k, v in loss_dict.items() if k !="contact_prob"]
        [optimizer.step() for optimizer in optimizers]

        losses.update(loss_dict)

    return losses

def validate(val_loader, model, criterion, refine=False):
    """
    Run evaluation
    """
    losses = AverageMeter(refine)

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(val_loader), total = len(val_loader))
    for i, (imu, nn_pose,leaf_jtr, full_jtr, stable, velocity_local, root_ori) in bar:
        imu = imu.transpose(0, 1)
        nn_pose = nn_pose.transpose(0, 1)
        leaf_jtr = leaf_jtr.transpose(0, 1)
        full_jtr = full_jtr.transpose(0, 1)
        if not refine:
            stable = stable.transpose(0, 1)
            velocity_local = velocity_local.transpose(0, 1)
        root_ori = root_ori.transpose(0, 1)
        
        if args.cuda:
            imu = imu.cuda()
            nn_pose = nn_pose.cuda()
            leaf_jtr = leaf_jtr.cuda()
            full_jtr = full_jtr.cuda()
            if not refine:
                stable = stable.cuda()
                velocity_local = velocity_local.cuda()
            root_ori = root_ori.cuda()

        if args.half:
            imu = imu.half()
            nn_pose = nn_pose.half()
            leaf_jtr = leaf_jtr.half()
            full_jtr = full_jtr.half()
            if not refine:
                stable = stable.half()
                velocity_local = velocity_local.half()
            root_ori = root_ori.half()

        # compute output
        with torch.no_grad():
            output = model.forward_my((imu, leaf_jtr, full_jtr), refine=refine)
            target = (nn_pose, stable, velocity_local)
            loss_dict = criterion(output, target, refine)

        bar.set_description("Val")
        bar.set_postfix(**{k:v.item() for k,v in loss_dict.items()})

        # measure accuracy and record loss
        losses.update(loss_dict)


    return losses

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, refine=False):
        self.reset(refine)

    def reset(self, refine):

        self.sum =  {"pose":0}
        self.__avg =  {"pose":0}
        
        self.count = 0

    def update(self, loss_dict):
        for k, v in loss_dict.items():
            self.sum[k] += v.item()
        self.count += 1

    def avg(self):
        for k in self.sum.keys():
            self.__avg[k] = self.sum[k]/self.count
        return self.__avg

def adjust_learning_rate(optimizers, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    for optimizer in optimizers:
        lr = args.lr * (0.8 ** (epoch // 100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def plot_metric(viz, loss_dict, epoch, mode):
    Y, label = [], []
    for k, v in loss_dict.avg().items():     
        Y.append(v)
        label.append(k)  
    viz.line([Y], [epoch], win=mode, opts=dict(title=mode, legend=label), update='append')

def main():
    setup_seed(20)
    global args
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, '==' ,getattr(args, arg))

    # visdom log
    viz = Visdom()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model = PoseNet(isMatrix=False).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True

    train_dataset = OwnDatasets(os.path.join(paths.amass_dir if not args.fineturning else paths.dipimu_dir, "train.pt"), isMatrix=False)
    val_dataset = OwnDatasets(os.path.join(paths.amass_dir if not args.fineturning else paths.dipimu_dir, "veri.pt"), isMatrix=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print(f"train len {len(train_loader)}, vail len{len(val_loader)}")
    # 
    criterion = MyLoss1Stage()
    if args.cuda:
        criterion = criterion.cuda()
    else:
        criterion = criterion.cpu()

    if args.half:
        model.half()
        criterion.half()
    
    optimizerPose1 = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    optimizers = [optimizerPose1]
    # if args.evaluate:
    #     validate(val_loader, model, criterion, args.fineturning)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizers, epoch)
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizers, epoch, args.fineturning)
        plot_metric(viz, train_loss, epoch, "train")
        
        # evaluate on validation set
        validate_loss = validate(val_loader, model, criterion, args.fineturning)
        plot_metric(viz, validate_loss, epoch, "valid")
        
        # remember best prec@1 and save checkpoint
        is_best = True
        if epoch % 10 == 0 :
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}_{}.tar'.format("fineturning" if args.fineturning else "pretrain", epoch)))


if __name__ == '__main__':
    main()