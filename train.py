import time
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import utils.visualization as visual
from utils import dataloader
from utils.utils import clip_gradient, adjust_lr
from utils.utils import cross_entropy_loss_RCF
from torch.optim import lr_scheduler
from utils.metrics import Evaluator
from network import UNet
from network.Deeplabv3_plus import deeplabv3
from network import HRNet
from network.PSPNet import Pspnet
from network.ABCNet import ABCNet
from network.CMTFNet import CMTFNet
from network import MCCA
from network import CGNet
from network import ENet
from network import DenseASPP
from network import SegNet
from network import BuildFormer
from network.UANet.UANet_final import UANet_final_Res50
from network import DSNet
from network.UNetFormer import UNetFormer
from network.DBBANet.model import DBBANet
from tqdm import tqdm



def dice_loss(predicted, target):
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)

    dice_coefficient = (2.0 * intersection) / (union + 1e-5)  # 添加一个小的常数以防分母为零

    return 1.0 - dice_coefficient


def train_UNetFormer(train_loader, val_loader, net, Eva_train,Eva_val, criterion, optimizer, iters, opt, epoch):
    global best_iou
    global best_epoch
    epoch_loss = 0
    length = 0
    net.train(True)
    st = time.time()
    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda()
        Y = mask.cuda()
        optimizer.zero_grad()
        output = net(X)
        loss = criterion(output[0], Y.long()) + criterion(output[1], Y.long()) 
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()
        st = time.time()
        pred = output[0].data.cpu().numpy()
        target = Y.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        Eva_train.add_batch(target, pred)
        length = length + 1

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    train_loss = epoch_loss / length

    print(
        'Epoch [%d/%d], Loss: %.4f, \n[Training] IOU: %.4f, F1: %.4f' % (
            epoch, opt.epoch, \
            train_loss, \
            IOU[1],F1))
    print("Strat validing!")

    net.train(False)
    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda()
            Y = mask.cuda()
            output = net(X)
            pred = output.data.cpu().numpy()
            target = Y.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            Eva_val.add_batch(target, pred)
    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()

    print(
        'Epoch [%d/%d], \n[Validing] IOU: %.4f, F1: %.4f' % (
            epoch, opt.epoch, \
            IOU[1], F1))
    new_iou = IOU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best %s Model Iou :%.4f; F1 : %.4f; Best epoch : %d' % (opt.model_name, IOU[1], F1, best_epoch))
        torch.save(best_net, opt.save_path + '/' + opt.model_name + '_' + opt.data_name + '_segmentation.pth')
    print('%s Model Iou :%.4f; Best IoU is :%.4f, Best epoch is :%d' % (opt.model_name, new_iou, best_iou, best_epoch))

    return train_loss, new_iou



def train_UANet(train_loader, val_loader, net, Eva_train,Eva_val, criterion, optimizer, iters, opt, epoch):
    global best_iou
    global best_epoch
    epoch_loss = 0
    length = 0
    net.train(True)
    st = time.time()
    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda()
        Y = mask.cuda()
        optimizer.zero_grad()
        output = net(X)
        loss = criterion(output[0], Y.long()) + criterion(output[1], Y.long()) +\
        criterion(output[2], Y.long()) +criterion(output[3], Y.long()) +criterion(output[4], Y.long())
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()
        st = time.time()
        pred = output[4].data.cpu().numpy()
        target = Y.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        Eva_train.add_batch(target, pred)
        length = length + 1

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    train_loss = epoch_loss / length

    print(
        'Epoch [%d/%d], Loss: %.4f, \n[Training] IOU: %.4f, F1: %.4f' % (
            epoch, opt.epoch, \
            train_loss, \
            IOU[1],F1))
    print("Strat validing!")

    net.train(False)
    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda()
            Y = mask.cuda()
            output = net(X)
            pred = output[4].data.cpu().numpy()
            target = Y.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            Eva_val.add_batch(target, pred)
    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()
    print(
        'Epoch [%d/%d],\n[Validing] IOU: %.4f, F1: %.4f' % (
            epoch, opt.epoch, \
            IOU[1], F1))
    new_iou = IOU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best %s Model Iou :%.4f; F1 : %.4f; Best epoch : %d' % (opt.model_name, IOU[1], F1, best_epoch))
        torch.save(best_net, opt.save_path + '/' + opt.model_name + '_' + opt.data_name + '_segmentation.pth')
    print('%s Model Iou :%.4f; Best IoU is :%.4f, Best epoch is :%d' % (opt.model_name, new_iou, best_iou, best_epoch))

    return train_loss, new_iou


def train_DSNet(train_loader, val_loader, net, Eva_train,Eva_val, criterion, optimizer, iters, opt, epoch):
    global best_iou
    global best_epoch
    epoch_loss = 0
    length = 0
    net.train(True)
    st = time.time()
    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda()
        Y = mask.cuda()
        # print(Y.size())
        optimizer.zero_grad()
        output = net(X)
        loss = criterion(output[0], Y.long()) + criterion(output[1], Y.long()) +criterion(output[2], Y.long()) +\
        criterion(F.interpolate(output[3][0], Y.size()[1:], mode='bilinear', align_corners=True), Y.long()) +\
        criterion(F.interpolate(output[3][1], Y.size()[1:], mode='bilinear', align_corners=True), Y.long()) + \
        criterion(F.interpolate(output[3][2], Y.size()[1:], mode='bilinear', align_corners=True), Y.long()) +\
        criterion(F.interpolate(output[3][3], Y.size()[1:], mode='bilinear', align_corners=True), Y.long()) + \
        criterion(F.interpolate(output[3][4], Y.size()[1:], mode='bilinear', align_corners=True), Y.long()) 
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()
        st = time.time()
        pred = output[0].data.cpu().numpy()
        target = Y.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        Eva_train.add_batch(target, pred)
        length = length + 1

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    train_loss = epoch_loss / length

    print(
        'Epoch [%d/%d], Loss: %.4f, \n[Training] IOU: %.4f, F1: %.4f' % (
            epoch, opt.epoch, \
            train_loss, \
            IOU[1],F1))
    print("Strat validing!")

    net.train(False)
    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda()
            Y = mask.cuda()
            output = net(X)
            pred = output[0].data.cpu().numpy()
            target = Y.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            Eva_val.add_batch(target, pred)
    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()
    print(
        'Epoch [%d/%d],\n[Validing] IOU: %.4f, F1: %.4f' % (
            epoch, opt.epoch, \
            IOU[1], F1))
    new_iou = IOU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best %s Model Iou :%.4f; F1 : %.4f; Best epoch : %d' % (opt.model_name, IOU[1], F1, best_epoch))
        torch.save(best_net, opt.save_path + '/' + opt.model_name + '_' + opt.data_name + '_segmentation.pth')
    print('%s Model Iou :%.4f; Best IoU is :%.4f, Best epoch is :%d' % (opt.model_name, new_iou, best_iou, best_epoch))

    return train_loss, new_iou


def train_ABC(train_loader, val_loader, net, Eva_train,Eva_val, criterion, optimizer, iters, opt, epoch):
    global best_iou
    global best_epoch
    epoch_loss = 0
    length = 0
    net.train(True)
    st = time.time()
    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda()
        Y = mask.cuda()
        optimizer.zero_grad()
        output = net(X)
        loss = criterion(output[0], Y.long()) + criterion(output[1], Y.long()) + criterion(output[2], Y.long())
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()
        st = time.time()
        pred = output[0].data.cpu().numpy()
        target = Y.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        Eva_train.add_batch(target, pred)
        length = length + 1

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    train_loss = epoch_loss / length

    print(
        'Epoch [%d/%d], Loss: %.4f, \n[Training] IOU: %.4f, F1: %.4f' % (
            epoch, opt.epoch, \
            train_loss, \
            IOU[1],F1))
    print("Strat validing!")

    net.train(False)
    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda()
            Y = mask.cuda()
            output = net(X)
            pred = output[0].data.cpu().numpy()
            target = Y.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            Eva_val.add_batch(target, pred)
    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()

    print(
        'Epoch [%d/%d], \n[Validing] IOU: %.4f, F1: %.4f' % (
            epoch, opt.epoch, \
            IOU[1], F1))
    new_iou = IOU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best %s Model Iou :%.4f; F1 : %.4f; Best epoch : %d' % (opt.model_name, IOU[1], F1, best_epoch))
        torch.save(best_net, opt.save_path + '/' + opt.model_name + '_' + opt.data_name + '_segmentation.pth')
    print('%s Model Iou :%.4f; Best IoU is :%.4f, Best epoch is :%d' % (opt.model_name, new_iou, best_iou, best_epoch))

    return train_loss, new_iou


def train(train_loader, val_loader, net, Eva_train,Eva_val, criterion, optimizer, iters, opt, epoch):
    global best_iou
    global best_epoch
    epoch_loss = 0
    length = 0
    net.train(True)
    st = time.time()
    for i, sample in enumerate(tqdm(train_loader)):
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda()
        Y = mask.cuda()
        label = Y.data.cpu().numpy()
        # print(np.unique(label))
        optimizer.zero_grad()
        output = net(X)
        # print(output.shape)
        # print(Y.shape)
        loss = criterion(output, Y.long())
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()
        st = time.time()
        pred = output.data.cpu().numpy()
        target = Y.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        Eva_train.add_batch(target, pred)
        length = length + 1
    # Acc = Eva_train.Pixel_Accuracy()
    # Acc_class = Eva_train.Pixel_Accuracy_Class()
    # mIoU = Eva_train.Mean_Intersection_over_Union()
    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    train_loss = epoch_loss / length

    print(
        'Epoch [%d/%d], Loss: %.4f, \n[Training] IOU: %.4f, F1: %.4f' % (
            epoch, opt.epoch, \
            train_loss, \
            IOU[1],F1))
    print("Strat validing!")

    net.train(False)
    net.eval()
    for i, sample in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            inputs, mask = sample['image'], sample['label']
            X = inputs.cuda()
            Y = mask.cuda()
            output = net(X)
            pred = output.data.cpu().numpy()
            target = Y.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            Eva_val.add_batch(target, pred)
    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()

    print(
        'Epoch [%d/%d], \n[Validing] IOU: %.4f, F1: %.4f' % (
            epoch, opt.epoch, \
            IOU[1], F1))
    new_iou = IOU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best %s Model Iou :%.4f; F1 : %.4f; Best epoch : %d' % (opt.model_name, IOU[1], F1, best_epoch))
        torch.save(best_net, opt.save_path + '/' + opt.model_name + '_' + opt.data_name + '_segmentation.pth')
    print('%s Model Iou :%.4f; Best IoU is :%.4f, Best epoch is :%d' % (opt.model_name, new_iou, best_iou, best_epoch))

    return train_loss, new_iou

    
    
class hyper_parameters:
    def __init__(self):
        self.blocks = ['BOTTLENECK','BASIC','BASIC','BASIC']#[Bottleneck,BasicBlock, BasicBlock, BasicBlock]
        self.num_modules = [1, 1, 4, 3]#modules重复数 [重复1次的4个Bottleneck,重复1次的1个BasicBlock,重复4次的1个BasicBlock，重复3次的1个BasicBlock]
        self.num_branches = [1, 2, 3, 4]#分支数
        self.num_blocks = [[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]]
        self.num_channels = [[64], [48, 96], [48, 96, 192], [48, 96, 192, 384]]#各条分支的通道数
        self.fuse_method = ['sum', 'sum', 'sum', 'sum']


if __name__ == '__main__':
    import argparse
    torch.manual_seed(3407)
    # using argparse ,we can execute python file using command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=512, help='training dataset size') # the crop size
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='6', help='train use gpu')
    parser.add_argument('--model_name', type=str, default='UNet',
                        help='the model')
    parser.add_argument('--data_name', type=str, default='FarmSeg',
                        help='the test rgb images root')
    parser.add_argument('--segclass', type=int, default=2, 
                        help='')# we have 3 class here
    parser.add_argument('--save_path', type=str,
                        default='./output/')
    parser.add_argument('--val',type= str,default='test' )
    parser.add_argument('--rename',type = str,default=None)
    opt = parser.parse_args()


    palette = [[0,0,0],[128,0,0],[0,128,0]]

    if opt.data_name == 'FarmSeg':
        opt.train_root = './data/train/img/'
        opt.train_gt = './data/train/label/'
        if opt.val =='val':
            opt.val_root = '/data/val/img/'
            opt.val_gt = './data/val/label/'
        elif opt.val =='test':
            opt.val_root = './data/test/img/'
            opt.val_gt = './data/test/label/'

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    if opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    if opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    if opt.gpu_id == '4':
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        print('USE GPU 4')
    if opt.gpu_id == '5':
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        print('USE GPU 5')
    elif opt.gpu_id == '6':
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        print('USE GPU 6')
    elif opt.gpu_id == '7':
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        print('USE GPU 7')

    train_loader = dataloader.get_loader(opt.train_root, opt.train_gt, palette, opt.batchsize, opt.trainsize, mode='train',
                                         num_workers=1, shuffle=True, pin_memory=True)
        
    val_loader = dataloader.get_loader(opt.val_root, opt.val_gt, palette, opt.batchsize, opt.trainsize, mode='val',
                                       num_workers=1, shuffle=False, pin_memory=True)
    

    if opt.model_name =='UNet':
        net = UNet.UNetWithResnet50Encoder(n_classes=opt.segclass).cuda()
    elif opt.model_name =='DeeplabV3+':
        net = deeplabv3.DeepLab(num_classes=opt.segclass).cuda()
    elif opt.model_name == 'PSPNet':
        net = Pspnet(num_classes=opt.segclass).cuda()
    elif opt.model_name =='HRNet':
        hp = hyper_parameters()
        net = HRNet.HighResolutionNet(hp.blocks, hp.num_channels, hp.num_modules, hp.num_branches, hp.num_blocks,
                                        hp.fuse_method,opt.segclass).cuda()
    elif opt.model_name == 'ABCNet':
        net = ABCNet.ABCNet(3, opt.segclass).cuda()
    elif opt.model_name == 'CMTFNet':
        net = CMTFNet.CMTFNet(num_classes= opt.segclass).cuda()
    elif opt.model_name == 'MCCANet':
        net = MCCA.MCCA(num_class = opt.segclass).cuda()
    elif opt.model_name =='CGNet':
        net = CGNet.Context_Guided_Network(classes = opt.segclass).cuda()
    elif opt.model_name == 'DenseASPP':
        net = DenseASPP.DenseASPP(n_class = opt.segclass).cuda()
    elif opt.model_name =='ENet':
        net =ENet.ENet(n_classes = opt.segclass).cuda()
    elif opt.model_name =='SegNet':
        net = SegNet.SegNet(classes=opt.segclass).cuda()
    elif opt.model_name == 'BuildFormer':
        net = BuildFormer.BuildFormerSegDP(num_classes=opt.segclass).cuda()
    elif opt.model_name =='UANet':
        net = UANet_final_Res50(channel=32, num_classes=opt.segclass).cuda()
    elif opt.model_name =='DSNet':
        net = DSNet.DSNetLocalGuide(classes = opt.segclass).cuda()
    elif opt.model_name =='UNetFormer':
        net = UNetFormer(num_classes = opt.segclass).cuda()
    elif opt.model_name =='DBBANet':
        net = DBBANet(num_class=opt.segclass).cuda()  

    if opt.load is not None:
        net.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    
    if opt.rename is not None:
        opt.model_name = opt.rename

    opt.vis_path = './Vis/'+ opt.model_name + '/' +opt.data_name +'/'
    if not os.path.exists(opt.vis_path):
        os.makedirs(opt.vis_path)
    edge_criterion = cross_entropy_loss_RCF
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(net.parameters(), opt.lr, weight_decay=1e-5)

    

    best_iou = 0
    best_epoch = 0
    loss_train = []
    iou_val = []
    Epoch_all = []
    dice = dice_loss
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    print("Start Training!")
    print("Training ",opt.model_name," on ",opt.data_name," with segclass: ",opt.segclass)
    print("Use ",opt.val," as the Validation data")
    for epoch in range(1, opt.epoch+1):
        Eva_train = Evaluator(opt.segclass)
        Eva_val = Evaluator(opt.segclass)
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        print('Learning rate: ', cur_lr)
        if opt.model_name ==  'ABCNet':
            train_loss, val_iou = train_ABC(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, 0, opt, epoch)
        elif opt.model_name ==  'UANet':
            train_loss, val_iou = train_UANet(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, 0, opt, epoch)
        elif opt.model_name ==  'DSNet':
            train_loss, val_iou = train_DSNet(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, 0, opt, epoch)
        elif opt.model_name ==  'UNetFormer':
            train_loss, val_iou = train_UNetFormer(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, 0, opt, epoch)
        else:
           train_loss, val_iou =  train(train_loader, val_loader, net, Eva_train, Eva_val, criterion, optimizer, 0, opt, epoch)
        Epoch_all.append(epoch)
        loss_train.append(train_loss)
        iou_val.append(val_iou)
    plt.plot(Epoch_all, loss_train, 'r--', label = 'aa')
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.savefig(opt.vis_path + 'train_loss.png', bbox_inches='tight')

    plt.clf()

    plt.plot(Epoch_all, iou_val, 'r--', label = 'aa')
    plt.xlabel('epoch')
    plt.ylabel('val_iou')
    plt.savefig(opt.vis_path + 'val_iou.png', bbox_inches='tight')

    with open(opt.vis_path + 'train_loss.txt', 'w') as f:
        for i in range(len(loss_train)):
            f.write(str(loss_train[i]))
            f.write('\n')
    f.close()
    with open(opt.vis_path + 'val_iou.txt', 'w') as f:
        for i in range(len(iou_val)):
            f.write(str(iou_val[i]))
            f.write('\n')
    f.close()

