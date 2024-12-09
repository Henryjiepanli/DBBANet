import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import dataloader
import numpy as np
from PIL import Image
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

class hyper_parameters:
    def __init__(self):
        self.blocks = ['BOTTLENECK','BASIC','BASIC','BASIC']#[Bottleneck,BasicBlock, BasicBlock, BasicBlock]
        self.num_modules = [1, 1, 4, 3]#modules重复数 [重复1次的4个Bottleneck,重复1次的1个BasicBlock,重复4次的1个BasicBlock，重复3次的1个BasicBlock]
        self.num_branches = [1, 2, 3, 4]#分支数
        self.num_blocks = [[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]]
        self.num_channels = [[64], [48, 96], [48, 96, 192], [48, 96, 192, 384]]#各条分支的通道数
        self.fuse_method = ['sum', 'sum', 'sum', 'sum']


def onehot_to_mask(semantic_map, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    #x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    semantic_map = np.uint8(colour_codes[semantic_map.astype(np.uint8)])
    return semantic_map


        
def test(test_load, net,Eva, opt):

    palette2 = [[0,0,0],[0,128,0]]
    net.train(False)
    net.eval()
    for sample,filename in test_load:
        inputs, mask = sample['image'], sample['label']
        X = inputs.cuda()
        Y = mask.cuda()

        if opt.model_name == 'ABCNet':
            output = net(X)[0]
        elif opt.model_name == 'UANet':
            output = net(X)[4]
        else:
            output = net(X)
        pred = output.data.cpu().numpy()
        target = Y.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        Eva.add_batch(target, pred)
        output = torch.argmax(output,dim = 1)
        for i in range(output.shape[0]):
            probs_array = (torch.squeeze(output[i])).data.cpu().numpy()
            final_mask = onehot_to_mask(probs_array,palette2).astype(np.uint8)
            final_savepath = opt.save_path + '/' + filename[i] + '.png'
            im = Image.fromarray(final_mask)
            # im.putpalette(sort_palette)
            # im =im.convert('P')
            im.save(final_savepath)

    IoU = Eva.Intersection_over_Union()
    F1Score = Eva.F1Score()
    Precision = Eva.Precision()
    Recall = Eva.Recall()
    OA = Eva.OA()

    print('****    [Test]IoU: %.4f, F1score: %.4f, Precision: %.4f, Recall: %.4f, OA:%.4f ****' \
          % ( IoU[1],F1Score,Precision,Recall,OA))
    string_print = 'IOU: %.4f, F1score: %.4f, Precision: %.4f, Recall: %.4f, OA:%.4f'\
                   % (IoU[1],F1Score,Precision,Recall,OA)
    with open(opt.txt_name, 'w') as f:
        f.write(string_print)
        f.write('\n')

def main():
    import argparse
    # using argparse ,we can execute python file using command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=512, help='training dataset size') # the crop size
    parser.add_argument('--gpu_id', type=str, default='7', help='train use gpu')
    parser.add_argument('--model_name', type=str, default='UNet',
                        help='the model')
    parser.add_argument('--data_name', type=str, default='FarmSeg',
                        help='the test rgb images root')
    parser.add_argument('--segclass', type=int, default=2, 
                        help='')# we have 2 class here
    parser.add_argument('--save_path', type=str,
                        default='./output/')
    parser.add_argument('--txt_name',type=str,default= '')

    opt = parser.parse_args()
    opt.save_path ='./'+'result/'+opt.model_name + '/' + opt.data_name + '/'
    opt.txt_path ='./'+ opt.model_name + opt.data_name + '_test_result.txt'
    opt.model_path = './output/' + opt.model_name + '_' + opt.data_name + '_segmentation.pth'
    opt.txt_name ='./TestResult/'+ opt.model_name + opt.data_name + '_test_result.txt'
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    if opt.data_name == 'FarmSeg':
        opt.test_img = './data/test/img/'
        opt.test_gt = './data/test/label/'
    
    palette = [[0,0,0],[128,0,0],[0,128,0]]
    with torch.no_grad():
        Eva = Evaluator(opt.segclass)
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
            
        net.load_state_dict(torch.load(opt.model_path))
    
        test_load = dataloader.get_loader(opt.test_img, opt.test_gt, palette, opt.batchsize, opt.trainsize,  mode ='test',num_workers=2, shuffle=False, pin_memory=True)
        print("****    Start Testing!    ****")
        print("****    Test ",opt.model_name," on ",opt.data_name," with class ",opt.segclass ," ****")
        test(test_load, net,Eva, opt)

if __name__ == '__main__':
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    main()