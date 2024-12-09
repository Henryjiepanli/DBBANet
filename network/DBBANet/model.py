from .backbone.resnet import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 3*3 CPDC,with the implement of
class CPDC_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 bias=False):
        super(CPDC_3, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = in_channels  # implement the separable convolution
        self.weight = nn.Parameter(
            torch.Tensor(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # init the parameters

    def reset_parameters(self):
        # the init parameters function
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        assert self.weight.size(2) == 3 and self.weight.size(3) == 3, 'kernel size for cpdc_3 should be 3x3'
        assert self.padding == self.dilation, 'padding for cpdc_3 conv set wrong, must be the same with dilation'
        weights_c = self.weight.sum(dim=[2, 3], keepdim=True)  # dim[2] and dim[3] is the convolution size
        yc = F.conv2d(x, weights_c, stride=self.stride, padding=0, groups=self.groups)
        y = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation,
                     groups=self.groups)
        return y - yc


# Except for the pre-delivery of the network, the rest of the initialization is the same
class APDC_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 bias=False):
        super(APDC_3, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = in_channels  # implement the separable convolution
        self.weight = nn.Parameter(
            torch.Tensor(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # init the parameters

    def reset_parameters(self):
        # the init parameters function
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        assert self.dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
        assert self.weight.size(2) == 3 and self.weight.size(3) == 3, 'kernel size for ad_conv should be 3x3'
        assert self.padding == self.dilation, 'padding for ad_conv set wrong, must be the same with dilation'

        shape = self.weight.shape
        weight = self.weight.view(shape[0], shape[1], -1)
        weights_conv = (weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
        y = F.conv2d(x, weights_conv, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation,
                     groups=self.groups)
        return y


class RPDC_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 bias=False):
        super(RPDC_3, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = in_channels  # implement the separable convolution
        self.weight = nn.Parameter(
            torch.Tensor(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # init the parameters

    def reset_parameters(self):
        # the init parameters function
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        assert self.dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
        assert self.weight.size(2) == 3 and self.weight.size(3) == 3, 'kernel size for rd_conv should be 3x3'
        self.padding = 2 * self.dilation

        shape = self.weight.shape
        if self.weight.is_cuda:
            buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
        else:
            buffer = torch.zeros(shape[0], shape[1], 5 * 5)
        weight = self.weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weight[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        y = F.conv2d(x, buffer, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation,
                     groups=self.groups)
        return y
    

class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=1, padding=0, dilation=1,
                      bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
    # FPN Module for spatial path
class FPN(nn.Module):
    def __init__(self, feature_list=None):
        super(FPN, self).__init__()
        if feature_list is None:
            feature_list = [64, 128, 256, 512, 1024, 2048]
        self.conv0_0 = nn.Conv2d(feature_list[-1], feature_list[-4], 1)
        self.conv0_1 = nn.Conv2d(feature_list[-2], feature_list[-4], 1)
        self.conv0_2 = nn.Conv2d(feature_list[-3], feature_list[-4], 1)
        self.conv0_3 = nn.Conv2d(feature_list[-4], feature_list[-4], 1)
        self.conv3 = nn.Conv2d(feature_list[-4], feature_list[-5], kernel_size=3, stride=1, padding=1)
        self.drop =nn.Dropout2d(0.3)

    def forward(self, out_4x, out_8x, out_16x, out_32x):
        P5 = self.conv0_0(out_32x)
        P4 = self.conv0_1(out_16x) + F.interpolate(P5, scale_factor=2, mode='bilinear', align_corners=False)
        P3 = self.conv0_2(out_8x) + F.interpolate(P4, scale_factor=2, mode='bilinear', align_corners=False)
        P2 = self.conv0_3(out_4x) + F.interpolate(P3, scale_factor=2, mode='bilinear', align_corners=False)
        final_out = self.conv3(P2)
        return self.drop(final_out)

class FPN_Bottlebeck(nn.Module):
    def __init__(self, feature_list=None):
        super(FPN_Bottlebeck, self).__init__()
        if feature_list is None:
            feature_list = [256, 512, 1024, 2048]


        self.conv1_0 = nn.Conv2d(feature_list[-4], feature_list[-4], 1)
        self.conv1_1 = nn.Conv2d(feature_list[-4], feature_list[-4], 1)
        self.conv1_2 = nn.Conv2d(feature_list[-4], feature_list[-4], 1)

        self.conv2_0 = nn.Conv2d(feature_list[-3], feature_list[-4], 1)
        self.conv2_1 = nn.Conv2d(feature_list[-3], feature_list[-4], 1)
        self.conv2_2 = nn.Conv2d(feature_list[-3], feature_list[-4], 1)
        self.conv2_3 = nn.Conv2d(feature_list[-3], feature_list[-4], 1)

        self.conv3_0 = nn.Conv2d(feature_list[-2], feature_list[-4], 1)
        self.conv3_1= nn.Conv2d(feature_list[-2], feature_list[-4], 1)
        self.conv3_2 = nn.Conv2d(feature_list[-2], feature_list[-4], 1)
        self.conv3_3 = nn.Conv2d(feature_list[-2], feature_list[-4], 1)
        self.conv3_4 = nn.Conv2d(feature_list[-2], feature_list[-4], 1)
        self.conv3_5 = nn.Conv2d(feature_list[-2], feature_list[-4], 1)

        self.conv4_0 = nn.Conv2d(feature_list[-1], feature_list[-4], 1)
        self.conv4_1 = nn.Conv2d(feature_list[-1], feature_list[-4], 1)
        self.conv4_2 = nn.Conv2d(feature_list[-1], feature_list[-4], 1)

        self.conv0 = nn.Conv2d(feature_list[-4], feature_list[-4], 1)
        self.conv1 = nn.Conv2d(feature_list[-4], feature_list[-4], 1)
        self.conv2 = nn.Conv2d(feature_list[-4], feature_list[-4], 1)
        self.conv3 = nn.Conv2d(feature_list[-4], feature_list[-4], 1)


        self.conv4 = nn.Conv2d(feature_list[-4], feature_list[-5], kernel_size=3, stride=1, padding=1)

    def forward(self, out_4x_list, out_8x_list, out_16x_list, out_32x_list):
        b1_0 = out_4x_list[0]
        b1_1 = out_4x_list[1]
        b1_2 = out_4x_list[2]
        
        b2_0 = out_8x_list[0]
        b2_1 = out_8x_list[1]
        b2_2 = out_8x_list[2]
        b2_3 = out_8x_list[3]

        b3_0 = out_16x_list[0]
        b3_1 = out_16x_list[1]
        b3_2 = out_16x_list[2]
        b3_3 = out_16x_list[3]
        b3_4 = out_16x_list[4]
        b3_5 = out_16x_list[5]

        b4_0 = out_32x_list[0]
        b4_1 = out_32x_list[1]
        b4_2 = out_32x_list[2]

        b10 = self.conv1_0(b1_0)
        b11 = self.conv1_1(b1_1)
        b12 = self.conv1_2(b1_2)

        b20 = self.conv2_0(b2_0)
        b21 = self.conv2_1(b2_1)
        b22 = self.conv2_2(b2_2)
        b23 = self.conv2_3(b2_3)

        b30 = self.conv3_0(b3_0)
        b31 = self.conv3_1(b3_1)
        b32 = self.conv3_2(b3_2)
        b33 = self.conv3_3(b3_3)
        b34 = self.conv3_4(b3_4)
        b35 = self.conv3_5(b3_5)

        b40 = self.conv4_0(b4_0)
        b41 = self.conv4_0(b4_1)
        b42 = self.conv4_0(b4_2)

        b1 = b10+b11+b12
        b2 = b20+b21+b22+b23
        b3 = b30 +b31+b32+b33+b34+b35
        b4 = b40+b41+b42

        P5 = self.conv3(b4)
        P4 = self.conv2(b3) + F.interpolate(P5, scale_factor=2, mode='bilinear', align_corners=False)
        P3 = self.conv1(b2) + F.interpolate(P4, scale_factor=2, mode='bilinear', align_corners=False)
        P2 = self.conv0(b1) + F.interpolate(P3, scale_factor=2, mode='bilinear', align_corners=False)

        final_out = self.conv4(P2)
        return final_out

class FPN_PSP(nn.Module):
    def __init__(self, feature_list=None):
        super(FPN_PSP, self).__init__()
        if feature_list is None:
            feature_list = [64, 128, 256, 512, 1024, 2048]

        self.conv0_0 = PSPModule(features = feature_list[-1], out_features=feature_list[-4])
        self.conv0_1 = PSPModule(features = feature_list[-2], out_features =feature_list[-4])
        self.conv0_2 = PSPModule(features =feature_list[-3], out_features =feature_list[-4])
        self.conv0_3 = PSPModule(features =feature_list[-4], out_features =feature_list[-4])
        self.conv3 = nn.Conv2d(feature_list[-4], feature_list[-5], kernel_size=3, stride=1, padding=1)
        self.drop =nn.Dropout2d(0.3)

    def forward(self, out_4x, out_8x, out_16x, out_32x):
        P5 = self.conv0_0(out_32x)
        P4 = self.conv0_1(out_16x) + F.interpolate(P5, scale_factor=2, mode='bilinear', align_corners=False)
        P3 = self.conv0_2(out_8x) + F.interpolate(P4, scale_factor=2, mode='bilinear', align_corners=False)
        P2 = self.conv0_3(out_4x) + F.interpolate(P3, scale_factor=2, mode='bilinear', align_corners=False)
        final_out = self.conv3(P2)
        return self.drop(final_out)

class EDB_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(EDB_Block, self).__init__()
        self.cpdc = CPDC_3(in_channels=in_features, out_channels=out_features, kernel_size=3)
        self.relu = nn.ReLU6()
        self.conv1 = nn.Conv2d(in_channels = in_features,out_channels = out_features,kernel_size = 1)

    def forward(self, x):
        Origin = x.clone()
        x = self.cpdc(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x + Origin

class EDB(nn.Module):
    def __init__(self, in_feature,hidden_feature):
        super(EDB, self).__init__()
        # using the 1*1 conv to reduce the channel
        self.edb_1 = EDB_Block(hidden_feature , hidden_feature )
        self.edb_2 = EDB_Block(hidden_feature, hidden_feature)
        self.edb_3 = EDB_Block(hidden_feature , hidden_feature )
        self.conv1 = nn.Conv2d(hidden_feature , hidden_feature, kernel_size=1)

    def forward(self, x):
        x = self.edb_1(x)
        x = self.edb_2(x)
        x = self.edb_3(x)
        x = self.conv1(x)

        return x

class DCB(nn.Module):
    def __init__(self,in_feature,hidden_feature):
        super(DCB,self).__init__()
        self.relu = nn.ReLU6()
        self.conv0 = nn.Conv2d(in_feature,hidden_feature,kernel_size = 1)

        self.conv_5 = nn.Conv2d(hidden_feature,hidden_feature,kernel_size = 3,dilation = 5,stride=1, padding=5)
        self.conv_7 = nn.Conv2d(hidden_feature,hidden_feature,kernel_size = 3,dilation = 7,stride=1, padding=7)
        self.conv_9 = nn.Conv2d(hidden_feature,hidden_feature,kernel_size = 3,dilation = 9,stride=1, padding=9)
        self.conv_11 = nn.Conv2d(hidden_feature,hidden_feature,kernel_size = 3,dilation = 11,stride=1, padding=11)
    def forward(self,x):
        x = self.relu(x)
        x = self.conv0(x)

        x_5 = self.conv_5(x)
        x_7 = self.conv_7(x)
        x_9 = self.conv_9(x)
        x_11 = self.conv_11(x)

        out = x_5 + x_7 + x_9 + x_11
        return out
    # spatial attention module
class SAB(nn.Module):
    def __init__(self,in_feature):
        super(SAB,self).__init__()

        self.relu = nn.ReLU6()
        self.conv0 = nn.Conv2d(in_channels = in_feature,out_channels = 4,kernel_size =1)
        self.conv3 = nn.Conv2d(4,1,kernel_size = 3,stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        Origin = x.clone()
        x = self.relu(x)
        x = self.conv0(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return Origin * x



class Head(nn.Module):
    def __init__(self,in_feature, out_feature,kernel_size=3, stride=1, dilation=1, drop_out=0.):
        super(Head,self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_feature, in_feature, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_feature, bias=False),
            nn.BatchNorm2d(in_feature),
            nn.Conv2d(in_feature, in_feature, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_feature),
            nn.ReLU6(),
            nn.Conv2d(in_feature, out_feature, kernel_size=1, bias=False))
        
    def forward(self,x):

        return self.head(x)
    
class Head_4x(nn.Module):
    def __init__(self,in_feature, out_feature,kernel_size=3, stride=1, dilation=1, drop_out=0.):
        super(Head_4x,self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_feature, 256, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(),
            nn.Conv2d(256, out_feature, kernel_size=1, bias=False))
        
    def forward(self,x):
        
        return self.head(x)
    

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_chan),
                                  nn.ReLU6())
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()
 
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
 
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s
 
    @staticmethod
    def get_module_name():
        return "simam"
 
    def forward(self, x):
 
        b, c, h, w = x.size()
        
        n = w * h - 1
 
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
 
        return x * self.activaton(y)


class Decoder_text(nn.Module):
    def __init__(self):
        super(Decoder_text,self).__init__()

        self.arm16 = AttentionRefinementModule(1024, 128)
        self.arm32 = AttentionRefinementModule(2048, 128)
        self.conv_head32 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU6())
        self.conv_head16 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU6())
        self.conv_avg = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU6())
    def forward(self,feat8,feat16,feat32):
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        feat16_up = F.interpolate(feat16_up,scale_factor = 2, mode='nearest')

        return feat16_up


    
class FeatureFusionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule,self).__init__()
        self.in_channels = in_channels
        self.convblock =nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU6())
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(
            1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

    

class Decoder_spatial_onehead(nn.Module):
    def __init__(self, num_class = 2, feature_list=None, drop_out=0.1):
        super(Decoder_spatial_onehead, self).__init__()
        #we use fpn
        if feature_list is None:
            feature_list=[32,64, 128, 256, 512, 1024, 2048]
        self.head = Head(in_feature=feature_list[-5],out_feature=num_class,drop_out=drop_out)
        self.fpn = FPN(feature_list=feature_list)

    def forward(self, out_4x, out_8x, out_16x, out_32x):
        fpn_out = self.fpn(out_4x, out_8x, out_16x, out_32x)
        seg_out_final = self.head(fpn_out)
        return fpn_out,seg_out_final


class Decoder_spatialV2(nn.Module):
    def __init__(self, num_class = 2, feature_list=None, drop_out=0.1):
        super(Decoder_spatialV2, self).__init__()
        #we use fpn
        if feature_list is None:
            feature_list=[32,64, 128, 256, 512, 1024, 2048]
        self.head_1 = Head(in_feature=feature_list[-5],out_feature=feature_list[-6],drop_out=drop_out)
        self.head_2 = Head(in_feature=feature_list[-6],out_feature=feature_list[-7],drop_out=drop_out)
        self.head_3 = Head(in_feature=feature_list[-7],out_feature=num_class,drop_out=drop_out)
        self.fpn_bottleneck = FPN_Bottlebeck(feature_list=feature_list)

    def forward(self, out_4x_list, out_8x_list, out_16x_list, out_32x_list):
        fpn_out = self.fpn_bottleneck(out_4x_list, out_8x_list, out_16x_list, out_32x_list)
        seg_out_2x= self.head_1(F.interpolate(fpn_out, scale_factor=2, mode='bilinear', align_corners=False))
        seg_out = self.head_2(F.interpolate(seg_out_2x, scale_factor=2, mode='bilinear', align_corners=False))
        seg_out_final = self.head_3(seg_out)
        return seg_out,seg_out_final


class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            # norm_layer(inplane),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            # norm_layer(inplane),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True)
        )
        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = F.interpolate(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output
    
    
class Decoder_edgeV3(nn.Module):
    def __init__(self):
        super(Decoder_edgeV3,self).__init__()
        self.ppm = PSPModule(features=2048,out_features=256)
        self.squeeze_body_edge = SqueezeBodyEdge(256)
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.refine_out4x =  nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.edge_4x_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
    def forward(self,out_32x,out_4x):
        out_32x_ppm =self.ppm(out_32x)
        seg_body, seg_edge = self.squeeze_body_edge(out_32x_ppm)
        refine_out4x = self.refine_out4x(out_4x)
        seg_edge = F.interpolate(seg_edge,scale_factor=8, mode='bilinear', align_corners=False)
        seg_edge = self.edge_4x_fusion(torch.cat([seg_edge, refine_out4x], dim=1))
        seg_edge_out = self.sigmoid(F.interpolate(self.edge_out(seg_edge),scale_factor=4, mode='bilinear', align_corners=False))
    
        seg_out = seg_edge + F.interpolate(seg_body,scale_factor=8, mode='bilinear', align_corners=False)
        aspp = F.interpolate(out_32x_ppm,scale_factor=8, mode='bilinear', align_corners=False)
        edge_feature = torch.cat([aspp, seg_out], dim=1)
        return edge_feature,seg_edge_out

class SEM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SEM, self).__init__()
        self.relu = nn.ReLU(True)
        out_channel_sum = out_channel * 3

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, 1, 0, 1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, 1, 0, 1),
            BasicConv2d(out_channel, out_channel, 3, 1, 4, 4),
            BasicConv2d(out_channel, out_channel, 3, 1, 8, 8),
            BasicConv2d(out_channel, out_channel, 3, 1, 16, 16)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, 1, 0, 1),
            BasicConv2d(out_channel, out_channel, 3, 1, 2, 2),
            BasicConv2d(out_channel, out_channel, 3, 1, 4, 4),
            BasicConv2d(out_channel, out_channel, 3, 1, 8, 8)
        )
        self.conv_cat =BasicConv2d(out_channel_sum, out_channel, 3, 1, 1, 1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1, 1, 0, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class DBBANet(nn.Module):
    def __init__(self,num_class=2):
        super(DBBANet, self).__init__()
        self.num_class = num_class
                
        self.backbone = ResNet50()
        self.decoder_spatial = Decoder_spatial_onehead(num_class=2, feature_list=[32,64, 128, 256, 512, 1024, 2048], drop_out=0.1)
        self.decoder_edge = Decoder_edgeV3()
        self.decoder_text = Decoder_text()
        self.feature_fusion = FeatureFusionModule(in_channels=256,out_channels=128)
        self.head = Head_4x(in_feature = 128 + 512,out_feature =num_class)

    def forward(self,x):
        x1, x2, x3, x4 = self.backbone(x)
        seg_out,_ = self.decoder_spatial(x1, x2, x3, x4)
        edge_out,_ = self.decoder_edge(x4,x1)
        text_out = self.decoder_text(x2,x3,x4)
        seg_atten_out = self.feature_fusion(seg_out,text_out)
        fusion_feature = torch.cat((seg_atten_out,edge_out),dim = 1)
        out = self.head(fusion_feature)
        out = F.interpolate(out,scale_factor = 4,mode='bilinear', align_corners=False)
        return out


if __name__ == '__main__':
    data = torch.rand([2, 3, 512, 512])
    net = DBBANet()
    out = net(data)
    for feature in out :
        print(feature.shape)

