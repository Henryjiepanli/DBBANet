import torch
import torch.nn as nn
import torch.nn.functional as F
# from network.Res2Net_v1b import *
# from network import resnet
# from network.pvtv2 import pvt_v2_b2
# from network.SST import *
# from .pvtv2 import pvt_v2_b2
from torchvision import models
# from .ViT import ViT



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out



class Decoder(nn.Module):
    def __init__(self, num_classes,in_ch,low_level_inplanes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)  #256->48 1*1
        self.bn1 = nn.BatchNorm2d(48) #nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(in_ch + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        # self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)  #插值上采样
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder_v3(nn.Module):
    def __init__(self, num_classes,in_ch,low_level_inplanes):
        super(Decoder_v3, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)  #256->48 1*1
        self.bn1 = nn.BatchNorm2d(48) #nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(in_ch + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.enchance_1 = HMU(in_ch, num_groups=6, hidden_dim=32)
        self.enchance_2 = HMU(low_level_inplanes, num_groups=6, hidden_dim=32)
        # self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.enchance_1(low_level_feat)
        x = self.enchance_2(x)
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)  #插值上采样
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
            解释  :
                bmm : 实现batch的叉乘
                Parameter：绑定在层里，所以是可以更新的
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DARNetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DARNetHead, self).__init__()
        inter_channels = out_channels
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        sa_feat = self.conv5a(x)
        sa_feat = self.sa(sa_feat)
        sa_feat = self.conv51(sa_feat)

        sc_feat = self.conv5c(x)
        sc_feat = self.sc(sc_feat)
        sc_feat = self.conv52(sc_feat)

        # 两个注意力是相加的
        feat_sum = sa_feat + sc_feat

        output = self.dropout(feat_sum)
        return output


class FEM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FEM, self).__init__()
        self.relu = nn.ReLU(True)
        out_channel_sum = out_channel * 3

        self.branch0 = nn.Sequential(
           BasicConv2d(in_channel, out_channel, 1, 1, 0, 1),
        )
        self.branch1 = nn.Sequential(
           BasicConv2d(in_channel, out_channel, 1, 1, 0, 1),
           BasicConv2d(out_channel, out_channel, 3, 1, 4, 4),
           BasicConv2d(out_channel, out_channel, 3, 1, 8, 8),
           BasicConv2d(out_channel, out_channel, 3, 1, 16, 16),
           BasicConv2d(out_channel, out_channel, 3, 1, 32, 32)
        )
        self.branch2 = nn.Sequential(
           BasicConv2d(in_channel, out_channel, 1, 1, 0, 1),
           BasicConv2d(out_channel, out_channel, 3, 1, 2, 2),
           BasicConv2d(out_channel, out_channel, 3, 1, 4, 4),
           BasicConv2d(out_channel, out_channel, 3, 1, 8, 8),
           BasicConv2d(out_channel, out_channel, 3, 1, 16, 16)
        )
        self.conv_cat = BasicConv2d(out_channel_sum, out_channel, 3, 1, 1, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), dim=1))

        x = self.relu(x_cat)
        return x

class FEM_v2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FEM_v2, self).__init__()
        self.relu = nn.ReLU(True)
        out_channel_sum = out_channel * 3

        self.branch0 = nn.Sequential(
           BasicConv2d(in_channel, out_channel, 3, 1, 1, 1),
        )
        self.branch1 = nn.Sequential(
           BasicConv2d(in_channel, out_channel, 1, 1, 0, 1),
           BasicConv2d(out_channel, out_channel, 3, 1, 4, 4),
           BasicConv2d(out_channel, out_channel, 3, 1, 8, 8),
           BasicConv2d(out_channel, out_channel, 3, 1, 16, 16),
           BasicConv2d(out_channel, out_channel, 3, 1, 32, 32)
        )
        self.branch2 = nn.Sequential(
           BasicConv2d(in_channel, out_channel, 1, 1, 0, 1),
           BasicConv2d(out_channel, out_channel, 3, 1, 2, 2),
           BasicConv2d(out_channel, out_channel, 3, 1, 4, 4),
           BasicConv2d(out_channel, out_channel, 3, 1, 8, 8),
           BasicConv2d(out_channel, out_channel, 3, 1, 16, 16)
        )
        self.conv_cat = BasicConv2d(out_channel_sum, out_channel, 3, 1, 1, 1)
        self.res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), dim=1))

        x = self.relu(self.res(x) + x_cat)
        return x

class DRM(nn.Module):
    def __init__(self, channel):
        super(DRM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1

class DRM_v2(nn.Module):
    def __init__(self, channel):
        super(DRM_v2, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(4 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3, x4):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample6(self.conv_upsample2(self.upsample(self.upsample(self.upsample(x1))))) \
               * self.conv_upsample7(self.conv_upsample3(self.upsample(self.upsample(x2)))) * self.conv_upsample8(self.upsample(x3)) * x4

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x4_2 = torch.cat((x4_1, self.conv_upsample9(self.upsample(x3_2))), 1)

        x1 = self.conv4(x4_2)

        return x1
class DRM_v3(nn.Module):
    def __init__(self, channel):
        super(DRM_v3, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x1)) \
               * self.conv_upsample3(x2) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1


class sa_layer(nn.Module):
    def __init__(self, channel, groups=8):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class HMU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = BasicConv2d(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = BasicConv2d(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = BasicConv2d(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = BasicConv2d(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []

        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(3, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(3, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(2, dim=1))

        out = torch.cat([o[0] for o in outs], dim=1)
        gate = self.gate_genator(torch.cat([o[-1] for o in outs], dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)

class MSCA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)
        return wei


class FBEM(nn.Module):
    def __init__(self, channel, num_classes):
        super(FBEM, self).__init__()
        self.prob = nn.Softmax(dim=1)
        self.back_enhance = FEM(channel,channel)
        self.fore_enhance = FEM(channel,channel)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.out = nn.Conv2d(channel, num_classes,1)

    def forward(self, map, feature):
        map = F.interpolate(map, size= feature.size()[2:], mode='bilinear', align_corners=True)
        seg = self.prob(map)
        back = seg[:,0,:,:].unsqueeze(1)
        fore = seg[:,1,:,:].unsqueeze(1)
        feature_back = self.back_enhance(back*feature)
        feature_fore = self.fore_enhance(fore*feature)
        out_feature = self.alpha*feature_fore - self.beta* feature_back + feature
        out = self.out(out_feature)
        return out


class FBEM_v2(nn.Module):
    def __init__(self, channel):
        super(FBEM_v2, self).__init__()
        self.prob = nn.Softmax(dim=1)
        self.back_enhance = FEM(channel,channel)
        self.fore_enhance = FEM(channel,channel)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.out = nn.Conv2d(3*channel, channel,1)

    def forward(self, map, feature, size):
        ori_size = feature.size()[2:]
        map = F.interpolate(map, size= size, mode='bilinear', align_corners=True)
        feature = F.interpolate(feature, size= size, mode='bilinear', align_corners=True)
        seg = self.prob(map)
        back = seg[:,0,:,:].unsqueeze(1)
        fore = seg[:,1,:,:].unsqueeze(1)
        feature_back = self.back_enhance(back*feature)
        feature_fore = self.fore_enhance(fore*feature)
        out = torch.cat((feature_back,feature_fore,feature),dim=1)
        out = F.interpolate(self.out(out), size= ori_size, mode='bilinear', align_corners=True)
        return out


class FBEM_v3(nn.Module):
    def __init__(self, channel):
        super(FBEM_v3, self).__init__()
        self.prob = nn.Softmax(dim=1)
        self.back_enhance = FEM(channel,channel)
        self.fore_enhance = FEM(channel,channel)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.out = nn.Conv2d(5*channel, channel,1)
        self.sig = nn.Sigmoid()
        

    def forward(self, map, feature, size):
        ori_size = feature.size()[2:]
        map = F.interpolate(map, size= size, mode='bilinear', align_corners=True)
        feature = F.interpolate(feature, size= size, mode='bilinear', align_corners=True)
        seg = self.prob(map)
        back = seg[:,0,:,:].unsqueeze(1)
        fore = seg[:,1,:,:].unsqueeze(1)
        feature_back = self.back_enhance(back*feature)
        feature_fore = self.fore_enhance(fore*feature)

        avg_back = torch.mean(feature_back, dim=1, keepdim=True)
        max_back, _ = torch.max(feature_back, dim=1, keepdim=True)
        # fusion_back = torch.cat([avg_back, max_back], dim=1)

        avg_fore = torch.mean(feature_fore, dim=1, keepdim=True)
        max_fore, _ = torch.max(feature_fore, dim=1, keepdim=True)
        # fusion_fore = torch.cat([avg_fore, max_fore], dim=1)
        f_back_mean = feature_back * self.sig(avg_back)
        f_back_max = feature_back * self.sig(max_back)
        f_fore_mean = feature_back * self.sig(avg_fore)
        f_fore_max = feature_back * self.sig(max_fore)
        fusion = torch.cat((f_back_mean,f_back_max,f_fore_mean,f_fore_max,feature),dim=1)
        fusion = F.interpolate(self.out(fusion), size= ori_size, mode='bilinear', align_corners=True)
        return fusion

class FBEM_v6(nn.Module):
    def __init__(self, channel, num_classes):
        super(FBEM_v6, self).__init__()
        self.prob = nn.Softmax(dim=1)
        self.out = nn.Conv2d((num_classes+1)*channel, channel, 1)
        self.num_classes = num_classes
        

    def forward(self, map, feature, size):
        ori_size = feature.size()[2:]
        map = F.interpolate(map, size= size, mode='bilinear', align_corners=True)
        feature = F.interpolate(feature, size= size, mode='bilinear', align_corners=True)
        seg = self.prob(map)
        feature_cat = feature
        for i in range(self.num_classes):
            class_feature = seg[:,i,:,:].unsqueeze(1)
            class_feature = class_feature*feature+feature
            feature_cat = torch.cat((feature_cat,class_feature),dim=1)
        fusion = F.interpolate(self.out(feature_cat), size= ori_size, mode='bilinear', align_corners=True)
        return fusion


class FBEM_v5(nn.Module):
    def __init__(self, channel):
        super(FBEM_v5, self).__init__()
        self.prob = nn.Softmax(dim=1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.out = nn.Conv2d(3*channel, channel,1)

    def forward(self, map, feature, size):
        ori_size = feature.size()[2:]
        map = F.interpolate(map, size= size, mode='bilinear', align_corners=True)
        feature = F.interpolate(feature, size= size, mode='bilinear', align_corners=True)
        seg = self.prob(map)
        back = seg[:,0,:,:].unsqueeze(1)
        fore = seg[:,1,:,:].unsqueeze(1)
        feature_back = self.alpha*back*feature + feature
        feature_fore = self.alpha*fore*feature + feature
        out = torch.cat((feature_back,feature_fore,feature),dim=1)
        out = F.interpolate(self.out(out), size= ori_size, mode='bilinear', align_corners=True)
        return out

class FBEM_v4(nn.Module):
    def __init__(self, channel):
        super(FBEM_v4, self).__init__()
        self.prob = nn.Softmax(dim=1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.out = nn.Conv2d(5*channel, channel,1)
        self.sig = nn.Sigmoid()
        

    def forward(self, map, feature, size):
        ori_size = feature.size()[2:]
        map = F.interpolate(map, size= size, mode='bilinear', align_corners=True)
        feature = F.interpolate(feature, size= size, mode='bilinear', align_corners=True)
        seg = self.prob(map)
        back = seg[:,0,:,:].unsqueeze(1)
        fore = seg[:,1,:,:].unsqueeze(1)
        feature_back = back*feature
        feature_fore = fore*feature

        avg_back = torch.mean(feature_back, dim=1, keepdim=True)
        max_back, _ = torch.max(feature_back, dim=1, keepdim=True)
        # fusion_back = torch.cat([avg_back, max_back], dim=1)

        avg_fore = torch.mean(feature_fore, dim=1, keepdim=True)
        max_fore, _ = torch.max(feature_fore, dim=1, keepdim=True)
        # fusion_fore = torch.cat([avg_fore, max_fore], dim=1)
        f_back_mean = feature_back * self.sig(avg_back) + feature
        f_back_max = feature_back * self.sig(max_back) + feature
        f_fore_mean = feature_fore * self.sig(avg_fore) + feature
        f_fore_max = feature_fore * self.sig(max_fore) + feature
        fusion = torch.cat((f_back_mean,f_back_max,f_fore_mean,f_fore_max,feature),dim=1)
        fusion = F.interpolate(self.out(fusion), size= ori_size, mode='bilinear', align_corners=True)
        return fusion


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel//subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        else:
            raise Exception("Invalid Channel")

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y


class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)

    def forward(self, x, y):
        # reverse guided block
        y = -1 * (torch.sigmoid(y)) + 1

        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y

class DA_vgg_v29(nn.Module):
    def __init__(self, channel=64, num_classes =1):
        super(DA_vgg_v29, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 64
        self.down2 = vgg16_bn.features[12:22]  # 64
        self.down3 = vgg16_bn.features[22:32]  # 64
        self.down4 = vgg16_bn.features[32:42]  # 64
        self.Translayer_x = nn.Sequential(BasicConv2d(64, channel, 3,2,1))
        self.Translayer_0 = nn.Sequential(BasicConv2d(128, channel, 1))
        self.Translayer_1 = nn.Sequential(BasicConv2d(256, channel, 1),FEM(channel,channel))
        self.Translayer_2 = nn.Sequential(BasicConv2d(512, channel, 1),FEM(channel,channel))
        self.Translayer_3 = nn.Sequential(BasicConv2d(512, channel, 1),FEM(channel,channel))
        self.drm = DRM(channel)
        self.fusion_process = FEM(channel,channel)
        self.out = Decoder(num_classes,channel,channel)
        self.fbem_1 = FBEM_v6(channel,num_classes)
        self.fbem_2 = FBEM_v6(channel,num_classes)
        self.fusion_head = nn.Sequential(
            nn.Conv2d(channel, channel,
                      kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel,
                      kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.layer_head = nn.Sequential(
            nn.Conv2d(channel, channel,
                      kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel,
                      kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.final_out = Decoder(num_classes,channel,channel)
    def forward(self, x):
        size = x.size()[2:]
        x = self.inc(x)
        layer0 = self.down1(x)
        layer1 = self.down2(layer0)
        layer2 = self.down3(layer1)
        layer3 = self.down4(layer2)
        x= self.Translayer_x(x)
        layer0 = self.Translayer_0(layer0)
        layer1 = self.Translayer_1(layer1)
        layer2 = self.Translayer_2(layer2)
        layer3 = self.Translayer_3(layer3)
        fusion = self.drm(layer3,layer2,layer1)
        out = self.out(layer3, fusion)
        layer3 = self.layer_head(layer3)
        fusion = self.fusion_head(fusion)
        feature =self.fbem_1(out,layer3,layer3.size()[2:])
        fusion =self.fbem_2(out,fusion,layer3.size()[2:])
        final_out = self.final_out(feature,fusion)


        return F.interpolate(out, size= size, mode='bilinear', align_corners=True), F.interpolate(final_out, size= size, mode='bilinear', align_corners=True)


if __name__ == '__main__':
    input = torch.randn(1,3,512,512).cuda()
    model = DA_vgg_v29(channel=64, num_classes =2).cuda()
    out = model(input)
    print(out)

        # return 



