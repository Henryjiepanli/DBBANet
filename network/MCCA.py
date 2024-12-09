import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)


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
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class CCA(nn.Module):
    """
    CCA Block
    """

    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=256, F_x=256)
        self.nConvs = _make_nConv(in_channels * 2, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        # up = self.up(x)
        size = skip_x.size()
        up = Upsample(x, size[2:])
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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
        self.upsample8x_op = nn.UpsamplingBilinear2d(scale_factor=8)

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


class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x):
        b0 = self.relu(self.bn1(self.conv1(x)))
        b = self.maxpool(b0)
        b1 = self.layer1(b)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return [b1, b2, b3, b4]


class MCCA(nn.Module):  # Inherit from nn.Module instead of CVModule
    def __init__(self, num_class=2, backbone=ResNet50()):
        super(MCCA, self).__init__()
        self.num_class = num_class
        self.register_buffer('buffer_step', torch.zeros((), dtype=torch.float32))
        self.en = backbone  # Assuming resnet_encoder is a dictionary
        self.cls_pred_conv = nn.Conv2d(256, self.num_class, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample2x_op = nn.UpsamplingBilinear2d(scale_factor=2)
        self.changec2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.changec3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.aff1 = AFF(channels=256)
        self.aff2 = AFF(channels=256)
        self.aff3 = AFF(channels=256)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.ppm = PSPModule(2048, norm_layer=nn.BatchNorm2d, out_features=256)
        self.down = DownBlock(2048, 256, nb_Conv=2)
        self.up4 = UpBlock_attention(256, 256, nb_Conv=2)
        self.up3 = UpBlock_attention(256, 256, nb_Conv=2)
        self.up2 = UpBlock_attention(256, 256, nb_Conv=2)
        self.up1 = UpBlock_attention(256, 256, nb_Conv=2)
        self.se = SELayer(channel=256)
        self.squeeze_body_edge = SqueezeBodyEdge(256)
        self.bot_fine = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.edge_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
        self.upsample8x = nn.Upsample(scale_factor=8)
        self.upsample4x = nn.Upsample(scale_factor=4)
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )
        self.sigmoid_edge = nn.Sigmoid()
        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_class, kernel_size=1, bias=False)
        )
        self.final_seg = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_class, kernel_size=1, bias=False)
        )
        self.conv0 = nn.Conv2d(2048,256,1)

    def forward(self, x, y=None):
        feat_list = self.en(x)
        E5 = self.down(feat_list[3])
        # aff_cc = self.ppm(feat_list[3])
        aff_cc = self.conv0(feat_list[3])
        seg_body, seg_edge = self.squeeze_body_edge(aff_cc)
        dec0_fine = self.bot_fine(feat_list[0])
        seg_edge = self.upsample8x(seg_edge)
        seg_edge = self.edge_fusion(torch.cat([seg_edge, dec0_fine], dim=1))
        seg_edge_out = self.edge_out(seg_edge)
        seg_edge_out = self.upsample4x(seg_edge_out)
        seg_edge_out = self.sigmoid_edge(seg_edge_out)
    
        seg_out = seg_edge + self.upsample8x(seg_body)
        aspp = self.upsample8x(aff_cc)
        seg_out = torch.cat([aspp, seg_out], dim=1)

        aff_high_layer1 = self.upsample2x_op(aff_cc)
        aff_c_layer1 = self.changec2(feat_list[2])
        aff_layer2 = self.aff1(aff_c_layer1, aff_high_layer1)
        aff_high_layer2 = self.upsample2x_op(aff_layer2)
        aff_c_layer2 = self.changec3(feat_list[1])
        aff_layer3 = self.aff2(aff_c_layer2, aff_high_layer2)
        aff_high_layer3 = self.upsample2x_op(aff_layer3)
        aff_layer4 = self.aff3(feat_list[0], aff_high_layer3)

        x = self.up4(E5, aff_cc)
        x = self.up3(x, aff_layer2)
        x = self.up2(x, aff_layer3)
        x = self.up1(x, aff_layer4)
        x = torch.cat([x, seg_out], dim=1)
        x = self.final_seg(x)
        cls_pred = self.upsample4x_op(x)

        return cls_pred.softmax(dim=1)


if __name__ == '__main__':
    print('Start Testing')
    x = torch.randn(2, 3, 512, 512)
    net = MCCA()
    out = net(x)
    print(out.shape)

