import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import models


class DSNetLocalGuide(nn.Module):
    """ DSNet + PAM + Local prob map guidance"""

    def __init__(self,
                 backbone_name='vgg16_bn',
                 pretrained=True,
                 encoder_freeze=False,
                 classes=1,
                 decoder_filters=(256, 128, 64, 32, 16),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_batchnorm=True,
                 upsampling_rates=[2, 2, 2, 2, 2],
                 nonlocal_in_channels=512):
        super(DSNetLocalGuide, self).__init__()

        self.backbone_name = backbone_name

        self.backbone_wrap = Backbone_wrap(backbone_name, pretrained=pretrained)
        self.backbone = self.backbone_wrap.model
        self.shortcut_features = self.backbone_wrap.skip_connections_name
        self.bb_out_name = self.backbone_wrap.output_name
        # ^Previous: self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(backbone_name, pretrained=pretrained)

        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # link part
        self.link = SpatialDepsLocalGuide(nonlocal_in_channels)

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[
                          :len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            print('upsample_blocks[{}] in: {}   out: {}'.format(i, filters_in, filters_out))
            upsampling_rate = upsampling_rates[i]
            if upsampling_rate == 1:
                identical = True
            else:
                identical = False
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks - i - 1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm,
                                                      identical=identical))

        # build decoder2 part
        self.decoder2_upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[
                          :len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            print('decoder2 upsample_blocks[{}] in: {}   out: {}'.format(i, filters_in, filters_out))
            upsampling_rate = upsampling_rates[i]
            if upsampling_rate == 1:
                identical = True
            else:
                identical = False
            self.decoder2_upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                               skip_in=shortcut_chs[num_blocks - i - 1],
                                                               parametric=parametric_upsampling,
                                                               use_bn=decoder_use_batchnorm,
                                                               identical=identical))

        # final conv
        self.final_conv = nn.Conv2d(decoder_filters[-1] * 2, classes, kernel_size=(1, 1))
        self.final_conv_dec1 = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))
        self.final_conv_dec2 = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))

        # build decoder fusion for the regresion of density maps
        self.side_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(3, 3), padding=1)
        self.side_conv_2x = nn.Conv2d(decoder_filters[-2], classes, kernel_size=(3, 3), padding=1)
        self.side_conv_4x = nn.Conv2d(decoder_filters[-3], classes, kernel_size=(3, 3), padding=1)
        self.side_conv_8x = nn.Conv2d(decoder_filters[-4], classes, kernel_size=(3, 3), padding=1)
        self.side_conv_16x = nn.Conv2d(decoder_filters[-5], classes, kernel_size=(3, 3), padding=1)
        self.side_conv_list = [self.side_conv, self.side_conv_2x, self.side_conv_4x, self.side_conv_8x,
                               self.side_conv_16x][::-1]  # ^ they are all the same

        if encoder_freeze:
            self.freeze_encoder()

        self.replaced_conv1 = False  # for accommodating inputs with different number of channels later

    def freeze_encoder(self):

        """ Freezing encoder parameters, the newly initialized decoder parameters are remaining trainable. """

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):
        encoder_out, features = self.forward_backbone(*input)
        encoder_out2, features2 = self.forward_backbone(*input)

        # decoder 1 (local)
        dec1_outputs = []
        x_dec1 = encoder_out
        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x_dec1 = upsample_block(x_dec1, skip_features)
            dec1_outputs.append(x_dec1)
        self.x_out_dec1 = self.final_conv_dec1(x_dec1)

        # decoder 2 (global)
        guiding_map = self.x_out_dec1.detach()
        guiding_map = F.interpolate(guiding_map, size=encoder_out2.shape[-2:]) # or scale_factor=0.5 ** 5
        x_dec2 = self.link(encoder_out2, guiding_map)
        dec2_outputs = []
        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.decoder2_upsample_blocks):
            skip_features = features2[skip_name]
            x_dec2 = upsample_block(x_dec2, skip_features)
            dec2_outputs.append(x_dec2)
        self.x_out_dec2 = self.final_conv_dec2(x_dec2)

        # fusion
        x = torch.cat([x_dec1, x_dec2], dim=1)
        self.x_out = self.final_conv(x)

        # side convolutions
        self.side_outputs = []
        for i in range(len(dec1_outputs)):  # ^ len = 5
            # x = torch.cat([dec1_outputs[i], dec2_outputs[i]], dim=1)
            x = dec2_outputs[i] - dec1_outputs[i]
            x = self.side_conv_list[i](x)
            self.side_outputs.append(x)

        return self.x_out, self.x_out_dec1, self.x_out_dec2, self.side_outputs

    def forward_backbone(self, x):

        """ Forward propagation in backbone encoder network.  """

        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self):

        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = torch.zeros(1, 3, 224, 224)
        if self.backbone_name.startswith('vgg'):
            has_fullres_features = True
        else:
            has_fullres_features = False
        # has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution

        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))

        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param


class SpatialDepsLocalGuide(nn.Module):
    def __init__(self, in_dim):
        super(SpatialDepsLocalGuide, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        guiding_map0 = guiding_map0[:,1,:,:].unsqueeze(1)
        m_batchsize, C, height, width = x.size()

        torch.sigmoid(guiding_map0, out=guiding_map0)
        guiding_map = guiding_map0
        self.guiding_map = guiding_map

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


class Backbone_wrap:
    def __init__(self, name, pretrained=False):

        model, feature_names, backbone_output = self.get_backbone(name, pretrained=pretrained)

        self.model = model
        self.skip_connections_name = feature_names
        self.output_name = backbone_output

    def get_backbone(self, name, pretrained=False):

        """ Loading backbone, defining names for skip-connections and encoder output. """

        # TODO: More backbones

        # loading backbone model
        
        if name == 'vgg16_bn':  # custom
            backbone = models.vgg16_bn(pretrained=pretrained).features
        else:
            raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

        # specifying skip feature and output names
        if name == 'vgg16_bn':
            # TODO: consider using a 'bridge' for VGG models, there is just a MaxPool between last skip and backbone output
            feature_names = ['5', '12', '22', '32', '42']
            backbone_output = '43'
        else:
            raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

        return backbone, feature_names, backbone_output


class UpsampleBlock(nn.Module):
    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False, identical=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        self.identical = identical
        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):

        if not self.identical:
            x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=True) # ^Previous: align_corners=None
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x

if __name__ == '__main__':
    device = torch.device('cuda:6')
    model = DSNetLocalGuide().to(device)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    input = torch.autograd.Variable(torch.randn(2, 3, 512, 512)).to(device)
    x = model(input)
    for o in x:
        if isinstance(o, torch.Tensor):
            print(o.size())
        elif isinstance(o, list):
            for s in o:
                print(s.size())
        else:
            raise TypeError