import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp
from .decoder import build_decoder
from .resnet import *

class DeepLab(nn.Module):
    def __init__(self, output_stride=16, num_classes=21,
                 sync_bn=True):
        super(DeepLab, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = resnet50(backbone_path = './pretrain_models/resnet/resnet50-19c8e357.pth')
        self.aspp = build_aspp(output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, BatchNorm)

        # self.freeze_bn = freeze_bn

    def forward(self, input):
        _, low_level_feat, _, _, x = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    # def freeze_bn(self):
    #     for m in self.modules():
    #         if isinstance(m, SynchronizedBatchNorm2d):
    #             m.eval()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.eval()

    # def get_1x_lr_params(self):
    #     modules = [self.backbone]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if self.freeze_bn:
    #                 if isinstance(m[1], nn.Conv2d):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p
    #             else:
    #                 if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
    #                         or isinstance(m[1], nn.BatchNorm2d):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p

    # def get_10x_lr_params(self):
    #     modules = [self.aspp, self.decoder]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if self.freeze_bn:
    #                 if isinstance(m[1], nn.Conv2d):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p
    #             else:
    #                 if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
    #                         or isinstance(m[1], nn.BatchNorm2d):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p

if __name__ == "__main__":
    model = DeepLab(output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output.size())


