import torch
import torch.nn as nn
import torchvision

class ResNet18(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=pretrained)
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
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4



class ResNet34(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True):
        super(ResNet34, self).__init__()
        pretrained = torchvision.models.resnet34(pretrained=pretrained)

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
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4


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

        return b1, b2, b3, b4
    
class ResNet50_5(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(ResNet50_5, self).__init__()
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

        return b,b1, b2, b3, b4

class ResNet50_all(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(ResNet50_all, self).__init__()
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
        b1_list =[]
        b2_list = []
        b3_list = []
        b4_list = []
        b0 = self.relu(self.bn1(self.conv1(x)))
        b = self.maxpool(b0)
        for Bottleneck in self.layer1:
            b = Bottleneck(b)
            b1_list.append(b)
        
        for Bottleneck in self.layer2:
            b = Bottleneck(b)
            b2_list.append(b)

        for Bottleneck in self.layer3:
            b = Bottleneck(b)
            b3_list.append(b)

        for Bottleneck in self.layer4:
            b = Bottleneck(b)
            b4_list.append(b)
        return b1_list, b2_list, b3_list, b4_list



class resnext50_32x4d(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(resnext50_32x4d, self).__init__()
        pretrained = torchvision.models.resnext50_32x4d(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=True):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        if get_ha:
            return b1, b2, b3, b4, pool

        return pool


class resnet152(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(resnet152, self).__init__()
        pretrained = torchvision.models.resnet152(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=True):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        if get_ha:
            return b1, b2, b3, b4, pool

        return pool

if __name__ =='__main__':
    print('Start Test:')
    input = torch.rand((1,3,512,512))
    res50_all = ResNet50_5()
    b = res50_all(input)

    for feature in b:
        print('Layer1 Bottleneck:',feature.shape)
