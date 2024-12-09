import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
from utils import custom_transforms as tr
import cv2


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)  # 单通道有索引使用
        # class_map = equality.astype(int)  #单通道无索引使用
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    semantic_map = np.argmax(semantic_map, axis=-1)

    semantic_map[semantic_map == 1] = 0
    semantic_map[semantic_map == 2] = 1

    return semantic_map


def onehot_to_mask(semantic_map, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    # x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    semantic_map = np.uint8(colour_codes[semantic_map.astype(np.uint8)])
    return semantic_map


class Multi_Class_Segmentation_Dataset(data.Dataset):
    def __init__(self, img_root, gt_root, palette, trainsize, mode):
        self.trainsize = trainsize
        self.image_root = img_root
        self.mode = mode
        self.palette = palette
        self.gt_root = gt_root
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if
                       f.endswith('.png') or f.endswith('.tif')]
        self.gts = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.tif') \
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        # gt =np.array(Image.open(self.gts[index]).convert('RGB'), dtype=np.uint8)
        gt = np.array(Image.open(self.gts[index]))
        gt = mask_to_onehot(gt, self.palette)
        gt = Image.fromarray(np.uint8(gt))

        sample = {'image': image, 'label': gt}

        if self.mode == 'train':
            return self.transform_tr(sample)
        elif self.mode == 'val':
            return self.transform_val(sample)
        elif self.mode == 'test':
            file_name = self.images[index].split('/')[-1][:-len(".tif")]
            return self.transform_test(sample), file_name

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.FixScaleCrop(crop_size=self.trainsize),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.trainsize),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.trainsize),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()])

        return composed_transforms(sample)

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def __len__(self):
        return self.size


def get_loader(img_root, gt_root, palette, batchsize, trainsize, mode, num_workers=4, shuffle=True, pin_memory=True):
    dataset = Multi_Class_Segmentation_Dataset(img_root=img_root, gt_root=gt_root, palette=palette, trainsize=trainsize,
                                               mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class Multi_Class_Segmentation_Dataset_edge(data.Dataset):
    def __init__(self, img_root, gt_root, palette, trainsize, mode):
        self.trainsize = trainsize
        self.image_root = img_root
        self.mode = mode
        self.palette = palette
        self.gt_root = gt_root
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if
                       f.endswith('.png') or f.endswith('.tif')]
        self.gts = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.tif') \
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        # gt =np.array(Image.open(self.gts[index]).convert('RGB'), dtype=np.uint8)
        gt = np.array(Image.open(self.gts[index]))
        gt = mask_to_onehot(gt, self.palette)
        gt_for_edge = gt *255
        three_channel_image = np.zeros((512, 512, 3), dtype=np.uint8)
        three_channel_image[:, :, 0] = gt_for_edge
        gray = cv2.cvtColor(three_channel_image, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 0, 50)
        edge[edge == 255] = 1

        edge = Image.fromarray(np.uint8(edge))
        gt = Image.fromarray(np.uint8(gt))

        sample = {'image': image, 'label': gt, 'edge': edge}

        if self.mode == 'train':
            return self.transform_tr(sample)
        elif self.mode == 'val':
            return self.transform_val(sample)
        elif self.mode == 'test':
            file_name = self.images[index].split('/')[-1][:-len(".tif")]
            return self.transform_test(sample), file_name

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip_edge(),
            tr.RandomGaussianBlur_edge(),
            tr.FixScaleCrop_edge(crop_size=self.trainsize),
            tr.Normalize_edge(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor_edge()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.trainsize),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.trainsize),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()])

        return composed_transforms(sample)

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def __len__(self):
        return self.size


def get_loader_edge(img_root, gt_root, palette, batchsize, trainsize, mode, num_workers=4, shuffle=True,
                    pin_memory=True):
    dataset = Multi_Class_Segmentation_Dataset_edge(img_root=img_root, gt_root=gt_root, palette=palette,
                                                    trainsize=trainsize,
                                                    mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


if __name__ == '__main__':
    
    print('Test dataloader')
    train_root = ''
    train_gt = ''
    palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]
    batchsize = 1
    trainsize = 512
    train_loader = get_loader_edge(train_root, train_gt, palette, batchsize, trainsize,
                                   mode='train',
                                   num_workers=1, shuffle=True, pin_memory=True)
    for i, sample in enumerate(train_loader):
        img, label, edge = sample['image'], sample['label'], sample['edge']
        print(img.shape)
        print(label.shape)
        print(edge.shape)
        label_numpy = label.permute(1,2,0).data.cpu().numpy() * 255
        edge_numpy = edge.permute(1,2,0).data.cpu().numpy() * 255
        print(np.unique(edge_numpy))
        cv2.imshow('label', label_numpy)
        cv2.imshow('edge',edge_numpy)
        cv2.waitKey(0)
        break
