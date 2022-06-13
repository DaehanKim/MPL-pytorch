import logging
import math

import numpy as np
from PIL import Image
import PIL
from torchvision import datasets
from torchvision import transforms
import torch

from augmentation import RandAugment
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger(__name__)

custom_mean = (0.485, 0.456, 0.406) 
custom_std = (0.229, 0.224, 0.225)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_custom_dataset(args):
    args.resize = EfficientNet.get_image_size(args.arch)
    if args.randaug:
        n, m = args.randaug
    else:
        n, m = 2, 16  # default
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=custom_mean, std=custom_std),
    ])
    transform_finetune = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize * 0.125),
                              fill=128,
                              padding_mode='constant'),
        RandAugment(n=n, m=m),
        transforms.ToTensor(),
        transforms.Normalize(mean=custom_mean, std=custom_std),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(args.resize, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=custom_mean, std=custom_std)
    ])
    
    
    train_labeled_dataset = datasets.ImageFolder(
    '../data/train',
    transform_labeled)

    train_unlabeled_dataset = torch.utils.data.ConcatDataset(
        [datasets.ImageFolder('../data/train', TransformMPL(args, mean=custom_mean, std=custom_std)),
        datasets.ImageFolder('../data/valid', TransformMPL(args, mean=custom_mean, std=custom_std)),
        datasets.ImageFolder('../data/test', TransformMPL(args, mean=custom_mean, std=custom_std))]
    )

    finetune_dataset = datasets.ImageFolder(
        '../data/train',
        transform_finetune
    )

    valid_dataset = datasets.ImageFolder(
        '../data/valid',
        transform_val
    )
    test_dataset = datasets.ImageFolder(
        '../data/test',
        transform_val
    )
    

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset, finetune_dataset


class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant')])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize * 0.125),
                                  fill=128,
                                  padding_mode='constant'),
            RandAugment(n=n, m=m)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)


DATASET_GETTERS = {'custom' : get_custom_dataset}
