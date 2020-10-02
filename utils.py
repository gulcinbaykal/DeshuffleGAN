### REFERENCE: https://github.com/bbrattoli/JigsawPuzzlePytorch/blob/ec85994b9f244d08652a3975c1c7a55483cdfc05/Dataset/JigsawImageLoader.py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
from torchvision import datasets
import torch.nn.functional as F

class Shuffler():
    def __init__(self, batch, classes, resize):
        self.images = batch
        self.resize = resize
        self.permutations = self.retrieve_permutations(classes)
        self.__augment_tile = transforms.Compose([ 
            transforms.Resize((int(resize/3), int(resize/3)), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor()
        ])
    def retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        return all_perm
    def shuffle(self):
        b, _, _, _ = self.images.shape
        shuffled_batch = []
        orders = []
        for index in range(b):
            img = self.images[index].permute(1, 2, 0)
            w = int(img.shape[0] / 3)
            tiles = [None] * 9
            for n in range(9):
                y = int(n / 3)
                x = n % 3
                tile = img[int(y * w) : int((y + 1) * w), int(x * w) : int((x + 1) * w), :]
                tiles[n] = tile
            order = np.random.randint(len(self.permutations))
            data = [tiles[self.permutations[order][t]] for t in range(9)]
            data = torch.stack(data, 0)
            data = data.permute(0, 3, 1, 2)
            data = torchvision.utils.make_grid(data, 3, padding=0)
            shuffled_batch.append(data)
            orders.append(int(order))

        shuffled_batch = torch.stack(shuffled_batch, 0)
        if self.resize % 3 == 1:
            shuffled_batch = nn.ReplicationPad2d((1, 0, 1, 0))(shuffled_batch)
        elif self.resize % 3 == 2:
            shuffled_batch = nn.ReplicationPad2d((1, 1, 1, 1))(shuffled_batch)
        return shuffled_batch, orders

def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')

