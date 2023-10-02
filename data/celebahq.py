# Created by Baole Fang at 8/28/23

import numpy as np
import torch.utils.data as data
import os
from PIL import Image
import torchvision
import csv


class CelebahqDataset(data.Dataset):
    def __init__(self, root, attributes, transforms, preload=True):
        self.root = root
        self.filenames = sorted(os.listdir(root))
        self.transforms = transforms
        self.images = []
        self.attributes = []
        for filename in self.filenames:
            self.attributes.append(attributes[filename])
            if preload:
                self.images.append(Image.open(filename))
        self.attributes = np.array(self.attributes)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.images:
            image = self.images[idx]
        else:
            image = Image.open(self.filenames[idx])
        image = self.transforms(image)
        ys = self.attributes[idx]
        return image, ys


def create_transforms(transforms):
    mean = [0.5106, 0.4019, 0.3513]
    std = [0.3076, 0.2700, 0.2589]
    transforms_dict = torchvision.transforms.__dict__
    layers = []
    if transforms:
        for transform, args in transforms.items():
            layers.append(transforms_dict[transform](**args['args']))
    layers.append(torchvision.transforms.ToTensor())
    layers.append(torchvision.transforms.Normalize(mean, std))
    layers.append(torchvision.transforms.Resize(224))
    return torchvision.transforms.Compose(layers)


def read_attributes(path, keys_list):
    attributes = {}
    with open(path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_id = row['image_id']
            attribute = []
            for keys in keys_list:
                attribute.append([max(0, int(row[key])) for key in keys])
            attributes[image_id] = attribute
    return attributes


def create_dataloader(root, batch_size, train_transforms, valid_transforms, keys_list, preload=True):
    attributes = read_attributes(os.path.join(root, 'list_attr_celeba.txt'), keys_list)
    train_transforms = create_transforms(train_transforms)
    valid_transforms = create_transforms(valid_transforms)
    train_dataset = CelebahqDataset(os.path.join(root, 'train'), attributes, train_transforms, preload)
    valid_dataset = CelebahqDataset(os.path.join(root, 'valid'), attributes, valid_transforms, preload)
    test_dataset = CelebahqDataset(os.path.join(root, 'test'), attributes, valid_transforms, preload)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    valid_loader = data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=os.cpu_count() // 2
    )
    return train_loader, valid_loader, test_loader
