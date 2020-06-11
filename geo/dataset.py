import json
import os
import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import is_image_file
from torchvision.transforms.functional import crop


def read_annotation_file(path, img_root):
    with open(path) as annotation_file:
        annotation_list = json.load(annotation_file)
        # Transform list of annotations into dictionary
    annotation_dict = {}
    for annotation in annotation_list:
        sequence_id = annotation['sequence_id']
        frame_id = annotation['frame']
        img_path = os.path.join(img_root, '{}'.format(sequence_id), '{}.png'.format(frame_id))
        annotation_dict[img_path] = tuple(annotation['object_coords'])
    return annotation_dict


class GeoSetFromFolder(VisionDataset):
    """
    Loads images in directory and all subdirs as flat dataset.
    """

    def __init__(
            self,
            root: str,
            dataset: str,
            output_size: Tuple[int, int],
            transform=None,
            target_transform=None
    ):

        super(GeoSetFromFolder, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )

        assert dataset.lower() in {'train', 'test'}, 'unknown dataset'
        assert os.path.isdir(root), 'Root folder not found.'

        img_root = os.path.join(root, dataset)
        labelfile = os.path.join(root, 'train_anno.json')

        assert os.path.isdir(img_root), 'Image folder not found.'
        assert os.path.isfile(labelfile), 'Annotations file not found'

        self.dataset = dataset.lower()
        self.images = []
        self.output_size = output_size
        for root, _, files in os.walk(img_root):
            for x in files:
                if is_image_file(x):
                    self.images.append(os.path.join(root, x))
        self.labels = read_annotation_file(labelfile, img_root) if dataset == 'train' else {}

    def _get_crop(self, img):
        w, h = img.size
        th, tw = self.output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('L')
        labels = self.labels[img_path] if self.dataset == 'train' else ()
        idx = torch.from_numpy(np.atleast_2d(labels)).long()
        target = torch.zeros((img.height, img.width))
        if idx.size(1) > 0:
            target[idx[:, 1], idx[:, 0]] = 1.

        i, j, th, tw = self._get_crop(img)
        img = crop(img, i, j, th, tw)
        target = target[i:i+th, j:j+tw]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.images)


class GeoSetFromFolderCrop(VisionDataset):
    """
    Load images from directories ignoring series. Crop around target.
    """

    def __init__(
            self,
            root: str,
            dataset: str,
            output_size: Tuple[int, int],
            transform=None,
            target_transform=None
    ):

        super(GeoSetFromFolderCrop, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform
        )

        assert dataset.lower() in {'train', 'test'}, 'unknown dataset'
        assert os.path.isdir(root), 'Root folder not found.'

        img_root = os.path.join(root, dataset)
        labelfile = os.path.join(root, 'train_anno.json')

        assert os.path.isdir(img_root), 'Image folder not found.'
        assert os.path.isfile(labelfile), 'Annotations file not found'

        self.dataset = dataset.lower()
        self.images = []
        self.output_size = output_size
        for root, _, files in os.walk(img_root):
            for x in files:
                if is_image_file(x):
                    self.images.append(os.path.join(root, x))
        self.labels = read_annotation_file(labelfile, img_root) if dataset == 'train' else {}

    def _get_crop(self, img, labels):
        w, h = img.size
        th, tw = self.output_size
        if w == tw and h == th:
            return 0, 0, h, w
        if len(labels)>0:
            i = int(labels[0][1] + 0.5) - int(th/2)
            j = int(labels[0][0] + 0.5) - int(tw/2)
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        return np.clip(i, 0, h-th) , np.clip(j, 0, w-tw), th, tw # clip values so crop doesn't results in small images when target is close to border.

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('L')
        labels = self.labels[img_path] if self.dataset == 'train' else ()
        idx = torch.from_numpy(np.atleast_2d(labels)).long()
        target = torch.zeros((img.height, img.width))
        if idx.size(1) > 0:
            target[idx[:, 1], idx[:, 0]] = 1.

        i, j, th, tw = self._get_crop(img, labels)
        img = crop(img, i, j, th, tw)
        target = target[i:i+th, j:j+tw]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":


    from torch.utils.data.dataloader import DataLoader
    from torchvision.transforms import transforms
    import torchvision.utils as vutils

    dataset = GeoSetFromFolderCrop(
            root='data/',
            dataset='train',
            output_size=(16,16),
            transform = transforms.ToTensor()
        )
    
    
    train_load = DataLoader(dataset, batch_size=32,shuffle = True)
    for i in range(10):
        img, target = next(iter(train_load))
        print(img.shape)
    vutils.save_image(torch.cat((img,target.unsqueeze_(1)), 0), 'batch.png', normalize=True)
    