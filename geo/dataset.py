import json
import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import is_image_file


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
            target_transform=None,
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
        h, w = img.shape
        th, tw = self.output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __getitem__(self, index):
        img_path = self.images[index]
        img = plt.imread(img_path)
        labels = self.labels[img_path] if self.dataset == 'train' else ()

        idx = torch.from_numpy(np.atleast_2d(labels)).long()
        target = torch.zeros(img.shape)
        if idx.size(1) > 0:
            target[idx[:, 1], idx[:, 0]] = 1.

        i, j, th, tw = self._get_crop(img)
        img = img[i:i+th, j:j+tw]
        target = target[i:i+th, j:j+tw]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.images)
