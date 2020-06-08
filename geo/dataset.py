import json
import os

import numpy as np
from PIL import Image
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
        for root, _, files in os.walk(img_root):
            for x in files:
                if is_image_file(x):
                    self.images.append(os.path.join(root, x))
        self.labels = read_annotation_file(labelfile, img_root) if dataset == 'train' else {}

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('L')
        target = self.labels[img_path] if self.dataset == 'train' else ()
        target = np.array(target)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.images)
