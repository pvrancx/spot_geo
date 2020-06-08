from torchvision.models.segmentation import fcn_resnet50
import torch.nn as nn


def create_fcn_resnet(**kwargs) -> nn.Module:
    """Create fully convolutional resnet for 1 channel image segmentation"""
    m = fcn_resnet50(pretrained=False, num_classes=1, **kwargs)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return m
