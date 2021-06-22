import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from cl_gym.backbones import ContinualBackbone
from typing import Optional, Iterable

BN_MOMENTUM = 0.05
BN_AFFINE = True


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, config={}):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=BN_AFFINE, track_running_stats=False, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=BN_AFFINE, track_running_stats=False, momentum=BN_MOMENTUM)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, affine=False, track_running_stats=False, momentum=BN_MOMENTUM)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(ContinualBackbone):
    def __init__(self, multi_head: bool, num_classes_per_head: int,
                 block, num_blocks, num_classes, nf, config: dict = {}):
        
        super(ResNet, self).__init__(multi_head, num_classes_per_head)
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, affine=BN_AFFINE, track_running_stats=False, momentum=BN_MOMENTUM)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, config=config)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, config=config)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, config=config)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, config=config)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, config):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, config=config))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, inp: torch.Tensor, head_ids: Optional[Iterable] = None):
        bsz = inp.size(0)
        out = relu(self.bn1(self.conv1(inp.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        if self.multi_head:
            out = self.select_output_head(out, head_ids)
        return out


class ResNet18Small(ResNet):
    """
    ResNet-18 with 1/3 less feature maps, a common backbone in CL literature.
    See Appendix C in https://openreview.net/pdf?id=Fmg_fQYUejf for further details.
    """
    def __init__(self, multi_head=True, num_classes_per_head=5, num_classes=100):
        """
        Args:
            multi_head: Is this a multi-head backbone? Default: True
            num_classes_per_head: Number of classes for each head. Default: 5 (for SplitCIFAR100)
            num_classes: total number of classes for benchmark. Default: 100 (for SplitCIFAR100)
            
        . Note::
            Since this benchmark is mostly used for Split-CIFAR100, the default arguments
            are chosen to be suitable for this benchmark.
        """
        super().__init__(multi_head, num_classes_per_head, BasicBlock, [2, 2, 2, 2], num_classes, 20)
