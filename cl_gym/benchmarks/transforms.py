import PIL
import torch
import torchvision
import numpy as np
from torchvision.transforms.functional import rotate

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

FASHION_MNIST_MEAN, FASHION_MNIST_STD = (0.2860,), (0.3530,)

CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2762)


class GaussianNoiseTransform:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def corrupt_labels(dataset, num_classes: int, corruption_prob: float):
    if isinstance(dataset.targets, list):
        target_classes = np.asarray(dataset.targets)
    else:
        target_classes = dataset.targets.clone().detach().numpy()
    mask = np.random.rand(len(target_classes)) <= corruption_prob
    random_labels = np.random.choice(num_classes, mask.sum())
    target_classes[mask] = random_labels
    target_classes = [int(x) for x in target_classes]
    dataset.targets = target_classes

 
class RotationTransform:
    """
    Rotation transforms for the images in `Rotation MNIST` dataset.
    """
    def __init__(self, angle):
        self.angle = angle
    
    def __call__(self, x):
        return rotate(x, self.angle, fill=(0,), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)


class PermuteTransform:
    """
    Permutation transform, useful for permuted MNIST.
    """
    def __init__(self, permute_indices):
        self.permuted_indices = permute_indices
    
    def __call__(self, x):
        shape = x.shape
        return x.view(-1)[self.permuted_indices].view(shape)


def get_default_rotation_mnist_transform(num_tasks: int, per_task_rotation: float = None):
    if not per_task_rotation:
        per_task_rotation = 180.0/num_tasks
    transforms = []
    for task in range(1, num_tasks+1):
        rotation_degree = (task - 1) * per_task_rotation
        transform = torchvision.transforms.Compose([
            RotationTransform(rotation_degree),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])
        transforms.append(transform)
    assert len(transforms) == num_tasks
    return transforms


def get_default_permuted_mnist_transform(num_tasks: int):
    transforms = []
    for task in range(1, num_tasks+1):
        transform = [torchvision.transforms.ToTensor()]
        if task != 1:
            transform.append(PermuteTransform(torch.randperm(28 * 28)))
        transform.append(torchvision.transforms.Normalize(MNIST_MEAN, MNIST_STD))
        transforms.append(torchvision.transforms.Compose(transform))
    assert len(transforms) == num_tasks
    return transforms


def get_default_mnist_transform(num_tasks: int):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    return [transforms]*num_tasks


def get_default_fashion_mnist_transform(num_tasks: int):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
    ])
    return [transforms]*num_tasks


def get_default_cifar_transform(num_tasks: int, is_cifar_100=False):
    normalize_mean = CIFAR100_MEAN if is_cifar_100 else CIFAR10_MEAN
    normalize_std = CIFAR100_STD if is_cifar_100 else CIFAR10_STD
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(normalize_mean, normalize_std),
    ])
    return [transforms]*num_tasks


def get_default_mnist_fashion_mnist_transform():
    return get_default_mnist_transform(1) + get_default_fashion_mnist_transform(1)
