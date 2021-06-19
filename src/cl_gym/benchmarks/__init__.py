from cl_gym.benchmarks.base import Benchmark, DynamicTransformDataset, SplitDataset
from cl_gym.benchmarks.mnist import RotatedMNIST, PermutedMNIST, SplitMNIST
from cl_gym.benchmarks.toy_2D_clf import Toy2DClassification
from cl_gym.benchmarks.toy_1D_reg import Toy1DRegression
from cl_gym.benchmarks.pamap2 import PAMAP2
from cl_gym.benchmarks.mixed import MNISTFashionMNIST
from cl_gym.benchmarks.cifar import SplitCIFAR10, SplitCIFAR100


__all__ = ['Benchmark',
           'SplitMNIST',
           'RotatedMNIST',
           'PermutedMNIST',
           'Toy1DRegression',
           'Toy2DClassification',
           'PAMAP2',
           'MNISTFashionMNIST',
           'SplitCIFAR10',
           'SplitCIFAR100'
           ]
