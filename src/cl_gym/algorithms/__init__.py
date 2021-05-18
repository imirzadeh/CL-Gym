from cl_gym.algorithms.base import ContinualAlgorithm
from cl_gym.algorithms.agem import AGEM
from cl_gym.algorithms.er_ring import ERRingBuffer
from cl_gym.algorithms.ewc import EWC
from cl_gym.algorithms.ogd import OGD
from cl_gym.algorithms.mcsgd import MCSGD
from cl_gym.algorithms.mtl import Multitask

__all__ = ['ContinualAlgorithm', 'AGEM', 'ERRingBuffer', 'EWC', 'OGD', 'MCSGD', 'Multitask']
