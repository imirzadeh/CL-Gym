import torch
from torch import nn
from torch import optim
from cl_gym.algorithms import ContinualAlgorithm


class Multitask(ContinualAlgorithm):
    """
    Multitask (Joint) Training
    """
    def __init__(self, backbone, benchmark, params):
        super(Multitask, self).__init__(backbone, benchmark, params, requires_memory=True)

    def prepare_train_loader(self, task_id):
        return self.benchmark.load_joint(task_id, self.params['batch_size_train'], shuffle=True, pin_memory=True)[0]
