import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.utils import flatten_grads, assign_grads


class AGEM(ContinualAlgorithm):
    """
    | Averaged Gradient Episodic Memory
    | By Chaudhry et al.: https://arxiv.org/abs/1812.00420.pdf
    """
    # Implementation is partially based on: https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus
    def __init__(self, backbone, benchmark, params):
        super(AGEM, self).__init__(backbone, benchmark, params, requires_memory=True)
    
    @staticmethod
    def __is_violating_direction_constraint(grad_ref, grad_batch):
        """
        GEM and A-GEM operate on gradient directions.
        i.e., gradient direction should have angle less than 90 degrees with reference gradient.
        :param grad_ref: reference gradient (i.e., grads on episodic memory)
        :param grad_batch: batch gradient
        :return:
        """
        return torch.dot(grad_ref, grad_batch) < 0
    
    @staticmethod
    def __project_grad_vector(grad_ref, grad_batch):
        """
        A-GEM operates on regularized average gradient directions.
        In case of violation, gradients should be projected (see Eq.(11) in A-GEM paper).
        :param grad_ref: reference gradient (i.e., grads on episodic memory examples)
        :param grad_batch: current batch gradients
        :return: projected gradients
        """
        return grad_batch - (torch.dot(grad_batch, grad_ref) / torch.dot(grad_ref, grad_ref)) * grad_ref
    
    def training_step(self, task_ids, inp, targ, optimizer, criterion):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_ids)
        loss = criterion(pred, targ)
        loss.backward()
        if task_ids[0] > 1:
            grad_batch = flatten_grads(self.backbone).detach().clone()
            inp_ref, targ_ref, task_ids_ref = self.sample_batch_from_memory()
            loss = criterion(self.backbone(inp_ref, task_ids_ref), targ_ref.reshape(len(targ_ref)))
            loss.backward()
            grad_ref = flatten_grads(self.backbone).detach().clone()
            if self.__is_violating_direction_constraint(grad_ref, grad_batch):
                grad_ref = self.__project_grad_vector(grad_ref, grad_batch)
            optimizer.zero_grad()
            self.backbone = assign_grads(self.backbone, grad_ref)
        optimizer.step()
