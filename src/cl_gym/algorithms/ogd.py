import torch
import time
from torch import nn
from torch import optim
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.utils import flatten_grads, assign_grads


class OGD(ContinualAlgorithm):
    """
    | Orthogonal Gradient Descent
    | By Farajtabar et al. : https://arxiv.org/abs/1910.07104.pdf
    """
    # implementation is partially based on: https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus/
    def __init__(self, backbone, benchmark, params):
        super(OGD, self).__init__(backbone, benchmark, params)
        self.gradient_storage = []
        self.orthonormal_basis = None
    
    @torch.no_grad()
    def __update_orthonormal_basis(self):
        q, r = torch.qr(torch.stack(self.gradient_storage).T)
        self.orthonormal_basis = q.T
    
    @torch.no_grad()
    def __project_grad_vector(self, g):
        # print("Projection || Shape check >> g={}, basis={}".format(g.shape, self.orthonormal_basis.shape))
        mid = (torch.mm(self.orthonormal_basis, g.view(-1, 1))).T
        res = torch.mm(mid, self.orthonormal_basis)
        # print("New grad shape >> ", res.shape)
        return res.view(-1)
    
    def __update_gradient_storage(self, task):
        mem_loader_train, _ = self.benchmark.load_memory(task, batch_size=1)
        criteriton = self.prepare_criterion(self.current_task)
        optimizer = self.prepare_optimizer(self.current_task)
        self.backbone.train()
        for inp, targ, task_id in mem_loader_train:
            optimizer.zero_grad()
            pred = self.backbone(inp)
            loss = criteriton(pred, targ)
            loss.backward()
            grad_batch = flatten_grads(self.backbone).detach().clone()
            self.gradient_storage.append(grad_batch)
        optimizer.zero_grad()

    def training_task_end(self):
        self.__update_gradient_storage(self.current_task)
        self.__update_orthonormal_basis()
        
    def training_step(self, task_id, inp, targ, optimizer, criterion):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_id)
        loss = criterion(pred, targ)
        loss.backward()
        pred = self.backbone(inp)
        loss = criterion(pred, targ)
        loss.backward()
        if task_id > 1:
            grad_batch = flatten_grads(self.backbone).detach().clone()
            optimizer.zero_grad()
            proj_grad = self.__project_grad_vector(grad_batch)
            new_grad = grad_batch - proj_grad
            self.backbone = assign_grads(self.backbone, new_grad)
        optimizer.step()
