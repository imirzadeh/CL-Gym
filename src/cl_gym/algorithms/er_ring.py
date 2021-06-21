import torch
import time
from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm


class ERRingBuffer(ContinualAlgorithm):
    """
    | Experience Replay Ring Buffer
    | By Chaudhry et al. : https://arxiv.org/abs/1902.10486.pdf
    """
    def __init__(self, backbone, benchmark, params):
        super(ERRingBuffer, self).__init__(backbone, benchmark, params, requires_memory=True)
        self.episodic_memory_iter = None
        self.episodic_memory_loader = None

    def training_step(self, task_id, inp, targ, optimizer, criterion):
        optimizer.zero_grad()
        if task_id[0] > 1:
            mem_inp, mem_targ, mem_task_ids = self.sample_batch_from_memory()
            cat_inp = torch.cat([inp, mem_inp], dim=0)
            cat_task_ids = torch.cat([task_id, mem_task_ids], dim=0)
            assert len(cat_inp) == len(cat_task_ids)
            # print(targ.shape, mem_targ.shape)
            cat_targ = torch.cat([targ, mem_targ.reshape(len(mem_targ))], dim=0)
            pred = self.backbone(cat_inp, cat_task_ids)
            loss = criterion(pred, cat_targ)
        else:
            pred = self.backbone(inp, task_id)
            loss = criterion(pred, targ)
        loss.backward()
        optimizer.step()
