import torch
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from cl_gym.algorithms.utils import flatten_weights, assign_weights


class MCSGD(ContinualAlgorithm):
    """
    | Mode Connectivity SGD
    | By Mirzadeh et al. :https://openreview.net/forum?id=Fmg_fQYUejf
    """
    def __init__(self, backbone, benchmark, params):
        super(MCSGD, self).__init__(backbone, benchmark, params, requires_memory=True)
        self.w_bar_prev = None
        self.w_hat_curr = None
        self.num_samples_on_line = self.params.get('mcsgd_line_samples', 10)
        self.alpha = self.params.get('mcsgd_alpha', 0.25)
    
    def calculate_line_loss(self, w_start, w_end, loader):
        line_samples = np.arange(0.0, 1.01, 1.0 / float(self.num_samples_on_line))
        accum_grad = None
        for t in line_samples:
            grads = []
            w_mid = w_start + (w_end - w_start) * t
            m = assign_weights(self.backbone, w_mid)
            clf_loss = self.calculate_point_loss(m, loader)
            clf_loss.backward()
            for name, param in m.named_parameters():
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            if accum_grad is None:
                accum_grad = grads
            else:
                accum_grad += grads
        return accum_grad
   
    def calculate_point_loss(self, net, loader):
        criterion = self.prepare_criterion(-1)
        device = self.params['device']
        net.eval()
        total_loss, total_count = 0.0, 0.0
        for (inp, targ, task_ids) in loader:
            inp, targ, task_ids = inp.to(device), targ.to(device), task_ids.to(device)
            pred = net(inp, task_ids)
            total_count += len(targ)
            total_loss += criterion(pred, targ)
        total_loss /= total_count
        return total_loss
    
    def _prepare_mode_connectivity_optimizer(self, model):
        return torch.optim.SGD(model.parameters(),
                               lr=self.params['mcsgd_line_optim_lr'],
                               momentum=self.params['momentum'])

    def find_connected_minima(self, task):
        mc_model = assign_weights(self.backbone, self.w_bar_prev + (self.w_hat_curr - self.w_bar_prev) * self.alpha)
        optimizer = self._prepare_mode_connectivity_optimizer(mc_model)
        loader_prev, _ = self.benchmark.load_memory_joint(task-1, batch_size=self.params['batch_size_memory'],
                                                          num_workers=self.params.get('num_dataloader_workers', 0))
        loader_curr, _ = self.benchmark.load_subset(task, batch_size=self.params['batch_size_train'],
                                                    num_workers=self.params.get('num_dataloader_workers', 0))
        mc_model.train()
        optimizer.zero_grad()
        grads_prev = self.calculate_line_loss(self.w_bar_prev, flatten_weights(mc_model, True), loader_prev)
        grads_curr = self.calculate_line_loss(self.w_hat_curr, flatten_weights(mc_model, True), loader_curr)
        # mc_model = assign_grads(mc_model, (grads_prev + grads_curr)/2.0)
        mc_model = assign_grads(mc_model, (grads_prev + grads_curr))
        optimizer.step()
        return mc_model
    
    def training_epoch_end(self):
        self.w_hat_curr = flatten_weights(self.backbone, True)

    def training_task_end(self):
        if self.current_task > 1:
            self.backbone = self.find_connected_minima(self.current_task)
        self.w_bar_prev = flatten_weights(self.backbone, True)
        self.current_task += 1
        
    def training_step(self, task_id, inp, targ, optimizer, criterion):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_id)
        loss = criterion(pred, targ)
        loss.backward()
        # if task_id > 1:
        #     self.find_connected_minima(task_id)
        optimizer.step()
