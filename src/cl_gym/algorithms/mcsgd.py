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
        self.num_samples_on_line = self.params.get('mcsgd_line_samples', 5)
        self.alpha = self.params.get('mcsgd_alpha', 0.5)
    
    def calculate_line_loss(self, w_start, w_end, loader):
        line_samples = np.arange(0.0, 1.01, 1.0 / float(self.num_samples_on_line))
        accum_grad = None
        import time
        t_calc_point = 0
        t_calc_grads = 0
        for t in line_samples:
            grads = []
            w_mid = w_start + (w_end - w_start) * t
            t0 = time.time()
            m = assign_weights(self.backbone, w_mid)
            t1 = time.time()
            clf_loss = self.calculate_point_loss(m, loader)
            t2 = time.time()
            t_calc_point += (t2 - t1)
            clf_loss.backward()
            for name, param in m.named_parameters():
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            if accum_grad is None:
                accum_grad = grads
            else:
                accum_grad += grads
            t3 = time.time()
            t_calc_grads += (t3 - t2)
        return accum_grad/self.num_samples_on_line
   
    def calculate_point_loss(self, net, loader):
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        net.eval()
        total_loss, total_count = 0.0, 0.0
        for (inp, targ, _) in loader:
            pred = net(inp)
            total_count += len(targ)
            total_loss += criterion(pred, targ)
        total_loss /= total_count
        return total_loss
    
    def find_connected_minima(self, task):
        # print(f"Debug >> w_bar_prev? {self.w_bar_prev is not None}, w_hat_curr? {self.w_hat_curr is not None}")
        mc_model = assign_weights(self.backbone, self.w_bar_prev + (self.w_hat_curr - self.w_bar_prev) * self.alpha)
        optimizer = self.prepare_optimizer(task)
        loader_prev, _ = self.benchmark.load_memory_joint(task-1, batch_size=32)
        loader_curr, _ = self.benchmark.load_subset(task, batch_size=32)
        mc_model.train()
        optimizer.zero_grad()
        grads_prev = self.calculate_line_loss(self.w_bar_prev, flatten_weights(mc_model, True), loader_prev)
        grads_curr = self.calculate_line_loss(self.w_hat_curr, flatten_weights(mc_model, True), loader_curr)
        optimizer.zero_grad()
        mc_model = assign_grads(mc_model, (grads_prev + grads_curr)/2.0)
        optimizer.step()
        return mc_model
    
    def training_epoch_end(self):
        self.w_hat_curr = flatten_weights(self.backbone, True)
        # if self.current_task > 1:
        #     self.backbone = self.find_connected_minima(self.current_task)

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
