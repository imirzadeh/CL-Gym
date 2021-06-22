import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from cl_gym.algorithms import ContinualAlgorithm


class EWC(ContinualAlgorithm):
    """
    | Elastic Weight Consolidation
    | By Kirkpatricka et al. : https://arxiv.org/abs/1612.00796.pdf
    """
    # implementation is partially based on: https://github.com/kuc2477/pytorch-ewc
    def __init__(self, backbone, benchmark, params, fisher_lambda: float = 50.0, fisher_sample_size: int = 128):
        """
        Args:
            backbone: the backbone model
            benchmark: the benchmark
            params: params for training
            fisher_lambda: The lambda coefficient of EWC algorithm
            fisher_sample_size: Sample size for calculating Fisher diagonal
        """
        super(EWC, self).__init__(backbone, benchmark, params)
        self.fisher_lambda = fisher_lambda
        self.fisher_sample_size = fisher_sample_size
    
    def __estimate_fisher_diagonal(self):
        log_likelihoods = []
        samples_so_far = 0
        train_loader, _ = self.benchmark.load(self.current_task, batch_size=32)
        for x, y, _ in train_loader:
            batch_size = len(y)
            x = x.to(self.params['device'])
            y = y.to(self.params['device'])
            log_out = F.log_softmax(self.backbone(x, self.current_task), dim=1)
            log_likelihoods.append(log_out[range(batch_size), y.data])
            samples_so_far += batch_size
            if samples_so_far > self.fisher_sample_size:
                break
        
        log_likelihoods = torch.cat(log_likelihoods).unbind()
        grads = zip(*[autograd.grad(l, self.backbone.parameters(), retain_graph=(i < len(log_likelihoods)))\
                      for i, l in enumerate(log_likelihoods, 1)])
        grads = [torch.stack(grad) for grad in grads]
        fisher_diagonals = [(grad ** 2).mean(0) for grad in grads]
        
        # note for the next line: in pytorch, module names are like W1.weight
        # but, we can't get attrs using getattr('W1.weight') because of the nested call (dot)
        # one trick is to replace the '.' with '_'
        # the other tick is to use: functools.reduce(getattr, [obj] + attr.split('.'))
        param_names = [n.replace('.', '_') for n, p in self.backbone.named_parameters()]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}
    
    def __consolidate(self):
        fisher_diagonals = self.__estimate_fisher_diagonal()
        for name, param in self.backbone.named_parameters():
            name = name.replace('.', '_')
            self.backbone.register_buffer(f"{name}_mean", param.data.clone())
            self.backbone.register_buffer(f"{name}_fisher", fisher_diagonals[name].data.clone())
    
    def __calculate_ewc_loss(self):
        # shouldn't be called for the first task
        # because we have not consolidated anything yet
        losses = []
        for name, param in self.backbone.named_parameters():
            name = name.replace('.', '_')
            mean = getattr(self.backbone, f"{name}_mean")
            fisher = getattr(self.backbone, f"{name}_fisher")
            losses.append((fisher * (param - mean)**2).sum())
        
        return (self.fisher_lambda/2.0)*sum(losses)
    
    def training_task_end(self):
        self.__consolidate()
        self.current_task += 1
        
    def training_step(self, task_id, inp, targ, optimizer, criterion):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_id)
        loss = criterion(pred, targ)
        if task_id > 1:
            loss += self.__calculate_ewc_loss()
        loss.backward()
        optimizer.step()