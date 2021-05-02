import torch
from torch import nn
from torch import optim
import copy
from cl_gym.algorithms import ContinualAlgorithm
from cl_gym.algorithms.utils import flatten_grads, assign_grads
from cl_gym.algorithms.utils import flatten_weights, assign_weights


class ORM(ContinualAlgorithm):
    def __init__(self, backbone, benchmark, params):
        self.prev_net = None
        super(ORM, self).__init__(backbone, benchmark, params, requires_memory=True)

    def training_task_end(self):
        self.prev_net = copy.deepcopy(self.backbone)#flatten_weights(self.backbone, True)
        self.current_task += 1
    
    def _orthogonal_repr_loss(self, inp):
        model_rep = self.backbone.record_activations(inp, detach=False)
        targ_rep = self.prev_net.record_activations(inp, detach=True)
        crit = torch.nn.CosineSimilarity(dim=1)
        loss = torch.mean(torch.abs(crit(model_rep['block_1'], targ_rep['block_1'])))
        loss += torch.mean(torch.abs(crit(model_rep['block_2'], targ_rep['block_2'])))
        # print(f"loss = {loss.item()}")
        return loss
        # loss = loss/6.0
        # loss.backward()
        # grad_orthogonal = flatten_grads(self.backbone)
        # return grad_orthogonal
    
    def project_vecs(self, grad_task, grad_targ):
        a = grad_task
        b = grad_targ
        return (torch.dot(a, b) / torch.norm(b, 2))/(b/torch.norm(b, 2))
    
    def training_step(self, task_id, inp, targ, optimizer, criterion):
        optimizer.zero_grad()
        pred = self.backbone(inp, task_id)
        loss = criterion(pred, targ)
        if self.current_task > 1:
            loss += self._orthogonal_repr_loss(inp) / 2.0
            # loss_orthogonal = self._orthogonal_repr_loss(inp) / 2.0
            # loss_orthogonal.backward()
            # nn.utils.clip_grad_value_(self.backbone.parameters(), clip_value=0.5)
        loss.backward()
        nn.utils.clip_grad_value_(self.backbone.parameters(), clip_value=0.1)
        optimizer.step()
