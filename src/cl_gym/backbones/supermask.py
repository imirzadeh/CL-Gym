import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        # out = scores.clone()
        out = (scores >= 0.0).float()
        # _, idx = scores.flatten().sort()
        # j = int((1 - k) * scores.numel())
        #
        # # flat_out and out access the same memory.
        # flat_out = out.flatten()
        # flat_out[idx[:j]] = 0
        # flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SupermaskLinear(nn.Linear):
    def __init__(self, sparsity, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        
        self.sparsity = sparsity
        
    def forward(self, x):
        subnet = GetSubnet.apply(self.scores, self.sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)


class SuperMaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, sparsity):
        super(SuperMaskMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        
        self.w1 = SupermaskLinear(sparsity, in_features=self.input_dim+1, out_features=self.hidden_dim_1, bias=True)
        self.w2 = SupermaskLinear(sparsity, in_features=self.hidden_dim_1, out_features=self.hidden_dim_2, bias=True)
        self.w3 = SupermaskLinear(sparsity, in_features=self.hidden_dim_2, out_features=self.output_dim, bias=True)
    
    def replace_weights(self, layer, weight):
        assert 1 <= layer <= 3
        layer_weights = {1: self.w1, 2: self.w2, 3: self.w3}
        layer_weights[layer].weight.data.copy_(weight)
        layer_weights[layer].weight.requires_grad = False

    def forward(self, x, task_id=None):
        x = x.view(x.shape[0], -1)
        ones = torch.ones(x.shape[0], 1).to(x.device)
        x = torch.cat((x, ones), dim=1)
        x = self.w1(x)
        x = F.relu(x)
        x = self.w2(x)
        x = F.relu(x)
        x = self.w3(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim

        self.w1 = nn.Linear(self.input_dim+1, self.hidden_dim_1, bias=True)
        self.w2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2, bias=True)
        self.w3 = nn.Linear(self.hidden_dim_2, self.output_dim, bias=False)
    
    @torch.no_grad()
    def get_layer_weights(self, layer):
        layer_weights = {1: self.w1, 2: self.w2, 3: self.w3}
        return layer_weights[layer].weight.data.detach().clone()
        
    def forward(self, x, task_id=None):
        x = x.view(x.shape[0], -1)
        # manually add bias since SuerMask doesn't support bias natively?
        ones = torch.ones(x.shape[0], 1).to(x.device)
        x = torch.cat((x, ones), dim=1)
        x = self.w1(x)
        x = F.relu(x)
        x = self.w2(x)
        x = F.relu(x)
        x = self.w3(x)
        return x


# if __name__ == "__main__":
#     net1 = SuperMaskMLP(2, 5, 5, 2, 0.5)
#     net2 = MLP(2, 5, 5, 2)
# #
#     print(net1.w1.weight)
#     print(net2.w1.weight)
#     print('----'*10)
#     net1.replace_weights(1, net2.get_layer_weights(1))
#     print(net1.w1.weight)