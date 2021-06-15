import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from cl_gym.backbones import ContinualBackbone
from typing import Union, List, Optional


class GetSubnet(autograd.Function):
    """
    source: https://github.com/allenai/hidden-networks/
    """
    @staticmethod
    def forward(ctx, scores, k):
        # out = (scores >= 0.0).float()
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SupermaskLinear(nn.Linear):
    def __init__(self, sparsity, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores / weights
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        
        self.sparsity = sparsity
    
    @torch.no_grad()
    def set_weight(self, weight: torch.Tensor):
        self.weight.data.copy_(weight)
        self.weight.requires_grad = False
    
    @torch.no_grad()
    def get_supermask(self):
        subnet = GetSubnet.apply(self.scores, self.sparsity)
        return subnet

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores, self.sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)


class SuperMaskMLP(nn.Module):
    def __init__(self, sparsity: Optional[float] = 0.5):
        super(SuperMaskMLP, self).__init__()
        self.sparsity: float = sparsity
        self.bias_dims: List[int] = []
        self.blocks: Union[nn.ModuleList, List[SupermaskLinear]] = []

    def _create_block(self, params: nn.Linear):
        inp_dim, out_dim = params.in_features, params.out_features
        weight = params.weight.data
        bias = params.bias.data if params.bias is not None else None
        bias_dim = 0 if bias is None else 1
        block = SupermaskLinear(self.sparsity, in_features=inp_dim+bias_dim, out_features=out_dim, bias=False)
        if bias is not None:
            weight = torch.cat((weight, bias.view(-1, 1)), dim=1)
            block.set_weight(weight)
        return block, bias_dim
        
    def from_backbone(self, backbone: ContinualBackbone):
        self.blocks, self.bias_dims = [], []
        for block_id in range(3):
            block, bias_dim = self._create_block(backbone.blocks[block_id].layers[0])
            self.blocks.append(block)
            self.bias_dims.append(bias_dim)
        
    def forward(self, x, task_id: Optional[int] = None):
        x = x.view(x.shape[0], -1)
        for i in range(3):
            if self.bias_dims[i]:
                ones = torch.ones(x.shape[0], 1).to(x.device)
                x = torch.cat((x, ones), dim=1)
            x = self.blocks[i](x)
            # last layer doesn't have activation
            if i != 2:
                x = F.relu(x)
        return x


if __name__ == "__main__":
    from cl_gym.backbones import MLP2Layers
    main_net = MLP2Layers(input_dim=2, hidden_dim_1=10, hidden_dim_2=10, output_dim=2, bias=False)
    inp = torch.randn(32, 2)
    supermask_net = SuperMaskMLP(sparsity=0.4)
    supermask_net.from_backbone(main_net)
    print(supermask_net(inp))