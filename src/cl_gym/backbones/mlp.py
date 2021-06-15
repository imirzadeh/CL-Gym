import torch
import torch.nn as nn
from typing import Optional, Dict, Literal
from cl_gym.backbones import ContinualBackbone

activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
}


class FCBlock(nn.Module):
    """
    Fully-Connected block with activations
    (i.e., linear layer, followed by ReLU and [optionally] a dropout layer)
    """
    
    def __init__(self, inp_dim: int, out_dim: int,
                 dropout_prob: float = 0.0,
                 include_activation: bool = True,
                 bias: bool = True,
                 activation: Literal['relu', 'tanh', 'sigmoid'] = 'relu'):
        
        super(FCBlock, self).__init__()
        layers = [nn.Linear(inp_dim, out_dim, bias=bias)]
        if include_activation:
            layers.append(activations[activation.lower()])
        if dropout_prob > 0.0:
            self.layers.append(nn.Dropout(dropout_prob))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class MLP2Layers(ContinualBackbone):
    def __init__(self, multi_head=False, num_classes_per_head=None,
                 input_dim=784, hidden_dim_1=256, hidden_dim_2=256,
                 output_dim=10, dropout_prob=0.0, activation='ReLU',
                 bias=True, include_final_layer_act=False):
        # model variables
        super(MLP2Layers, self).__init__(multi_head, num_classes_per_head)
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.block_1 = FCBlock(self.input_dim, self.hidden_dim_1, self.dropout_prob, True, bias, activation)
        self.block_2 = FCBlock(self.hidden_dim_1, self.hidden_dim_2, self.dropout_prob, True, bias, activation)
        self.block_3 = FCBlock(self.hidden_dim_2, self.output_dim, 0.0, include_final_layer_act, bias, activation)
        self.blocks: nn.ModuleList = nn.ModuleList([self.block_1, self.block_2, self.block_3])
    
    @torch.no_grad()
    def get_block_params(self, block_id: int) -> Dict[str, torch.Tensor]:
        assert 1 <= block_id <= 3
        block = self.blocks[block_id-1].layers[0]
        weight = block.weight.data if block.weight is not None else None
        bias = block.bias.data if block.bias is not None else None
        return {'weight': weight, 'bias': bias}
    
    @torch.no_grad()
    def get_block_outputs(self, inp: torch.Tensor, block_id: int, pre_act: bool = False):
        assert 1 <= block_id <= 3
        out = inp.view(inp.shape[0], -1)
        current_block = 0
        while current_block < block_id:
            out = self.blocks[current_block].layers[0](out)
            if not pre_act and len(self.blocks[current_block].layers) > 1:
                out = self.blocks[current_block].layers[1](out)
            current_block += 1
        return out.detach()
    
    def get_block_grads(self, block_id: int) -> Dict[str, Optional[torch.Tensor]]:
        block = self.blocks[block_id-1].layers[0]
        weight_grad, bias_grad = None, None
        if block.weight is not None and block.weight.grad is not None:
            weight_grad = block.weight.grad.data
        if block.bias is not None and block.bias.grad is not None:
            bias_grad = block.bias.grad.data
        return {'weight': weight_grad, 'bias': bias_grad}

    def forward(self, inp: torch.Tensor, head_id: Optional[int] = None) -> torch.Tensor:
        inp = inp.view(inp.shape[0], -1)
        return super(MLP2Layers, self).forward(inp, head_id)
