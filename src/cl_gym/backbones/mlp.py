import torch
import torch.nn as nn
from typing import Optional, Dict, Iterable
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
                 activation: str = 'relu'):
        """
        
        Args:
            inp_dim: Input dimension.
            out_dim: Output dimension.
            dropout_prob: Dropout probability. Default: 0.0 (No dropout)
            include_activation: Should a linear layer followed by an activation layer?
            bias: Should a linear layer have bias?
            activation: Activation function. Currently supports `relu`, `tanh`, `sigmoid`.
        """
        
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
    """
    MLP model (feed-forward) with two hidden layers.
    """
    def __init__(self, multi_head=False, num_classes_per_head=None,
                 input_dim=784, hidden_dim_1=256, hidden_dim_2=256,
                 output_dim=10, dropout_prob=0.0, activation='ReLU',
                 bias=True, include_final_layer_act=False):
        """
        
        Args:
            multi_head: Is this a multi-head model?
            num_classes_per_head: If backbone is multi-head, then what is the head size?
            input_dim:
            hidden_dim_1:
            hidden_dim_2:
            output_dim:
            dropout_prob: Dropout probability. Default: No dropout.
            activation: The name of activation function. Default: 'ReLU'.
            bias: Should linear layers have bias?
            include_final_layer_act: Should the last layer have activation function? Default: False.
        """
        # model variables
        super(MLP2Layers, self).__init__(multi_head, num_classes_per_head)
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        block_1 = FCBlock(self.input_dim, self.hidden_dim_1, self.dropout_prob, True, bias, activation)
        block_2 = FCBlock(self.hidden_dim_1, self.hidden_dim_2, self.dropout_prob, True, bias, activation)
        block_3 = FCBlock(self.hidden_dim_2, self.output_dim, 0.0, include_final_layer_act, bias, activation)
        self.blocks: nn.ModuleList = nn.ModuleList([block_1, block_2, block_3])
    
    @torch.no_grad()
    def get_block_params(self, block_id: int) -> Dict[str, torch.Tensor]:
        """
        Args:
            block_id: the block number. In this case, layer.

        Returns:
            params: a dictionary of form {'weight': weight_params, 'bias': bias_params}
        """
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

    def forward(self, inp: torch.Tensor, head_ids: Optional[Iterable] = None) -> torch.Tensor:
        """
        Args:
            inp: The input of shape [BatchSize x input_dim]
            head_ids: Optional iterable (e.g., List or 1-D Tensor) object of shape [BatchSize] includes head_ids.

        Returns:
            output: The forward-pass output. Shape: [BatchSize x output_dim]
        
        Note: the `head_id` will only be used if the backbone is initiated with `multi_head = True`.
        """
        inp = inp.view(inp.shape[0], -1)
        return super(MLP2Layers, self).forward(inp, head_ids)
