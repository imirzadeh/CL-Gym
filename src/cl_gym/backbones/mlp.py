import torch
import torch.nn as nn
import torch.nn.functional as F
from cl_gym.backbones import ContinualBackbone


class FCBlock(nn.Module):
    """
    Fully-Connected block with activations
    (i.e., linear layer, followed by ReLU and [optionally] dropout layer)
    """
    
    def __init__(self, inp_dim: int, out_dim: int,
                 dropout_prob: float = 0.0,
                 include_activation=True,
                 activation_func='ReLU'):
        super(FCBlock, self).__init__()
        if include_activation:
            # self.layers = [nn.Linear(inp_dim, out_dim), nn.ReLU(inplace=True)]
            acts = {
                'relu': nn.ReLU(),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
            }
            self.layers = [nn.Linear(inp_dim, out_dim), acts[activation_func.lower()]]
        else:
            self.layers = [nn.Linear(inp_dim, out_dim)]
        if dropout_prob > 0.0:
            self.layers.append(nn.Dropout(dropout_prob))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class MLP2Layers(ContinualBackbone):
    def __init__(self, multi_head=False, num_classes_per_head=None,
                 input_dim=784, hidden_dim_1=200, hidden_dim_2=200,
                 output_dim=10, dropout_prob=0.0, activation='ReLU'):
        # model variables
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        super(MLP2Layers, self).__init__(multi_head, num_classes_per_head)
        self.block_1 = FCBlock(self.input_dim, self.hidden_dim_1, self.dropout_prob, True, activation)
        self.block_2 = FCBlock(self.hidden_dim_1, self.hidden_dim_2, self.dropout_prob, True, activation)
        self.block_3 = FCBlock(self.hidden_dim_2, self.output_dim, 0.0, False, activation)
    
    @torch.no_grad()
    def record_activations(self, x):
        result = {}
        x = x.view(x.shape[0], -1)
        
        out = self.block_1(x)
        result['block_1'] = out.detach().clone()
        
        out = self.block_2(out)
        result['block_2'] = out.detach().clone()
        
        result['total'] = torch.cat((result['block_1'], result['block_2']), dim=1)
        return result
    
    def forward(self, x, task_id=None):
        x = x.view(x.shape[0], -1)
        out = self.block_1(x)
        out = self.block_2(out)
        out = self.block_3(out)
        if self.multi_head:
            offset1 = int((task_id - 1) * self.num_classes_per_head)
            offset2 = int(task_id * self.num_classes_per_head)
            out[:, :offset1].data.fill_(-10e10)
            out[:, offset2:].data.fill_(-10e10)
        return out