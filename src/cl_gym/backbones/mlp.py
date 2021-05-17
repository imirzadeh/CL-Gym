import numpy as np
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
            self.layers = [nn.Linear(inp_dim, out_dim, bias=True), acts[activation_func.lower()]]
        else:
            self.layers = [nn.Linear(inp_dim, out_dim, bias=True)]
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
                 input_dim=784, hidden_dim_1=256, hidden_dim_2=256,
                 output_dim=10, dropout_prob=0.0, activation='ReLU',
                 include_final_layer_act=False):
        # model variables
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        super(MLP2Layers, self).__init__(multi_head, num_classes_per_head)
        self.block_1 = FCBlock(self.input_dim, self.hidden_dim_1, self.dropout_prob, True, activation)
        self.block_2 = FCBlock(self.hidden_dim_1, self.hidden_dim_2, self.dropout_prob, True, activation)
        self.block_3 = FCBlock(self.hidden_dim_2, self.output_dim, 0.0, include_final_layer_act, activation)
    
    def record_activations(self, x, detach=True):
        result = {}
        x = x.view(x.shape[0], -1)
        
        out = self.block_1(x)
        result['block_1'] = out if not detach else out.detach().clone()
        
        out = self.block_2(out)
        result['block_2'] = out if not detach else out.detach().clone()

        result['total'] = torch.cat((result['block_1'], result['block_2']), dim=1)
        return result
    
    @torch.no_grad()
    def get_block_params(self, block_id):
        assert 0 < block_id <= 2
        if block_id == 1:
            block = self.block_1
        else:
            block = self.block_2
        
        params = {'weight': block.layers[0].weight.data.cpu().numpy(),
                  'bias': block.layers[0].bias.data.cpu().numpy().T.reshape(-1, 1)}
        return params
        
    @torch.no_grad()
    def record_distance_to_boundary(self, x, reduction=None):
        result = {'block_1': [], 'block_2': [],
                  'block_1_signs': [], 'block_2_signs': [],
                  'block_1_w': [], 'block_1_b': []}
        reduction_map = {'min': np.min, 'max': np.max, 'mean': np.mean}
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        for idx in range(self.hidden_dim_1):
            w = self.block_1.layers[0].weight.data[idx, :]
            b = self.block_1.layers[0].bias.data[idx]
            dist = (torch.abs(torch.matmul(x, w) + b) / torch.norm(w, p=2))
            result['block_1'].append(dist.cpu().numpy())
            result['block_1_signs'].append(np.sign((torch.matmul(x, w)+b).type(torch.float).cpu().numpy()))
            result['block_1_w'].append(w.cpu().numpy())
            result['block_1_b'].append(b.cpu().numpy())

        x = self.block_1(x)
        for idx in range(self.hidden_dim_2):
            w = self.block_2.layers[0].weight.data[idx, :]
            b = self.block_2.layers[0].bias.data[idx]
            dist = (torch.abs(torch.matmul(x, w) + b) / torch.norm(w, p=2))
            result['block_2'].append(dist.cpu().numpy())
            result['block_2_signs'].append(np.sign((torch.matmul(x, w)+b).type(torch.float).cpu().numpy()))

        result['block_1'] = np.concatenate(result['block_1'], axis=0).reshape((self.hidden_dim_1, batch_size)).T
        result['block_1_signs'] = np.concatenate(result['block_1_signs'], axis=0).reshape((self.hidden_dim_1, batch_size)).T
        result['block_2'] = np.concatenate(result['block_2'], axis=0).reshape((self.hidden_dim_2, batch_size)).T
        result['block_2_signs'] = np.concatenate(result['block_2_signs'], axis=0).reshape((self.hidden_dim_2, batch_size)).T
        if reduction:
            result['block_1_reduction'] = reduction_map[reduction](result['block_1'], axis=1)
            result['block_2_reduction'] = reduction_map[reduction](result['block_2'], axis=1)
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


class MLP4Layers(ContinualBackbone):
    def __init__(self, multi_head=False, num_classes_per_head=None,
                 input_dim=784, hidden_dim_1=200, hidden_dim_2=200,
                 hidden_dim_3=200, hidden_dim_4=200, output_dim=10,
                 dropout_prob=0.0, activation='ReLU', include_final_layer_act=False):
        # model variables
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.hidden_dim_3 = hidden_dim_3
        self.hidden_dim_4 = hidden_dim_4
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        super(MLP4Layers, self).__init__(multi_head, num_classes_per_head)
        self.block_1 = FCBlock(self.input_dim, self.hidden_dim_1, self.dropout_prob, True, activation)
        self.block_2 = FCBlock(self.hidden_dim_1, self.hidden_dim_2, self.dropout_prob, True, activation)
        self.block_3 = FCBlock(self.hidden_dim_2, self.hidden_dim_3, self.dropout_prob, True, activation)
        self.block_4 = FCBlock(self.hidden_dim_3, self.hidden_dim_4, self.dropout_prob, True, activation)
        self.block_5 = FCBlock(self.hidden_dim_4, self.output_dim, 0.0, include_final_layer_act, activation)
    
    @torch.no_grad()
    def record_activations(self, x, detach=False):
        result = {}
        x = x.view(x.shape[0], -1)
        
        out = self.block_1(x)
        result['block_1'] = out if not detach else out.detach().clone()

        out = self.block_2(out)
        result['block_2'] = out if not detach else out.detach().clone()

        out = self.block_3(out)
        result['block_3'] = out if not detach else out.detach().clone()

        out = self.block_4(out)
        result['block_4'] = out if not detach else out.detach().clone()

        result['total'] = torch.cat((result['block_1'], result['block_2'],
                                     result['block_3'], result['block_4']), dim=1)
        return result
    
    def forward(self, x, task_id=None):
        x = x.view(x.shape[0], -1)
        out = self.block_1(x)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        if self.multi_head:
            offset1 = int((task_id - 1) * self.num_classes_per_head)
            offset2 = int(task_id * self.num_classes_per_head)
            out[:, :offset1].data.fill_(-10e10)
            out[:, offset2:].data.fill_(-10e10)
        return out


if __name__ == "__main__":
    net = MLP2Layers(input_dim=2, hidden_dim_1=4, hidden_dim_2=4, output_dim=2)
    x = torch.randn((3, 2))
    result = net.record_distance_to_boundary(x, reduction='min')
    # print(net.get_block_params(1))
    # print(np.concatenate((net.get_block_params(1)['weight'], net.get_block_params(1)['bias']), axis=1))
    for key in result:
        print(f"{key}:", result[key])
        print('--'*10)
    # print()
    # print(np.multiply(result['block_1_signs'], result['block_1']))
    # print(result['block_1'])
    # print(result['block_1_T'])
    # new_block_1 = np.concatenate(result['block_1'], axis=0).reshape((4, 8))
    # print(new_block_1)
    
    # print(net.block_1.layers[0].weight[0,:].shape)
    # print(net.block_1.layers[0].bias.data.shape)
    # x = torch.randn((3, 2))
    # w = net.block_1.layers[0].weight.data[0,:]
    # b = net.block_1.layers[0].bias.data[0]
    # print(torch.matmul(x, w) + b)
    # block_1_dist = torch.abs(torch.matmul(x, w) + b) / torch.norm(w, p=2)
    # print(block_1_dist)
