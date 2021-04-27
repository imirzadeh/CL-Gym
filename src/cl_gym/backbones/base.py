import torch
import torch.nn as nn


class ContinualBackbone(nn.Module):
    def __init__(self, multi_head=False, num_classes_per_head=None):
        super(ContinualBackbone, self).__init__()
        self.multi_head = multi_head
        self.num_classes_per_head = num_classes_per_head



# if __name__ == "__main__":
#     data = torch.randn((2, 784))
#     net = MLP2Layers(input_dim=784, output_dim=4, hidden_dim_1=5, hidden_dim_2=5, dropout_prob=0.25, num_classes_per_head=2, multi_head=True)
#     net.train()
#     # print(net(data, 1))
#     # print(net(data, 2))
#     # net.eval()
#     print(net.record_activations(data))
#     print(net)
#     # print(net.blocks.keys())
#     for n, p in net.named_parameters():
#         print(n)
