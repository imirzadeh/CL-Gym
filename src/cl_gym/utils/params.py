import torch


class Params:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')