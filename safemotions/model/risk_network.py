import torch.nn as nn

class RiskNetwork(nn.Module):
    def __init__(self, input_size, fcnet_hiddens, dropout=None, activation='relu', last_activation='sigmoid'):
        super().__init__()
        layers = []
        act_map = {
            "relu": nn.ReLU,
            "selu": nn.SELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "swish": nn.SiLU,
            "leaky_relu": nn.LeakyReLU,
        }
        prev_size = input_size
        for h in fcnet_hiddens:
            layers.append(nn.Linear(prev_size, h))
            layers.append(act_map[activation]())
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        if last_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)