
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, Flatten
from torch.nn import LeakyReLU, ReLU
from torch.nn.functional import normalize


class Siamese(Module):

    def __init__(self):
        super(Siamese, self).__init__()

        input_shape = (1, 180, 320)
        embedding_size = 16

        layers = [
            Conv2d(input_shape[0], 4, kernel_size=7, stride=2, padding=3),
            LeakyReLU(negative_slope=0.1, inplace=True),
            Conv2d(4, 8, kernel_size=5, stride=2, padding=2),
            ReLU(inplace=True),
            Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            ReLU(inplace=True),
            Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            ReLU(inplace=True),
            Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            ReLU(inplace=True),
            Flatten(start_dim=1, end_dim=-1),
            Linear(6 * 10 * 16, embedding_size)
        ]
        self.branch = Sequential(*layers)

    def _forward_one_branch(self, x):
        x = self.branch(x)
        x = x.view(x.shape[0], -1)
        x = normalize(x, p=2)
        return x

    def forward(self, x1, x2):
        x1 = self._forward_one_branch(x1)
        x2 = self._forward_one_branch(x2)
        return x1, x2

    def get_config(self):
        pass

if __name__ == '__main__':
    import sys
    from ContrastiveLoss import ContrastiveLoss

    siamese = Siamese()
    criterion = ContrastiveLoss(margin=1.0)

    import numpy as np
    np.random.seed(0)
    N = 2
    x1 = torch.tensor(np.random.rand(N, 1, 180, 320), dtype=torch.float32)
    x2 = torch.tensor(np.random.rand(N, 1, 180, 320), dtype=torch.float32)

    y1 = torch.tensor(np.random.rand(N, 1), dtype=torch.float32)
    y_zeros = torch.zeros(N, 1)
    y_ones = torch.ones(N, 1)

    y_true = torch.where(y1 > 0, y_ones, y_zeros)
    y_true = torch.squeeze(y_true)

    f1, f2 = siamese(x1, x2)
    loss = criterion(f1, f2, y_true)
    print(f1)
    print(f2)
    print(loss)
    sys.exit()
