
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, Flatten
from torch.nn import LeakyReLU, ReLU
from torch.nn.functional import normalize


class Branch(Module):

    def __init__(self, input_shape, output_shape):
        super(Branch, self).__init__()

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
            # Flatten(start_dim=1, end_dim=-1),
            # Linear(6 * 10 * 16, output_shape)
        ]

        self.network = Sequential(*layers)
        self.fc = Sequential(*[Linear(6 * 10 * 16, 16)])

    def forward(self, x, training=None, mask=None):
        # for layer in self.layers:
        #     x = layer(x)
        x = self.network(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Siamese(Module):

    def __init__(self, input_shape=(1, 180, 320), embedding_size=16):
        super(Siamese, self).__init__()
        self.branch_model = Branch(input_shape, embedding_size)

    def _forward_one_branch(self, x):
        x = self.branch_model(x)
        x = x.view(x.shape[0], -1)
        x = normalize(x, p=2)
        return x

    def forward(self, inputs):
        inp1, inp2 = inputs
        x1 = self._forward_one_branch(inp1)
        x2 = self._forward_one_branch(inp2)
        return [x1, x2]

    def get_config(self):
        pass

if __name__ == '__main__':
    import sys
    from ContrastiveLoss import ContrastiveLoss

    siamese = Siamese(input_shape=(1, 180, 320))
    criterion = ContrastiveLoss(margin=1.0)

    import numpy as np
    np.random.seed(0)
    N = 2
    x1 = torch.tensor(np.random.rand(N, 1, 180, 320), dtype=torch.float32)
    x2 = torch.tensor(np.random.rand(N, 1, 180, 320), dtype=torch.float32)

    y1 = torch.tensor(np.random.rand(N), dtype=torch.float32)
    y_zeros = torch.zeros(N)
    y_ones = torch.ones(N)

    y_true = torch.where(y1 > 0, y_ones, y_zeros)

    y_pred = siamese((x1, x2))
    loss = criterion(y_true, y_pred)
    f1, f2 = y_pred
    print(f1)
    print(f2)
    print(loss)
    sys.exit()