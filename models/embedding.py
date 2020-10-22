import torch
from torch import nn
import torch.nn.functional as F

from models.chebyshev import SphericalChebConv
from models.healpix_pool_unpool import Healpix
from models.laplacians import get_healpix_laplacians
from models.layers import SphericalChebBNPool


class SphericalGraphCNN(nn.Module):
    """Spherical GCNN Autoencoder.
    """

    def __init__(self, nside_list, indexes_list, kernel_size=4, laplacian_type="combinatorial"):
        """Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            kernel_size (int): chebychev polynomial degree
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.pooling_class = Healpix(mode="max")

        self.laps = get_healpix_laplacians(nside_list=nside_list, laplacian_type=laplacian_type, indexes_list=indexes_list)
        self.cnn_layers = []

        for i, (in_ch, out_ch) in enumerate([(1, 32), (32, 64), (64, 128), (128, 256), (256, 256), (256, 256), (256, 256)]):
            layer = SphericalChebBNPool(in_ch, out_ch, self.laps[i], self.pooling_class.pooling, self.kernel_size)
            setattr(self, "layer_{}".format(i), layer)
            self.cnn_layers.append(layer)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 64)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.

        Returns:
            :obj:`torch.Tensor`: output
        """
        x = x.view(-1, 16384, 1)
        
        for layer in self.cnn_layers:
            x = layer(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x[:, 0, :]
