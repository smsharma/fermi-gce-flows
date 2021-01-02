import torch
from torch import nn
import torch.nn.functional as F

from models.healpix_pool_unpool import Healpix
from models.laplacians import get_healpix_laplacians
from models.layers import SphericalChebBNPool

# pylint: disable=W0223


class SphericalGraphCNN(nn.Module):
    """Spherical GCNN Autoencoder.
    """

    def __init__(self, nside_list, indexes_list, kernel_size=4, laplacian_type="combinatorial", fc1_out_dim=2048, fc2_out_dim=512, n_aux_var=1):
        """Initialization.

        Args:
            pooling_class (obj): One of three classes of pooling methods
            N (int): Number of pixels in the input image
            kernel_size (int): chebychev polynomial degree
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.pooling_class = Healpix(mode="max")

        self.n_aux_var = n_aux_var

        self.laps = get_healpix_laplacians(nside_list=nside_list, laplacian_type=laplacian_type, indexes_list=indexes_list)
        self.cnn_layers = []

        for i, (in_ch, out_ch) in enumerate([(1, 32), (32, 64), (64, 128), (128, 256), (256, 256), (256, 256), (256, 256)]):
            layer = SphericalChebBNPool(in_ch, out_ch, self.laps[i], self.pooling_class.pooling, self.kernel_size)
            setattr(self, "layer_{}".format(i), layer)
            self.cnn_layers.append(layer)

        # Feed auxiliary variables into fully-connected part
        self.fc1 = nn.Linear(256 + self.n_aux_var, fc1_out_dim)
        self.fc2 = nn.Linear(fc1_out_dim, fc2_out_dim)

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.

        Returns:
            :obj:`torch.Tensor`: output
        """

        # Initialize tensor
        x = x.view(-1, 16384 + self.n_aux_var, 1)

        # Extract auxiliary variable
        x_aux = x[:, -self.n_aux_var:, :]
        x_aux = x_aux.view(-1, 1, self.n_aux_var)
        
        # Extract map to input into convolutional layers
        x = x[:, :-self.n_aux_var, :]

        # Convolutional layers
        for layer in self.cnn_layers:
            x = layer(x)

        # Concatenate auxiliary variable along last dimension
        x = torch.cat([x, x_aux], -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x[:, 0, :]
