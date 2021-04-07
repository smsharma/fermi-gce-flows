import torch
from torch import nn
import torch.nn.functional as F

# try:
from torch_geometric.nn import ChebConv, GCNConv, LEConv
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.data import Data
# except:
#     print("PyTorch Geometric not importable")

from models.chebyshev import SphericalChebConv

# pylint: disable=W0223


class SphericalChebBNPool(nn.Module):
    """Building Block with a pooling/unpooling, a calling the SphericalChebBN block.
    """

    def __init__(self, in_channels, out_channels, lap, pooling, kernel_size, activation="relu"):
        """Initialization.

        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()
        self.spherical_cheb = SphericalChebConv(in_channels, out_channels, lap, kernel_size)
        self.pooling = pooling
        self.batchnorm = nn.BatchNorm1d(out_channels)

        if activation == "relu":
            self.activation_function = F.relu
        elif activation == "selu":
            self.activation_function = F.selu
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward Pass.

        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]

        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.spherical_cheb(x)
        x = self.batchnorm(x.permute(0, 2, 1))
        x = self.activation_function(x.permute(0, 2, 1))
        x = self.pooling(x)

        return x

class SphericalChebBNPoolGeom(nn.Module):
    """Building Block with a pooling/unpooling, a calling the SphericalChebBN block.
    """

    def __init__(self, in_channels, out_channels, adj, pooling, kernel_size, laplacian_type, indexes_list, activation="relu", conv_type="chebconv"):
        """Initialization.
        Args:
            in_channels (int): initial number of channels.
            out_channels (int): output number of channels.
            lap (:obj:`torch.sparse.FloatTensor`): laplacian.
            pooling (:obj:`torch.nn.Module`): pooling/unpooling module.
            kernel_size (int, optional): polynomial degree. Defaults to 3.
        """
        super().__init__()

        self.laplacian_type = laplacian_type
        assert self.laplacian_type in ['normalized', 'combinatorial'], 'Invalid normalization'

        edge_index = adj.indices()
        edge_weight = adj.values()

        self.register_buffer("edge_index", edge_index)
        if edge_weight is None:
            setattr(self, 'edge_weight', None)
        else:
            self.register_buffer("edge_weight", edge_weight)
        
        # Graph convolution layer to use
        if conv_type == "chebconv":
            
            # Get lambda_max for non-symmetric Laplacian
            data = Data(num_nodes=len(indexes_list), edge_index=edge_index, edge_attr=edge_weight)
            lambda_max = LaplacianLambdaMax(normalization='sym' if laplacian_type == 'normalized' else None, is_undirected=True)(data).lambda_max
            self.register_buffer('lambda_max', torch.tensor(lambda_max, dtype=torch.float32))
            
            self.conv = ChebConv(in_channels, out_channels, kernel_size, normalization='sym' if self.laplacian_type == 'normalized' else None)
            self.conv_kwargs = {'lambda_max':self.lambda_max if self.laplacian_type == "combinatorial" else None}
        
        elif conv_type == "gcn":
            
            self.conv = GCNConv(in_channels, out_channels)
            self.conv_kwargs = {}
        
        else:
            raise NotImplementedError

        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.pooling = pooling

        if activation == "relu":
            self.activation_function = F.relu
        elif activation == "selu":
            self.activation_function = F.selu
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.tensor`): input [batch x vertices x channels/features]
        Returns:
            :obj:`torch.tensor`: output [batch x vertices x channels/features]
        """
        x = self.conv(x, self.edge_index, self.edge_weight, **self.conv_kwargs)
        x = self.batchnorm(x.permute(0, 2, 1))
        x = self.activation_function(x.permute(0, 2, 1))
        x = self.pooling(x)
        return x