# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from torch import Tensor, nn

from sbi.utils.sbiutils import standardizing_net


class StandardizeInputs(nn.Module):
    def __init__(self, embedding_net_x, embedding_net_y, batch_x, batch_y, z_score_x, z_score_y):
        super().__init__()
        self.embedding_net_x = embedding_net_x
        self.embedding_net_y = embedding_net_y

        self.batch_x = batch_x
        self.batch_y = batch_y
        
        if z_score_x:
            self.standardizing_net_x = standardizing_net(batch_x)
        else: 
            self.standardizing_net_x = nn.Identity()

        if z_score_y:
            self.standardizing_net_y = standardizing_net(batch_y)
        else: 
            self.standardizing_net_y = nn.Identity()


    def forward(self, x, x_aux, theta):
        
        x = self.standardizing_net_x(x)

        theta = self.standardizing_net_y(theta)
        theta = self.embedding_net_y(theta)
        
        x = self.embedding_net_x(x, x_aux, theta)
        
        return x

class SequentialMulti(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def build_mlp_mixed_classifier(batch_x: Tensor = None, batch_x_aux: Tensor = None, batch_y: Tensor = None, z_score_x: bool = True, z_score_y: bool = True, hidden_features: int = 50, embedding_net_x: nn.Module = nn.Identity(), embedding_net_y: nn.Module = nn.Identity(), additional_layers: bool = True) -> nn.Module:
    """Builds MLP classifier.

    In SNRE, the classifier will receive batches of thetas and xs.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        z_score_y: Whether to z-score ys passing into the network.
        hidden_features: Number of hidden features.
        embedding_net_x: Optional embedding network for x.
        embedding_net_y: Optional embedding network for y.

    Returns:
        Neural network.
    """

    # Infer the output dimensionalities of the embedding_net by making a forward pass.
    x_numel = embedding_net_x(batch_x[:1], batch_x_aux[:1], batch_y[:1]).numel()

    if additional_layers:
        neural_net = nn.Sequential(nn.Linear(x_numel, hidden_features), nn.BatchNorm1d(hidden_features), nn.ReLU(), nn.Linear(hidden_features, hidden_features), nn.BatchNorm1d(hidden_features), nn.ReLU(), nn.Linear(hidden_features, 1),)
    else:
        neural_net = nn.Sequential(nn.ReLU(), nn.Linear(x_numel, 1),)

    input_layer = StandardizeInputs(embedding_net_x, embedding_net_y, batch_x, batch_y, z_score_x=z_score_x, z_score_y=z_score_y)

    neural_net = SequentialMulti(input_layer, neural_net)

    return neural_net