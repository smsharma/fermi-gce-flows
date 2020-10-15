import torch
from pyro.distributions.torch_distribution import TorchDistributionMixin


class LogUniformTorch(torch.distributions.TransformedDistribution):
    def __init__(self, low, high):
        super(LogUniformTorch, self).__init__(torch.distributions.Uniform(low.log(), high.log()), torch.distributions.ExpTransform())


class LogUniform(LogUniformTorch, TorchDistributionMixin):
    def expand(self, batch_shape):
        validate_args = self.__dict__.get("validate_args")
        low = self.low.expand(batch_shape)
        high = self.high.expand(batch_shape)
        return LogUniformTorch(low, high, validate_args=validate_args)

