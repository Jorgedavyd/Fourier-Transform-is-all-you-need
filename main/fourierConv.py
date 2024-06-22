import torch
from torch import Tensor, nn
from typing import Sequence, Tuple
from fourierConv2d import conv2d_fwd, conv2d_bwd
from torch.autograd import Function

class conv2d(Function):
    @staticmethod
    def forward(ctx: torch.Any, *args: torch.Any, **kwargs: torch.Any) -> torch.Any:
        return conv2d_fwd(input, weight, bias)
    @staticmethod
    def backward(ctx: torch.Any, .Any) -> torch.Any:
        return conv2d_bwd(grad_O, O, input, weight, bias)


class FourierConv2d(nn.Module):
    def __init__(self, *architecture: Tuple[int, int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], bool]) -> None:
        super().__init__()
        
        self.weights = nn.ParameterList()
        for in_channels, out_channels, kernel_size, stride, padding in architecture:
            self.weights.append(
                torch.empty(out_channels, in_channels, *kernel_size)
            )
            self.bias.append(
                torch.empty(out_channels)
            )
    
    def forward(self, ) -> Tensor:
        return conv2d()
        