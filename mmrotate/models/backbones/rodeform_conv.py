import torch
import torch.nn as nn
import math

from mmcv.cnn import CONV_LAYERS
from mmcv.ops import DeformConv2d, deform_conv2d
from mmcv.utils import print_log

from torch import Tensor
from torch.nn.modules.utils import _pair



@CONV_LAYERS.register_module('RDCN')
class DeformConv2dPack(DeformConv2d):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    The offset tensor is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
    The spatial arrangement is like:

    .. code:: text

        (x0, y0) (x1, y1) (x2, y2)
        (x3, y3) (x4, y4) (x5, y5)
        (x6, y6) (x7, y7) (x8, y8)

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_theta = nn.Conv2d(
            self.in_channels,
            self.deform_groups,
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.conv_alpha = nn.Conv2d(
            self.in_channels,
            self.deform_groups,
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_theta.weight.data.zero_()
        self.conv_theta.bias.data.zero_()
        self.conv_alpha.weight.data.zero_()
        self.conv_alpha.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        theta = self.conv_theta(x)
        theta = torch.tanh(theta) * (math.pi / 2)
        alphas = self.conv_alpha(x)
        alphas = torch.sigmoid(alphas) * 2
        offset = _get_offset(theta, alphas).contiguous()
        return deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups,
                             False, self.im2col_step)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, DeformConvPack loads previous benchmark models.
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                f'DeformConv2dPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)


origin_coors = torch.tensor(
    [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0],
        [1, 1]],
    dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu').reshape(1, 18, 1, 1)

def _get_offset(theta: Tensor, alphas: Tensor) -> Tensor:
    """Get offset from theta and alphas.

    Args:
        theta (Tensor): The theta tensor with shape (B, 1, H, W).
        alphas (Tensor): The alphas tensor with shape (B, 1, H, W).

    Returns:
        Tensor: The offset.
    """
    B, _, H, W = theta.size()
    coors = torch.mul(alphas, origin_coors)
    coors = coors.permute(0, 2, 3, 1).reshape(-1, 9, 2)
    rotated_coors = _batch_rotate_coordinates(coors, theta.reshape(-1))
    rotated_coors = rotated_coors.reshape(B, H, W, 18).permute(0, 3, 1, 2)
    offset = rotated_coors - origin_coors
    
    return offset

def _batch_rotate_coordinates(coors: Tensor, angles: Tensor) -> torch.Tensor:
    """Rotate coordinates by a given angle.

    Args:
        coors (torch.Tensor): The coordinates tensor with shape (B, N, 2).
        angles (torch.Tensor): The rotation angles tensor with shape (B,).

    Returns:
        torch.Tensor: The rotated coordinates.
    """
    batch_size = coors.size(0)
    num_points = coors.size(1)

    # Create rotation matrices for each angle in the batch
    rotation_matrices = torch.zeros((batch_size, 2, 2), dtype=torch.float32, device=coors.device)
    rotation_matrices[:, 0, 0] = torch.cos(angles)
    rotation_matrices[:, 0, 1] = -torch.sin(angles)
    rotation_matrices[:, 1, 0] = torch.sin(angles)
    rotation_matrices[:, 1, 1] = torch.cos(angles)

    # Rotate the coordinates
    rotated_coors = torch.matmul(coors, rotation_matrices)

    return rotated_coors