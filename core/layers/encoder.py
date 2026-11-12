"""Standard residual CNN feature encoder (RAFT-style, output at 1/8 resolution)."""
import torch.nn as nn


def make_norm(norm_fn, channels):
    if norm_fn == 'instance':
        return nn.InstanceNorm2d(channels)
    if norm_fn == 'batch':
        return nn.BatchNorm2d(channels)
    if norm_fn == 'group':
        return nn.GroupNorm(max(channels // 8, 1), channels)
    if norm_fn == 'none':
        return nn.Identity()
    raise ValueError(norm_fn)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='instance', stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.norm1, self.norm2 = make_norm(norm_fn, planes), make_norm(norm_fn, planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride),
                                            make_norm(norm_fn, planes))

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    """1x1 reduce -> 3x3 (strided) -> 1x1 expand, as in RAFT's SmallEncoder."""

    def __init__(self, in_planes, planes, norm_fn='instance', stride=1):
        super().__init__()
        mid = planes // 4
        self.conv1 = nn.Conv2d(in_planes, mid, 1)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(mid, planes, 1)
        self.norm1, self.norm2 = make_norm(norm_fn, mid), make_norm(norm_fn, mid)
        self.norm3 = make_norm(norm_fn, planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride),
                                            make_norm(norm_fn, planes))

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


BLOCKS = {'residual': ResidualBlock, 'bottleneck': BottleneckBlock}


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)) and m.weight is not None:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class CNNEncoder(nn.Module):
    """conv7x7/2 -> 2 blocks (dims[0]) -> 2 blocks /2 (dims[1]) -> 2 blocks /2 (dims[2]) -> 1x1 conv.

    Input : (B, in_channels, H, W), or (B, in_channels, T, H, W) where every frame is encoded independently.
    Output: (B, output_dim, H/8, W/8)  [or (B, output_dim, T, H/8, W/8)].
    in_channels can be 3 (RGB) or the channel count of a preceding layer (e.g. 2N ON/OFF DoG maps).
    block='bottleneck' with dims=(32, 64, 96) matches RAFT-small's SmallEncoder.
    """

    def __init__(self, in_channels=3, output_dim=256, norm_fn='instance', dims=(64, 96, 128), dropout=0.0,
                 block='residual'):
        super().__init__()
        Block = BLOCKS[block]
        self.conv1 = nn.Conv2d(in_channels, dims[0], 7, stride=2, padding=3)
        self.norm1 = make_norm(norm_fn, dims[0])
        self.relu = nn.ReLU(inplace=True)

        layers, in_planes = [], dims[0]
        for i, d in enumerate(dims):
            stride = 1 if i == 0 else 2
            layers += [Block(in_planes, d, norm_fn, stride), Block(d, d, norm_fn, 1)]
            in_planes = d
        self.layers = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(in_planes, output_dim, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        init_weights(self)

    def forward(self, x):
        is_5d = x.dim() == 5
        if is_5d:  # fold time into batch
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.dropout(self.conv2(self.layers(x)))
        if is_5d:
            x = x.reshape(b, t, *x.shape[1:]).permute(0, 2, 1, 3, 4)
        return x
