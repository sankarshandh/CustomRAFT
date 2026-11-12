"""Bio-inspired frame (feature) encoder: spatiotemporal DoG -> oriented Gabor -> light strided CNN head.

Both filter stages use 'valid' temporal convolution, so a clip of num_frames frames yields
exactly two output time steps, whose temporal windows are offset by one frame:

    DoG window (dog_temporal) and Gabor window (gabor_temporal) together span
    W = dog_temporal + gabor_temporal - 1 frames; output 0 sees clip[0 : W], output 1 sees clip[1 : W+1].

Because a fixed spatiotemporal filter commutes with translation, the two outputs are shifted copies of
each other (displaced by the flow), which is what the correlation volume needs. With centred kernels,
output 0 is centred on clip frame ref_offset, so the matching ground truth is flow(ref -> ref + 1).
"""
import torch.nn as nn

from core.layers.encoder import BottleneckBlock, init_weights
from core.layers.orientation import OrientationTunedLayer
from core.layers.spatiotemporal import SpatioTemporalDoG


class BioFeatureEncoder(nn.Module):
    """Input (B, 3, num_frames, H, W) -> (fmap1, fmap2), each (B, output_dim, H/8, W/8)."""

    def __init__(self, output_dim=128, dog_filters=8, dog_spatial=21, dog_temporal=5,
                 temporal_mode='free', gabor_orientations=8, gabor_scales=2, gabor_kernel=15,
                 gabor_temporal=3, head_dims=(64, 96), norm_fn='instance', dog_gain_mode='learned'):
        super().__init__()
        self.dog = SpatioTemporalDoG(3, dog_filters, spatial_size=dog_spatial, temporal_size=dog_temporal,
                                     temporal_mode=temporal_mode, temporal_padding='valid', gain_mode=dog_gain_mode)
        self.gabor = OrientationTunedLayer(2 * dog_filters, gabor_orientations, gabor_scales,
                                           kernel_size=gabor_kernel, temporal_size=gabor_temporal,
                                           stride=2, mode='quadrature', temporal_padding='valid')
        gabor_out = 2 * self.gabor.num_filters

        self.norm = nn.InstanceNorm2d(gabor_out)
        self.head = nn.Sequential(
            BottleneckBlock(gabor_out, head_dims[0], norm_fn, stride=2),       # 1/4
            BottleneckBlock(head_dims[0], head_dims[1], norm_fn, stride=2),    # 1/8
            BottleneckBlock(head_dims[1], head_dims[1], norm_fn, stride=1),
            nn.Conv2d(head_dims[1], output_dim, 1),
        )
        init_weights(self.head)

        self.num_frames = dog_temporal + gabor_temporal  # window W plus the one-frame offset
        self.ref_offset = (dog_temporal - 1) // 2 + (gabor_temporal - 1) // 2

    def bio_parameters(self):
        """Filter-shape parameters (sigmas, delays, k, theta, frequency, speed, gains).
        channel_mix is excluded: it is an ordinary weight matrix."""
        return [p for m in (self.dog, self.gabor) for n, p in m.named_parameters() if n != 'channel_mix']

    def forward(self, clip):
        assert clip.dim() == 5 and clip.shape[2] == self.num_frames, \
            f'expected (B, 3, {self.num_frames}, H, W), got {tuple(clip.shape)}'
        x = self.gabor(self.dog(clip))  # (B, C, 2, H/2, W/2)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.head(self.norm(x))
        x = x.reshape(b, t, *x.shape[1:])
        return x[:, 0], x[:, 1]
