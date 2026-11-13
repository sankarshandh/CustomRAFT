"""CustomRAFT: RAFT-small with a bio-inspired feature encoder.

    clip (B, 3, T, H, W) --BioFeatureEncoder (DoG -> Gabor -> head)--> fmap1, fmap2 (1/8, 128-d)
    clip[:, :, ref]      --CNNEncoder (context)---------------------> net (tanh, 96), inp (relu, 64)
    CorrPyramid(fmap1, fmap2)  4 levels, radius 3
    12 (train) / 32 (eval) iterations of the RAFT-small ConvGRU update, flow upsampled x8 bilinearly.

The predicted flow is flow(clip[:, :, ref] -> clip[:, :, ref + 1]), ref = model.ref_offset.
"""
import torch
import torch.nn as nn

from core.layers.correlation import CorrPyramid
from core.layers.encoder import CNNEncoder
from core.layers.frame_encoder import BioFeatureEncoder
from core.update import SmallUpdateBlock
from core.utils.utils import coords_grid, upflow8


class CustomRAFT(nn.Module):
    def __init__(self, temporal_mode='free', corr_levels=4, corr_radius=3, hidden_dim=96, context_dim=64,
                 dropout=0.0, mixed_precision=False, **fnet_kwargs):
        super().__init__()
        self.hidden_dim, self.context_dim = hidden_dim, context_dim
        self.corr_levels, self.corr_radius = corr_levels, corr_radius
        self.mixed_precision = mixed_precision

        self.fnet = BioFeatureEncoder(output_dim=128, temporal_mode=temporal_mode, **fnet_kwargs)
        self.cnet = CNNEncoder(3, output_dim=hidden_dim + context_dim, norm_fn='none', dims=(32, 64, 96),
                               block='bottleneck', dropout=dropout)
        self.update_block = SmallUpdateBlock(corr_levels * (2 * corr_radius + 1) ** 2, hidden_dim, context_dim)

    @property
    def num_frames(self):
        return self.fnet.num_frames

    @property
    def ref_offset(self):
        return self.fnet.ref_offset

    def bio_parameters(self):
        return self.fnet.bio_parameters()

    def forward(self, clip, iters=12, flow_init=None, test_mode=False, clip2=None):
        """clip: (B, 3, num_frames, H, W) in [0, 255] (uint8 or float), H and W divisible by 8.
        clip2 (optional, same shape): asymmetric views. fmap1 comes from clip and fmap2 from clip2 (the window
        centred on the next frame); the context encoder still sees clip. None = the original model.
        Returns a list of (B, 2, H, W) predictions, or (flow_low, flow_up) when test_mode."""
        clip = (2 * (clip.float() / 255.0) - 1.0).contiguous()
        amp = torch.autocast('cuda', enabled=self.mixed_precision)

        with amp:
            fmap1, fmap2 = self.fnet(clip)
            if clip2 is not None:
                fmap2 = self.fnet((2 * (clip2.float() / 255.0) - 1.0).contiguous())[1]
        with torch.autocast('cuda', enabled=False):  # all-pairs correlation in fp32
            corr_fn = CorrPyramid(fmap1.float(), fmap2.float(), self.corr_levels, self.corr_radius)

        with amp:
            cnet = self.cnet(clip[:, :, self.ref_offset])
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net, inp = torch.tanh(net), torch.relu(inp)

        b, _, h, w = fmap1.shape
        coords0 = coords_grid(b, h, w, device=clip.device)
        coords1 = coords0.clone()
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with amp:
                net, _, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow.float()
            flow_up = upflow8(coords1 - coords0)
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
        return flow_predictions
