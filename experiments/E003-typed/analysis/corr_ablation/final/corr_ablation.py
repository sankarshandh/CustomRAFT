import sys, json, numpy as np, torch
sys.path.insert(0, r'd:\IISc\MotionExperiments\CustomRAFT')
import core.custom_raft as cr
from evaluate import load_model, _loader
from core.utils.utils import InputPadder

ref = sys.argv[1] if len(sys.argv) > 1 else 'E003'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
model = load_model(ref).eval()
Orig = cr.CorrPyramid
class ZeroCorr(Orig):
    def __call__(self, coords): return torch.zeros_like(super().__call__(coords))
fnet_fwd = model.fnet.forward
def same_fmap(clip):
    f1, _ = fnet_fwd(clip); return f1, f1

def run(cond, loader):
    cr.CorrPyramid = ZeroCorr if cond == 'corr=0' else Orig
    model.fnet.forward = same_fmap if cond == 'fmap2=fmap1' else fnet_fwd
    e = []
    with torch.no_grad():
        for s in loader:
            clip = s['clip'].cuda().float(); pad = InputPadder(clip.shape); (clip,) = pad.pad(clip)
            if cond == 'zero flow':
                flow = torch.zeros_like(s['flow'][0])
            else:
                _, fu = model(clip, iters=32, test_mode=True); flow = pad.unpad(fu[0]).cpu()
            e.append(torch.linalg.vector_norm(flow - s['flow'][0], dim=0).view(-1).numpy())
    r = float(np.concatenate(e).mean()); print(cond, round(r, 4), flush=True); return r

base = _loader(model, r'D:\IISc\Motion Datasets\Sintel', 'training', 'clean', num_workers=0)
keep = sorted(set(np.linspace(0, len(base.dataset) - 1, N).astype(int).tolist()))
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(base.dataset, keep), batch_size=1, num_workers=0)
out = {c: run(c, loader) for c in ['normal', 'corr=0', 'fmap2=fmap1', 'zero flow']}
print(json.dumps({'ref': ref, 'pairs': len(keep), 'pass': 'clean', 'iters': 32, 'epe': out}, indent=1))

