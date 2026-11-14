"""Run the standard tests and visualisations on an experiment's checkpoints and write a report to tests/.

    python tools/pipeline.py E004                          # final checkpoint (full) + 4 evenly spaced steps (light)
    python tools/pipeline.py E004 --steps 50000,final      # explicit checkpoints (final is always run full)
    python tools/pipeline.py E004 --only motion,report     # some stages only
    python tools/pipeline.py E004 --force                  # redo stages that are already done
    python tools/pipeline.py --report_only                 # rebuild every report.md / index.html / tests/README.md

Output (never in kb/): tests/<experiment>/<step>/
    metrics/        Sintel val numbers (read from the run's val_metrics.jsonl when that step was validated there,
                    otherwise computed), training curves vs the parent              [light: numbers only]
    frontend/       silencing_check.json (every DoG neuron: widths, gain, raw response, attenuation),
                    temporal.json (analyse_temporal.py: temporal kernels, drive, Gabor input mix, freeze test),
                    filters_trajectory.png (filter parameters over training)                  [light: silencing]
    patterns/       eval_random_patterns.py: EPE on the random-pattern cases occluded / hollow / roi (+ visible
                    control), against texture flow and shape flow, per region      [light: 4 scenes, clean]
    probes/         probe_random_patterns.py: per-neuron responses to moving disks / rings / second-order motion
    static_viz/     show_layer_outputs.py, show_corr_inputs.py, show_context.py figures (+ context ablation EPE)
    motion/         render_motion.py: frame sequences + .mp4 of flow and layer panels, one scene per case,
                    DoG time courses, EPE with full clip vs pair-only input                [light: 1 scene]
    summary.json    every number above, plus the auto-flags
    report.md       key numbers vs the parent, an "unusual" list (simple threshold rules from configs/pipeline.json),
                    the per-neuron table, the training trajectory, links to all figures
    index.html      open in a browser: motion of this checkpoint and its parent, played in sync
tests/README.md lists every checkpoint that has been through the pipeline.

Each stage writes <stage>/done.json; a stage that is done is skipped unless --force. A failing stage is
recorded and the others still run. Stimuli and thresholds: configs/pipeline.json. Nothing here draws conclusions:
report.md is the fact sheet that findings are drafted from, together with the user.
"""
import argparse
import csv
import datetime
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.utils import paths  # noqa: E402

CONFIG = osp.join(ROOT, 'configs', 'pipeline.json')
STAGES = ['metrics', 'frontend', 'patterns', 'probes', 'static_viz', 'motion']
LIGHT_STAGES = {'metrics', 'frontend', 'patterns', 'motion'}
PATTERN_CASES = ('occluded', 'hollow', 'roi', 'visible')


# ----------------------------------------------------------------------------------------------- experiments
def exp_config(exp_id):
    with open(osp.join(paths.exp_dir(exp_id), 'config.json')) as f:
        return json.load(f)


def run_dir(exp_id, run=None):
    """runs/<run>/ of an experiment; default: its main run (config.json main_run)."""
    return osp.join(paths.exp_dir(exp_id), 'runs', run or exp_config(exp_id).get('main_run') or '')


def run_checkpoints(exp_id, run):
    """(sorted steps, final.pth or None) in runs/<run>/checkpoints/."""
    d = osp.join(run_dir(exp_id, run), 'checkpoints')
    files = os.listdir(d) if osp.isdir(d) else []
    steps = sorted(int(f[:-4]) for f in files if f.endswith('.pth') and f[:-4].isdigit())
    return steps, (osp.join(d, 'final.pth') if 'final.pth' in files else None)


def ckpt_step(ref):
    """Training step stored in a checkpoint (final.pth has no number in its name)."""
    tag = paths.describe(ref)[1].split('@')[-1]
    if tag.isdigit():
        return int(tag)
    import torch
    return int(torch.load(paths.resolve_ckpt(ref), map_location='cpu', weights_only=False).get('step', -1))


def pick_light_steps(exp_id, n, run=None):
    steps = (run_checkpoints(exp_id, run)[0] if run else paths.checkpoint_steps(exp_id))[:-1]
    # (the last saved step equals, or precedes, the final checkpoint)
    if n <= 0 or not steps:
        return []
    idx = np.unique(np.round(np.linspace(0, len(steps) - 1, n + 2)[1:-1]).astype(int))
    return [steps[i] for i in idx]


def git_commit():
    r = subprocess.run(['git', '-c', f'safe.directory={ROOT}', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                       capture_output=True, text=True)
    dirty = subprocess.run(['git', '-c', f'safe.directory={ROOT}', 'status', '--porcelain', 'core', 'tools',
                            'evaluate.py', 'configs'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return (r.stdout.strip() or 'unknown') + ('+dirty' if dirty else '')


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


# ----------------------------------------------------------------------------------------------- stages
class Ctx:
    def __init__(self, ref, full, cfg):
        self.ref, self.full, self.cfg = ref, full, cfg
        self.exp_dir, self.tag = paths.describe(ref)
        self.exp_id = osp.basename(self.exp_dir)[:4]
        self.run = paths.run_of(ref)  # None = the main run (checkpoints/ of the experiment)
        self.out = paths.tests_dir(ref)
        self.step = ckpt_step(ref)

    def sdir(self, stage):
        return osp.join(self.out, stage)

    def sh(self, stage, args):
        """Run a tool with this interpreter; its output goes to <stage>/log.txt."""
        os.makedirs(self.sdir(stage), exist_ok=True)
        with open(osp.join(self.sdir(stage), 'log.txt'), 'a') as log:
            log.write(f'$ {" ".join(args)}\n')
            log.flush()
            r = subprocess.run([sys.executable, '-u', *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, 'MPLBACKEND': 'Agg'})
        if r.returncode:
            raise RuntimeError(f'{args[0]} failed (exit {r.returncode}); see {stage}/log.txt')


def val_entry(exp_id, step, run=None):
    """(numbers, source) from the run's val_metrics.jsonl at this step; full validation preferred."""
    vj = osp.join(run_dir(exp_id, run), 'val_metrics.jsonl')
    rows = [json.loads(line) for line in open(vj)] if osp.exists(vj) else []
    rows = [r for r in rows if r.get('step') == step]
    full = [r for r in rows if any(k.startswith('full_') for k in r)]
    if full:
        r = full[-1]
        return {k[5:]: v for k, v in r.items() if k.startswith('full_')}, 'val_metrics.jsonl (full validation)'
    if rows:
        r = rows[-1]
        return {k: v for k, v in r.items() if k != 'step'}, 'val_metrics.jsonl (periodic subset)'
    return None, None


def stage_metrics(ctx):
    val, src = val_entry(ctx.exp_id, ctx.step, ctx.run)
    if val is None:
        import torch
        from evaluate import load_model, validate_sintel
        model = load_model(ctx.ref).eval()
        mp = None if ctx.full else ctx.cfg['metrics']['sintel_light_max_pairs']
        val = validate_sintel(model, max_pairs=mp, num_workers=0)
        src = 'computed by pipeline (' + ('full validation' if mp is None else f'{mp} pairs per pass') + ')'
        del model
        torch.cuda.empty_cache()
    keep = ('clean', 'final', 'clean_noc', 'clean_occ', 'final_noc', 'final_occ', 'clean_1px', 'final_1px')
    res = {'step': ctx.step, 'sintel': {k: round(float(val[k]), 4) for k in keep if k in val}, 'sintel_source': src}
    if ctx.full:
        os.makedirs(ctx.sdir('metrics'), exist_ok=True)
        plot_curves(ctx, osp.join(ctx.sdir('metrics'), 'curves.png'))
    return res


def load_curves(exp_id, run=None):
    rd = run_dir(exp_id, run)
    tr, va = [], []
    if osp.exists(osp.join(rd, 'train_metrics.csv')):
        with open(osp.join(rd, 'train_metrics.csv')) as f:
            tr = [(int(r['step']), float(r['epe'])) for r in csv.DictReader(f)]
    if osp.exists(osp.join(rd, 'val_metrics.jsonl')):
        for line in open(osp.join(rd, 'val_metrics.jsonl')):
            r = json.loads(line)
            if 'clean' in r:
                va.append((r['step'], r['clean'], r['final']))
    return tr, va


def plot_curves(ctx, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    exps = [(ctx.exp_id, '-', ctx.run)]
    parent = exp_config(ctx.exp_id).get('parent')
    if parent:
        exps.append((parent[:4], '--', None))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for eid, ls, run in exps:
        tr, va = load_curves(eid, run)
        eid = f'{eid} {run}' if run else eid
        if tr:
            s, e = np.array(tr).T
            k = max(1, len(e) // 200)
            e = np.convolve(e, np.ones(k) / k, mode='valid')
            axes[0].plot(s[:len(e)], e, ls, label=eid)
        if va:
            s, c, f = np.array(va).T
            axes[1].plot(s, c, ls, color='C0', label=f'{eid} clean')
            axes[1].plot(s, f, ls, color='C3', label=f'{eid} final')
    for ax, t in zip(axes, ('train EPE (smoothed)', 'Sintel val EPE (periodic subset)')):
        ax.axvline(ctx.step, color='gray', lw=0.7)
        ax.set_title(t)
        ax.set_xlabel('step')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
    fig.suptitle(f'{ctx.tag} (solid) vs parent (dashed); grey line = this checkpoint')
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def stage_frontend(ctx):
    d = ctx.sdir('frontend')
    ctx.sh('frontend', ['tools/silencing_check.py', ctx.ref, '--out', osp.join(d, 'silencing_check.json')])
    import torch
    from evaluate import load_model
    dog = load_model(ctx.ref, device='cpu').fnet.dog
    res = {'silencing': read_json(osp.join(d, 'silencing_check.json')),
           'params': {'gain_mode': dog.gain_mode, 'temporal_mode': dog.temporal_mode,
                      'gain_range': [dog.g_lo, dog.g_hi] if dog.gain_mode == 'contrast' else None}}
    del dog
    if ctx.full:
        ctx.sh('frontend', ['tools/analyse_temporal.py', ctx.ref, '--out_dir', d, '--no_rdk'])
        res['temporal'] = read_json(osp.join(d, 'temporal.json'))
        plot_filter_trajectory(ctx, osp.join(d, 'filters_trajectory.png'))
    torch.cuda.empty_cache()
    return res


def plot_filter_trajectory(ctx, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fj = osp.join(run_dir(ctx.exp_id, ctx.run), 'filters', 'filters.jsonl')
    if not osp.exists(fj):
        return
    rows = [json.loads(line) for line in open(fj)]
    rows = [r for r in rows if r['step'] <= ctx.step] or rows
    steps = [r['step'] for r in rows]
    panels = [('dog', 'sigma_center', 'DoG centre σ'), ('dog', 'sigma_surround', 'DoG surround σ'),
              ('dog', 'gain', 'DoG gain'), ('dog', 'temporal_dc_gain', 'DoG temporal DC gain'),
              ('gabor', 'speed', 'Gabor speed'), ('gabor', 'sigma_t', 'Gabor temporal σ')]
    panels = [p for p in panels if p[1] in rows[-1].get(p[0], {})]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (layer, key, title) in zip(axes.flat, panels):
        a = np.array([r[layer][key] for r in rows], dtype=float)
        for i in range(a.shape[1]):
            ax.plot(steps, a[:, i], lw=1, label=str(i) if layer == 'dog' else None)
        ax.set_title(title)
        ax.set_xlabel('step')
        if layer == 'dog':
            ax.legend(fontsize=6, ncol=4)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis('off')
    fig.suptitle(f'{ctx.exp_id}: learned filter parameters over training (filters.jsonl), up to step {ctx.step}')
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def stage_patterns(ctx):
    d, rp = ctx.sdir('patterns'), ctx.cfg['random_patterns']
    ctx.sh('patterns', ['tools/eval_random_patterns.py', ctx.ref, '--cases', rp['cases'],
                        '--passes', rp['passes'] if ctx.full else rp['light_passes'],
                        '--max_scenes', str(rp['max_scenes'] if ctx.full else rp['light_max_scenes']),
                        '--iters', str(ctx.cfg['iters']), '--out', osp.join(d, 'random_patterns.json')])
    return read_json(osp.join(d, 'random_patterns.json'))


def stage_probes(ctx):
    d = ctx.sdir('probes')
    ctx.sh('probes', ['tools/probe_random_patterns.py', ctx.ref, '--seeds', str(ctx.cfg['probes']['seeds']),
                      '--out_json', osp.join(d, 'random_pattern_probe.json'), '--fig_dir', d])
    return {'files': sorted(f for f in os.listdir(d) if f.endswith('.png'))}


def stage_static_viz(ctx):
    d, sv = ctx.sdir('static_viz'), ctx.cfg['static_viz']
    for tool in ('show_layer_outputs', 'show_corr_inputs', 'show_context'):
        ctx.sh('static_viz', [f'tools/{tool}.py', '--model', ctx.ref, '--out', osp.join(d, tool),
                              '--scene', str(sv['scene']), '--seed', str(sv['seed'])])
    ctx_stats = read_json(osp.join(d, 'show_context', 'context_stats.json'), {})
    return {'context_ablation_epe': ctx_stats.get('ablation_epe')}


def stage_motion(ctx):
    d = ctx.sdir('motion')
    ctx.sh('motion', ['tools/render_motion.py', ctx.ref, '--out_dir', d] + ([] if ctx.full else ['--light']))
    return read_json(osp.join(d, 'summary.json'))


STAGE_FN = {'metrics': stage_metrics, 'frontend': stage_frontend, 'patterns': stage_patterns,
            'probes': stage_probes,
            'static_viz': stage_static_viz, 'motion': stage_motion}


def run_stage(ctx, stage, force):
    done = osp.join(ctx.sdir(stage), 'done.json')
    if not force and osp.exists(done):
        print(f'  {stage:10s} done already (--force to redo)', flush=True)
        return read_json(done)
    if force and osp.isdir(ctx.sdir(stage)):
        shutil.rmtree(ctx.sdir(stage))  # generated output only: drop stale files (e.g. scenes no longer rendered)
    print(f'  {stage:10s} running ...', flush=True)
    t0 = time.time()
    try:
        res = {'ok': True, 'result': STAGE_FN[stage](ctx)}
    except Exception as e:  # noqa: BLE001  record and carry on with the other stages
        res = {'ok': False, 'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()}
        print(f'  {stage:10s} FAILED: {res["error"]}', flush=True)
    res.update(seconds=round(time.time() - t0, 1), when=datetime.datetime.now().isoformat(timespec='seconds'),
               git=git_commit(), full=ctx.full)
    if res['ok']:
        write_json(done, res)
        print(f'  {stage:10s} ok, {res["seconds"]:.0f} s', flush=True)
    else:
        write_json(osp.join(ctx.sdir(stage), 'failed.json'), res)
    return res


# ----------------------------------------------------------------------------------------------- flags
def collect(ckpt_dir):
    """Gather the done.json results of every stage of one tests/<exp>/<step>/ folder."""
    out = {}
    for stage in STAGES:
        r = read_json(osp.join(ckpt_dir, stage, 'done.json'))
        if r and r.get('ok'):
            out[stage] = r['result']
    return out


def dog_types(sil):
    bi = sil.get('is_biphasic') if sil else None
    n = len(sil['gain']) if sil else 0
    return [('bi' if b else 'mono') for b in bi] if bi else ['free'] * n


def make_flags(s, parent, cfg):
    """Simple threshold rules. Each flag names the file its numbers come from."""
    th, flags = cfg['flags'], []
    fe = s.get('frontend', {})
    sil = fe.get('silencing')
    if sil:
        types = dog_types(sil)
        src = 'frontend/silencing_check.json'
        for i, a in enumerate(sil['attenuation_vs_mono']):
            if a > th['attenuation_vs_mono']:
                flags.append(f'DoG {i} ({types[i]}): raw Sintel response {a:,.0f}× below the median '
                             f'{"monophasic" if sil.get("is_biphasic") else ""} neuron ({src})')
        for i, (c, u) in enumerate(zip(sil['sigma_center'], sil['sigma_surround'])):
            if (u - c) / u < th['centre_surround_rel_gap']:
                flags.append(f'DoG {i} ({types[i]}): centre σ {c:.3f} ≈ surround σ {u:.3f}, so the spatial kernel '
                             f'nearly cancels (spatial L1 {sil["spatial_l1"][i]:.4f}) ({src})')
        gr = fe.get('params', {}).get('gain_range')
        if gr:
            for i, g in enumerate(sil['gain']):
                for b, name in ((gr[1], 'upper'), (gr[0], 'lower')):
                    if abs(g - b) / b < th['gain_at_bound_rel']:
                        flags.append(f'DoG {i} ({types[i]}): gain {g:.3f} pinned at the {name} bound {b:g} ({src})')
    tmp = fe.get('temporal')
    if tmp:
        fr = tmp.get('sintel_freeze', {})
        ch = fr.get('feature_rel_change_mean')
        if ch is not None and ch < th['freeze_feature_change']:
            flags.append(f'freeze test: features change only {ch:.1%} when every frame of the window is replaced '
                         f'by its centre frame (EPE normal {fr["normal"]["all"]} vs frozen {fr["freeze"]["all"]}) '
                         f'(frontend/temporal.json)')
        share = tmp.get('mix', {}).get('dog_neuron_share_of_gabor_input', [])
        for i, v in enumerate(share):
            if v < th['gabor_input_share']:
                flags.append(f'Gabor layer takes only {v:.1%} of its input weight from DoG {i} '
                             f'(frontend/temporal.json, mix)')
    mo = s.get('motion')
    if mo:
        for sc in mo['scenes']:
            e, p = sc['epe'], sc['epe_pair_only']
            if abs(p - e) / max(e, 1e-6) < th['pair_only_epe_rel']:
                flags.append(f'{sc["name"]}: pair-only input gives the same EPE as the full clip ({p:.3f} vs '
                             f'{e:.3f}): the extra clip frames are not used here (motion/summary.json)')
        ratio, silent = mo.get('dog_object_over_background'), mo.get('dog_silent')
        if ratio and silent is not None:
            live = [i for i in range(len(ratio)) if not silent[i]]
            if live and max(ratio[i] for i in live) < th['dog_motion_ratio']:
                flags.append('no active DoG neuron responds more inside moving objects than on the background '
                             f'(max ratio {max(ratio[i] for i in live):.2f}) (motion/summary.json)')
    for case, region, what in (('hollow', 'hole', 'inside the ring (hole)'), ('roi', 'object', 'inside the ROI')):
        tx, sh = pat(s, case, 'texture', region), pat(s, case, 'shape', region)
        if tx is None or sh is None:
            continue
        r = th['pattern_shape_vs_texture']
        if sh < r * tx:
            flags.append(f'{case}: {what} the predicted flow follows the SHAPE motion (EPE vs shape {sh:.2f}, '
                         f'vs texture {tx:.2f}) (patterns/random_patterns.json, clean)')
        elif tx < r * sh:
            flags.append(f'{case}: {what} the predicted flow follows the DOTS, not the shape motion (EPE vs texture '
                         f'{tx:.2f}, vs shape {sh:.2f}) (patterns/random_patterns.json, clean)')
    if parent:
        for key, get in (('Sintel val clean', lambda x: x.get('metrics', {}).get('sintel', {}).get('clean')),
                         ('Sintel val final', lambda x: x.get('metrics', {}).get('sintel', {}).get('final')),
                         ('random pattern occluded (clean)', lambda x: pat(x, 'occluded')),
                         ('random pattern hollow (clean)', lambda x: pat(x, 'hollow')),
                         ('random pattern roi (clean)', lambda x: pat(x, 'roi')),
                         ('random pattern visible (clean)', lambda x: pat(x, 'visible'))):
            a, b = get(s), get(parent['stages'])
            if a is not None and b is not None and abs(a - b) > th['epe_delta_vs_parent']:
                flags.append(f'{key} EPE {a:.3f} vs parent {b:.3f} (Δ {a - b:+.3f})')
    return flags


def pat(s, case, gt='texture', region='all', dstype='clean'):
    """EPE on a random-pattern case (patterns/random_patterns.json)."""
    return ((((s.get('patterns') or {}).get(case) or {}).get(dstype) or {}).get(gt) or {}).get(region)


# ----------------------------------------------------------------------------------------------- reports
def parent_summary(exp_id, step=None, full=True):
    """The parent's summary to compare with: its final checkpoint for a full run, else the parent checkpoint in
    tests/ whose step is closest to `step` (light steps of different experiments rarely coincide)."""
    parent = exp_config(exp_id).get('parent')
    if not parent:
        return None, None
    try:
        if full or step is None:
            return parent, read_json(osp.join(paths.tests_dir(parent[:4]), 'summary.json'))
        pfolder = osp.join(paths.TESTS, osp.basename(paths.exp_dir(parent[:4])))
    except FileNotFoundError:
        return parent, None
    cands = [read_json(osp.join(pfolder, d, 'summary.json')) for d in os.listdir(pfolder)] \
        if osp.isdir(pfolder) else []
    cands = [c for c in cands if c]
    return parent, (min(cands, key=lambda c: abs(c['step'] - step)) if cands else None)


def fmt(v, nd=3):
    if v is None:
        return '–'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


def key_numbers(st):
    m = st.get('metrics', {}).get('sintel', {})
    t = st.get('frontend', {}).get('temporal') or {}
    fr = t.get('sintel_freeze', {})
    rows = [('Sintel val clean EPE', m.get('clean')), ('Sintel val final EPE', m.get('final')),
            ('Sintel val clean noc / occ', f"{fmt(m.get('clean_noc'))} / {fmt(m.get('clean_occ'))}" if m else None),
            *[(f'random pattern {c}: EPE clean / final', f"{fmt(pat(st, c))} / {fmt(pat(st, c, dstype='final'))}"
               if st.get('patterns') else None) for c in PATTERN_CASES],
            ('random pattern occluded: object interior', pat(st, 'occluded', region='object')),
            ('random pattern hollow: ring / hole vs texture / hole vs shape',
             f"{fmt(pat(st, 'hollow', region='object'))} / {fmt(pat(st, 'hollow', region='hole'))} / "
             f"{fmt(pat(st, 'hollow', 'shape', 'hole'))}" if st.get('patterns') else None),
            ('random pattern roi: inside ROI vs texture / vs shape',
             f"{fmt(pat(st, 'roi', region='object'))} / {fmt(pat(st, 'roi', 'shape', 'object'))}"
             if st.get('patterns') else None),
            ('freeze test EPE normal / frozen', f"{fmt(fr.get('normal', {}).get('all'))} / "
             f"{fmt(fr.get('freeze', {}).get('all'))}" if fr else None),
            ('freeze test feature change', fr.get('feature_rel_change_mean'))]
    for sc in (st.get('motion') or {}).get('scenes', []):
        rows.append((f"motion {sc['name']} EPE (pair-only)", f"{fmt(sc['epe'])} ({fmt(sc['epe_pair_only'])})"))
    return rows


def neuron_table(st):
    fe = st.get('frontend', {})
    sil = fe.get('silencing')
    if not sil:
        return []
    mix = ((fe.get('temporal') or {}).get('mix') or {}).get('dog_neuron_share_of_gabor_input')
    mo = st.get('motion') or {}
    ratio, silent = mo.get('dog_object_over_background'), mo.get('dog_silent')
    types = dog_types(sil)
    lines = ['| DoG | type | σ centre / surround | gain | raw RMS Sintel | attenuation | drive std Sintel / dots '
             '| share of Gabor input | motion obj/bg |', '|' + '---|' * 9]
    for i in range(len(sil['gain'])):
        r = '–' if ratio is None else ('silent' if silent and silent[i] else f'{ratio[i]:.2f}')
        lines.append(f"| {i} | {types[i]} | {sil['sigma_center'][i]:.3f} / {sil['sigma_surround'][i]:.3f} | "
                     f"{sil['gain'][i]:.3f} | {sil['raw_rms_sintel'][i]:.2g} | {sil['attenuation_vs_mono'][i]:,.1f}× | "
                     f"{sil['drive_std_sintel'][i]:.3f} / {sil['drive_std_dots'][i]:.3f} | "
                     f"{'–' if mix is None else f'{mix[i]:.1%}'} | {r} |")
    return lines


def run_prefix(step_dir):
    """'scratch-p03_final' -> 'scratch-p03'; '40000' or 'final' (main run) -> ''."""
    return step_dir.rsplit('_', 1)[0] if '_' in step_dir else ''


def trajectory(exp_folder, step_dir=''):
    """Checkpoints of the same run of this experiment that are in tests/."""
    rows = []
    for d in os.listdir(osp.join(paths.TESTS, exp_folder)):
        if run_prefix(d) != run_prefix(step_dir):
            continue
        s = read_json(osp.join(paths.TESTS, exp_folder, d, 'summary.json'))
        if s:
            rows.append(s)
    rows.sort(key=lambda s: s['step'])
    out = ['| step | checkpoint | Sintel clean / final | DoG att. > 10× | motion EPE (1st scene) | flags |',
           '|---|---|---|---|---|---|']
    for s in rows:
        m = s['stages'].get('metrics', {}).get('sintel', {})
        sil = s['stages'].get('frontend', {}).get('silencing')
        n_att = sum(a > 10 for a in sil['attenuation_vs_mono']) if sil else None
        mo = (s['stages'].get('motion') or {}).get('scenes') or [{}]
        out.append(f"| {s['step']} | [{s['dir']}](../{s['dir']}/report.md) | {fmt(m.get('clean'))} / "
                   f"{fmt(m.get('final'))} | {fmt(n_att)} | {fmt(mo[0].get('epe'))} | {len(s['flags'])} |")
    return out


def figure_links(ckpt_dir):
    links = []
    for stage in STAGES:
        sd = osp.join(ckpt_dir, stage)
        if not osp.isdir(sd):
            continue
        for root, _, files in os.walk(sd):
            if osp.basename(root) in ('flow', 'layers'):
                continue
            for f in sorted(files):
                if f.endswith('.png'):
                    rel = osp.relpath(osp.join(root, f), ckpt_dir).replace('\\', '/')
                    links.append(f'- [{rel}]({rel})')
    return links


def write_report(ckpt_dir, cfg):
    exp_folder, step_dir = osp.basename(osp.dirname(ckpt_dir)), osp.basename(ckpt_dir)
    st = collect(ckpt_dir)
    if not st:
        return None
    ref = f'{exp_folder[:4]}@{step_dir}'
    failed = [s for s in STAGES if osp.exists(osp.join(ckpt_dir, s, 'failed.json'))
              and not osp.exists(osp.join(ckpt_dir, s, 'done.json'))]
    step = st.get('metrics', {}).get('step')
    if step is None:
        step = ckpt_step(ref)
    full = bool(read_json(osp.join(ckpt_dir, 'metrics', 'done.json'), {}).get('full'))
    parent_id, parent = parent_summary(exp_folder[:4], step, full)
    summary = {'checkpoint': f'{exp_folder}@{step_dir}', 'experiment': exp_folder, 'dir': step_dir, 'step': step,
               'full': full, 'parent': parent_id, 'parent_checkpoint': parent and parent['checkpoint'],
               'generated': datetime.datetime.now().isoformat(timespec='seconds'), 'git': git_commit(),
               'stages': st, 'failed_stages': failed}
    summary['flags'] = make_flags(st, parent, cfg)
    write_json(osp.join(ckpt_dir, 'summary.json'), summary)

    pst = parent['stages'] if parent else {}
    pk = dict(key_numbers(pst)) if parent else {}
    L = [f'# {summary["checkpoint"]}: pipeline report', '',
         f'> Generated by `tools/pipeline.py` on {summary["generated"][:10]} (code {summary["git"]}). '
         'Numbers and flags only, no conclusions: KB items are drafted from this together with the user.', '',
         f'- Run: {"full" if full else "light"} · step {step} · parent: '
         + (f'{parent["checkpoint"]} ([report](../../{parent["experiment"]}/{parent["dir"]}/report.md))' if parent
            else (f'{parent_id} (not yet through the pipeline: `python tools/pipeline.py {parent_id[:4]}`)'
                  if parent_id else 'none')),
         '- Motion, in sync with the parent: open [index.html](index.html) in a browser']
    if failed:
        L.append(f'- **Failed stages:** {", ".join(failed)} (see `<stage>/failed.json`)')
    L += ['', '## Key numbers', '', f'| metric | this (step {step}) | parent'
          + (f' {parent["checkpoint"]} (step {parent["step"]})' if parent else '') + ' |', '|---|---|---|']
    for k, v in key_numbers(st):
        L.append(f'| {k} | {fmt(v)} | {fmt(pk.get(k))} |')
    src = st.get('metrics', {}).get('sintel_source')
    L += ['', f'Sintel source: {src or "–"}. Random patterns: patterns/random_patterns.json (EPE vs texture '
          'flow unless "vs shape"; clean pass). Freeze test: frontend/temporal.json. Motion: motion/summary.json.', '', '## Unusual (auto-flagged by threshold rules; check before believing)', '']
    L += [f'- {f}' for f in summary['flags']] or ['- nothing flagged']
    nt = neuron_table(st)
    if nt:
        L += ['', '## DoG neurons', '', 'Sources: frontend/silencing_check.json, frontend/temporal.json (mix), '
              'motion/summary.json (obj/bg = mean response inside moving objects / on background).', ''] + nt
    ab = (st.get('static_viz') or {}).get('context_ablation_epe')
    if ab:
        L += ['', '## Context ablation (static_viz/show_context/context_stats.json)', '',
              '| condition | all | object interior | boundary | background |', '|---|---|---|---|---|']
        for c, r in ab.items():
            L.append(f"| {c} | " + ' | '.join(fmt(r.get(k)) for k in ('all', 'object interior', 'boundary',
                                                                       'background')) + ' |')
    L += ['', '## Training trajectory (checkpoints of this experiment in tests/)', ''] + trajectory(exp_folder, step_dir)
    L += ['', '## Figures', ''] + (figure_links(ckpt_dir) or ['- none'])
    with open(osp.join(ckpt_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    write_index_html(ckpt_dir, summary, parent)
    return summary


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e2dc;--acc:#2a6fdb}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece8;--mut:#a3a39c;--card:#20201e;--line:#34342f;--acc:#7aa7ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1800px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:16px;margin:24px 0 8px} .mut{color:var(--mut)}
.bar{position:sticky;top:0;z-index:2;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);
 display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button,select{font:inherit;padding:4px 10px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
input[type=range]{flex:1;min-width:160px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,640px),1fr));gap:12px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
figure img{width:100%;height:auto;display:block;image-rendering:pixelated}
figcaption{font-size:12px;color:var(--mut);margin-bottom:6px}
table{border-collapse:collapse;font-size:13px;margin:6px 0} td,th{border:1px solid var(--line);padding:3px 8px;text-align:left}
ul.flags li{margin:2px 0} a{color:var(--acc)}
.missing{padding:40px;text-align:center;color:var(--mut)}
</style></head><body><main>
<h1>__TITLE__</h1>
<div class="mut">__SUB__ · numbers in <a href="report.md">report.md</a> · generated by tools/pipeline.py; no conclusions</div>
<div class="bar">
 <button id="play">❚❚ pause</button><button id="prev">◀</button><button id="next">▶</button>
 <input id="seek" type="range" min="0" value="0"><span id="pos" class="mut"></span>
 <label class="mut">panel <select id="panel"><option value="flow">flow</option><option value="layers">layers</option></select></label>
 <label class="mut">fps <select id="fps"><option>2</option><option>4</option><option selected>__FPS__</option><option>12</option></select></label>
</div>
<div id="scenes"></div>
<h2>Key numbers</h2>__TABLE__
<h2>Unusual (auto-flagged)</h2><ul class="flags">__FLAGS__</ul>
</main>
<script>
const SCENES = __SCENES__;
let k = 0, playing = true, timer = null;
const maxN = Math.max(1, ...SCENES.flatMap(s => s.cols.map(c => c.n)));
const seek = document.getElementById('seek'); seek.max = maxN - 1;
const host = document.getElementById('scenes');
for (const s of SCENES) {
  const h = document.createElement('h2'); h.textContent = s.name; host.appendChild(h);
  const g = document.createElement('div'); g.className = 'grid'; host.appendChild(g);
  for (const c of s.cols) {
    const f = document.createElement('figure');
    f.innerHTML = `<figcaption>${c.label}</figcaption>` + (c.n ? '<img loading="eager">' : '<div class="missing">not rendered yet</div>');
    c.img = f.querySelector('img'); g.appendChild(f);
  }
}
const pad = i => String(i).padStart(4, '0');
function show() {
  const panel = document.getElementById('panel').value;
  for (const s of SCENES) for (const c of s.cols) if (c.img) c.img.src = `${c.base}/${panel}/${pad(Math.min(k, c.n - 1))}.jpg`;
  seek.value = k; document.getElementById('pos').textContent = `pair ${k + 1} / ${maxN}`;
}
function tick() { k = (k + 1) % maxN; show(); }
function restart() { clearInterval(timer); if (playing) timer = setInterval(tick, 1000 / +document.getElementById('fps').value); }
document.getElementById('play').onclick = e => { playing = !playing; e.target.textContent = playing ? '❚❚ pause' : '► play'; restart(); };
document.getElementById('prev').onclick = () => { k = (k - 1 + maxN) % maxN; show(); };
document.getElementById('next').onclick = () => { k = (k + 1) % maxN; show(); };
seek.oninput = () => { k = +seek.value; show(); };
document.getElementById('panel').onchange = show; document.getElementById('fps').onchange = restart;
document.addEventListener('keydown', e => { if (e.key === ' ') { e.preventDefault(); document.getElementById('play').click(); }
  if (e.key === 'ArrowLeft') document.getElementById('prev').click(); if (e.key === 'ArrowRight') document.getElementById('next').click(); });
show(); restart();
</script></body></html>
'''


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def write_index_html(ckpt_dir, summary, parent):
    mo = summary['stages'].get('motion') or {}
    pmo = (parent or {}).get('stages', {}).get('motion') or {}
    pscenes = {s['name']: s for s in pmo.get('scenes', [])}
    scenes = []
    for s in mo.get('scenes', []):
        cols = [{'label': f'{summary["checkpoint"]} (this)', 'base': f'motion/{s["name"]}', 'n': s['n_frames']}]
        if parent:
            ps = pscenes.get(s['name'])
            cols.append({'label': f'{parent["checkpoint"]} (parent)',
                         'base': f'../../{parent["experiment"]}/{parent["dir"]}/motion/{s["name"]}',
                         'n': ps['n_frames'] if ps else 0})
        scenes.append({'name': f'{s["name"]}: EPE {s["epe"]:.3f} (pair-only {s["epe_pair_only"]:.3f})', 'cols': cols})
    pk = dict(key_numbers(parent['stages'])) if parent else {}
    rows = ''.join(f'<tr><td>{esc(k)}</td><td>{esc(fmt(v))}</td><td>{esc(fmt(pk.get(k)))}</td></tr>'
                   for k, v in key_numbers(summary['stages']))
    table = f'<table><tr><th>metric</th><th>this</th><th>parent</th></tr>{rows}</table>'
    flags = ''.join(f'<li>{esc(f)}</li>' for f in summary['flags']) or '<li>nothing flagged</li>'
    sub = f'step {summary["step"]} · parent {parent["checkpoint"] if parent else summary["parent"] or "none"}'
    html = (HTML.replace('__TITLE__', esc(summary['checkpoint'])).replace('__SUB__', esc(sub))
            .replace('__FPS__', str(mo.get('fps', 6))).replace('__TABLE__', table).replace('__FLAGS__', flags)
            .replace('__SCENES__', json.dumps(scenes)))
    with open(osp.join(ckpt_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)


def write_readme():
    rows = []
    for exp_folder in sorted(os.listdir(paths.TESTS)) if osp.isdir(paths.TESTS) else []:
        ed = osp.join(paths.TESTS, exp_folder)
        if not osp.isdir(ed):
            continue
        for d in os.listdir(ed):
            s = read_json(osp.join(ed, d, 'summary.json'))
            if s:
                rows.append(s)
    rows.sort(key=lambda s: (s['experiment'], s['step']))
    L = ['# tests/: pipeline results', '',
         '> Generated by `tools/pipeline.py`. **Not reviewed**: numbers and auto-flags only, no claims. '
         'Findings are drafted from these reports together with the user and written to `kb/` only after approval.',
         '', 'Run: `python tools/pipeline.py E00N` after training (see README.md). Stimuli and thresholds: '
         '`configs/pipeline.json`.', '',
         '| experiment | checkpoint | run | Sintel clean / final | occluded / hollow / roi / visible (clean) '
         '| flags | report | motion |', '|---|---|---|---|---|---|---|---|']
    for s in rows:
        m = s['stages'].get('metrics', {}).get('sintel', {})
        st = s['stages']
        rel = f'{s["experiment"]}/{s["dir"]}'
        L.append(f'| {s["experiment"]} | {s["dir"]} (step {s["step"]}) | {"full" if s["full"] else "light"} | '
                 f'{fmt(m.get("clean"))} / {fmt(m.get("final"))} | '
                 + ' / '.join(fmt(pat(st, c)) for c in PATTERN_CASES)
                 + f' | {len(s["flags"])} | [report.md]({rel}/report.md) | '
                 f'[index.html]({rel}/index.html) |')
    os.makedirs(paths.TESTS, exist_ok=True)
    with open(osp.join(paths.TESTS, 'README.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


def all_ckpt_dirs():
    out = []
    for e in sorted(os.listdir(paths.TESTS)) if osp.isdir(paths.TESTS) else []:
        ed = osp.join(paths.TESTS, e)
        if osp.isdir(ed) and e[:1] == 'E':
            out += [osp.join(ed, d) for d in os.listdir(ed) if osp.isdir(osp.join(ed, d))]
    return out


def build_reports(dirs, cfg):
    # parents first, so that a child's report can compare against its parent's fresh summary
    dirs = sorted(dirs, key=lambda d: osp.basename(osp.dirname(d)))
    for d in dirs:
        s = write_report(d, cfg)
        if s:
            print(f'report: {osp.relpath(d, ROOT)}/report.md ({len(s["flags"])} flags)', flush=True)
    write_readme()


# ----------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('exp', nargs='?', help='experiment id, e.g. E004')
    p.add_argument('--run', default=None, help='a non-main run of the experiment (checkpoints in runs/<run>/checkpoints/); '
                                              'results go to tests/<experiment>/<run>_<step>/')
    p.add_argument('--steps', default=None, help="comma list of steps and/or 'final' (default: final + light steps)")
    p.add_argument('--light_steps', type=int, default=None, help='number of evenly spaced light steps (config default)')
    p.add_argument('--only', default=None, help=f'comma list of stages: {",".join(STAGES)},report')
    p.add_argument('--skip', default='', help='comma list of stages to skip')
    p.add_argument('--force', action='store_true', help='redo stages that are already done')
    p.add_argument('--report_only', action='store_true', help='only rebuild reports for everything in tests/')
    p.add_argument('--config', default=CONFIG)
    a = p.parse_args()
    cfg = read_json(a.config)
    if a.report_only:
        build_reports(all_ckpt_dirs(), cfg)
        return
    if not a.exp:
        p.error('give an experiment id, or --report_only')
    exp_id = osp.basename(paths.exp_dir(a.exp))[:4]
    if a.run:
        steps, final = run_checkpoints(exp_id, a.run)
        ckd = osp.join(run_dir(exp_id, a.run), 'checkpoints')
        if not steps and not final:
            p.error(f'no checkpoints in {ckd}')
        final = final or osp.join(ckd, f'{steps[-1]}.pth')
        if a.steps:
            refs = [(final if s == 'final' else osp.join(ckd, f'{s}.pth'), s == 'final') for s in a.steps.split(',')]
        else:
            n = cfg['light_steps'] if a.light_steps is None else a.light_steps
            refs = [(final, True)] + [(osp.join(ckd, f'{s}.pth'), False) for s in pick_light_steps(exp_id, n, a.run)]
    elif a.steps:
        # 'final' = final.pth, or the highest step when there is none (E001, E002); only that one runs full
        refs = [(exp_id if s == 'final' else f'{exp_id}@{s}', s == 'final') for s in a.steps.split(',')]
    else:
        n = cfg['light_steps'] if a.light_steps is None else a.light_steps
        refs = [(exp_id, True)] + [(f'{exp_id}@{s}', False) for s in pick_light_steps(exp_id, n)]
    only = set(a.only.split(',')) if a.only else set(STAGES) | {'report'}
    skip = set(filter(None, a.skip.split(',')))
    stages = [s for s in STAGES if s in only and s not in skip]

    t0, dirs, failed = time.time(), [], []
    for ref, full in refs:
        ctx = Ctx(ref, full, cfg)
        print(f'== {ctx.tag} (step {ctx.step}, {"full" if full else "light"}) -> {osp.relpath(ctx.out, ROOT)}',
              flush=True)
        for stage in stages:
            if not full and stage not in LIGHT_STAGES:
                continue
            r = run_stage(ctx, stage, a.force)
            if not r.get('ok'):
                failed.append(f'{ctx.tag}:{stage}')
        dirs.append(ctx.out)
    if 'report' in only and 'report' not in skip:
        build_reports(dirs, cfg)
    print(f'done in {(time.time() - t0) / 60:.1f} min' + (f'; FAILED: {", ".join(failed)}' if failed else ''),
          flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
