"""Experiment folders and checkpoint references.

An experiment lives in experiments/<id>-<name>/ (e.g. experiments/E003-typed/), with checkpoints in
experiments/<id>-<name>/checkpoints/<step>.pth and the last weights of the run in final.pth.

Checkpoint references accepted everywhere a model is loaded (evaluate.load_model, tools/*):
    E003            final checkpoint of E003 (final.pth, else the highest step)
    E003@100000     step 100000 of E003
    E003-typed@5000 the full folder name also works
    path/to/x.pth   any file path (unchanged)
"""
import glob
import os.path as osp
import re

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
EXPERIMENTS = osp.join(ROOT, 'experiments')
_REF = re.compile(r'^(E\d{3})(-[\w.-]+)?(?:@(\d+|final))?$')


def exp_dir(exp_id):
    """'E003' or 'E003-typed' -> absolute experiment folder (must exist exactly once)."""
    m = re.match(r'^(E\d{3})', exp_id)
    if m is None:
        raise ValueError(f'not an experiment id: {exp_id!r} (expected E000-style)')
    hits = sorted(glob.glob(osp.join(EXPERIMENTS, m.group(1) + '-*')))
    if len(hits) != 1:
        raise FileNotFoundError(f'expected one folder experiments/{m.group(1)}-*, found {hits}')
    return hits[0]


def checkpoint_steps(exp_id):
    """Sorted list of saved steps for an experiment."""
    files = glob.glob(osp.join(exp_dir(exp_id), 'checkpoints', '*.pth'))
    return sorted(int(osp.basename(f)[:-4]) for f in files if osp.basename(f)[:-4].isdigit())


def latest_checkpoint():
    """Reference to the final checkpoint of the highest-numbered experiment that has checkpoints."""
    for d in sorted(glob.glob(osp.join(EXPERIMENTS, 'E[0-9][0-9][0-9]-*')), reverse=True):
        if glob.glob(osp.join(d, 'checkpoints', '*.pth')):
            return osp.basename(d)[:4]
    raise FileNotFoundError('no experiment has checkpoints; pass --model')


def run_of(ref):
    """Name of the run whose own checkpoint folder holds this checkpoint (experiments/<exp>/runs/<run>/checkpoints/,
    used by runs other than the main run), else None."""
    ckdir = osp.dirname(osp.abspath(resolve_ckpt(ref)))
    runs = osp.dirname(osp.dirname(ckdir))
    if (osp.basename(ckdir) == 'checkpoints' and osp.basename(runs) == 'runs'
            and osp.normcase(osp.dirname(osp.dirname(runs))) == osp.normcase(EXPERIMENTS)):
        return osp.basename(osp.dirname(ckdir))
    return None


def describe(ref):
    """Checkpoint reference or path -> (experiment folder or None, short tag such as 'E003-typed@100000').
    A checkpoint of a non-main run gets the run in its tag: 'E004-asymviews@scratch-p03_final'."""
    path = osp.abspath(resolve_ckpt(ref))
    ckdir = osp.dirname(path)
    name = osp.splitext(osp.basename(path))[0]
    if osp.basename(ckdir) == 'checkpoints' and osp.dirname(osp.dirname(ckdir)) == EXPERIMENTS:
        exp = osp.dirname(ckdir)
        return exp, f'{osp.basename(exp)}@{name}'
    run = run_of(path)
    if run is not None:
        exp = osp.dirname(osp.dirname(osp.dirname(ckdir)))
        return exp, f'{osp.basename(exp)}@{run}_{name}'
    return None, name


def analysis_dir(ref, tool):
    """Default output folder for a tool run on one checkpoint: <experiment>/analysis/<tool>/<step>/."""
    exp, tag = describe(ref)
    if exp is None:
        return osp.join(ROOT, 'scratch', tool, tag)
    return osp.join(exp, 'analysis', tool, tag.split('@')[1])


TESTS = osp.join(ROOT, 'tests')


def tests_dir(ref, stage=None):
    """Pipeline output folder for one checkpoint (tools/pipeline.py): tests/<experiment>/<step>[/<stage>]/.
    <step> is 'final' for final.pth, else the step number; a bare .pth path goes to tests/_other/<name>/."""
    exp, tag = describe(ref)
    d = osp.join(TESTS, osp.basename(exp), tag.split('@')[1]) if exp else osp.join(TESTS, '_other', tag)
    return osp.join(d, stage) if stage else d


def resolve_ckpt(ref):
    """Checkpoint reference -> file path (see module docstring)."""
    m = _REF.match(ref)
    if m is None or osp.exists(ref):
        return ref
    ckdir = osp.join(exp_dir(m.group(1)), 'checkpoints')
    step = m.group(3)
    if step is None or step == 'final':
        final = osp.join(ckdir, 'final.pth')
        if osp.exists(final):
            return final
        steps = checkpoint_steps(m.group(1))
        if not steps:
            raise FileNotFoundError(f'no checkpoints in {ckdir}')
        step = steps[-1]
    path = osp.join(ckdir, f'{step}.pth')
    if not osp.exists(path):
        raise FileNotFoundError(f'{path} (available steps: {checkpoint_steps(m.group(1))})')
    return path
