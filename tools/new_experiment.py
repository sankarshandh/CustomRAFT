"""Create a new experiment folder from a parent experiment's config.

    python tools/new_experiment.py E004-magno --parent E003 --set num_steps=50000 --set dog_gain_mode=contrast
    python tools/new_experiment.py E004-magno --parent E003 --dry_run      # print the config, write nothing

Creates experiments/E004-magno/ with
    config.json   {"id", "parent", "created", "changes_vs_parent": {key: [parent value, new value]}, "args": {...}}
    card.md       an EMPTY copy of kb/templates/card.md (only id / parent / command filled in)
    analysis/     empty
Without --parent the defaults come from configs/base.json.
Values are converted with train.py's own argparse types (lists: comma-separated, booleans: true/false).
Next steps (printed at the end): commit, train, tag.
"""
import argparse
import datetime
import json
import os
import os.path as osp
import re
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.utils.paths import EXPERIMENTS, exp_dir  # noqa: E402
from train import build_parser  # noqa: E402

NOT_CONFIG = {'exp', 'name'}  # per-launch, not part of an experiment's definition


def convert(parser, key, text):
    action = {a.dest: a for a in parser._actions}.get(key)
    if action is None or key in NOT_CONFIG:
        raise SystemExit(f'unknown train.py argument: {key}')
    if action.const is True or action.const is False:  # store_true / store_false
        if text.lower() not in ('true', 'false'):
            raise SystemExit(f'{key} is a flag: use {key}=true or {key}=false')
        return text.lower() == 'true'
    cast = action.type or str
    if text.lower() == 'none':
        return None
    if action.nargs in ('+', '*'):
        return [cast(v) for v in text.split(',')]
    value = cast(text)
    if action.choices is not None and value not in action.choices:
        raise SystemExit(f'{key}={value!r} not in {action.choices}')
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('name', help='E00N-short-name, e.g. E004-magno')
    p.add_argument('--parent', default=None, help="parent experiment id, e.g. E003 (default: configs/base.json)")
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='override a train.py argument')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()

    if not re.fullmatch(r'E\d{3}-[a-z0-9][a-z0-9-]*', a.name):
        raise SystemExit('name must look like E004-short-name (lower case, digits, hyphens)')
    existing = [d for d in os.listdir(EXPERIMENTS) if d[:4] == a.name[:4]] if osp.isdir(EXPERIMENTS) else []
    if existing:
        raise SystemExit(f'{a.name[:4]} already exists: {existing}')

    if a.parent is None:
        parent_id, base_file = None, osp.join(ROOT, 'configs', 'base.json')
        base = json.load(open(base_file))
    else:
        pdir = exp_dir(a.parent)
        parent_id = osp.basename(pdir)
        base = json.load(open(osp.join(pdir, 'config.json')))['args']

    parser = build_parser()
    args, changes = dict(base), {}
    for item in a.set:
        if '=' not in item:
            raise SystemExit(f'--set needs KEY=VALUE, got {item!r}')
        k, v = item.split('=', 1)
        args[k] = convert(parser, k.strip(), v.strip())
        if args[k] != base.get(k):
            changes[k] = [base.get(k), args[k]]

    cfg = {'id': a.name, 'parent': parent_id, 'created': datetime.date.today().isoformat(),
           'changes_vs_parent': changes, 'args': args}
    print(json.dumps({k: v for k, v in cfg.items() if k != 'args'}, indent=1))
    if a.dry_run:
        return

    out = osp.join(EXPERIMENTS, a.name)
    os.makedirs(osp.join(out, 'analysis'))
    with open(osp.join(out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    card = open(osp.join(ROOT, 'kb', 'templates', 'card.md'), encoding='utf-8').read()
    card = card.replace('E00N-name', a.name).replace('E00M-name', parent_id or 'none').replace('E00N-', a.name[:4] + '-')
    with open(osp.join(out, 'card.md'), 'w', encoding='utf-8') as f:
        f.write(card)
    rel = f'experiments/{a.name}'
    print(f'\ncreated {rel}/  (config.json, empty card.md, analysis/)\nnext:\n'
          f'  1. fill in card.md: Question, Change vs parent, Expected  (before training)\n'
          f'  2. git add -A && git commit -m "{a.name}: <change>"\n'
          f'  3. python -u train.py --exp {rel}      (log: {rel}/runs/{a.name}/train.log)\n'
          f'  4. git tag {a.name}')


if __name__ == '__main__':
    main()
