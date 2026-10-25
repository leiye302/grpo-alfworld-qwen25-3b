"""CPU-only release and experiment contracts."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.deps/sdar'
WORLD_SIZE = 4


def load(name):
    return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def devices(spec):
    ids = [x.strip() for x in spec.split(',')]
    if len(ids) != WORLD_SIZE or len(set(ids)) != WORLD_SIZE or any(not x.isdigit() for x in ids):
        raise ValueError('Choose four distinct allocated GPU indices, e.g. 0,1,2,3')
    return ids


def tasks(assets=None):
    rows = load('seen140_tasks.json')
    if len(rows) != 140 or len({x['task_id'] for x in rows}) != 140:
        raise ValueError('Expected 140 unique validation tasks')
    for i, row in enumerate(rows):
        if row['seed'] != 2026 + i or not row['task_id'].startswith('json_2.1.1/valid_seen/'):
            raise ValueError('Validation task/seed contract differs')
        if assets is not None:
            assets = Path(assets).resolve()
            game = (assets / row['task_id']).resolve()
            if not game.is_relative_to(assets / 'json_2.1.1/valid_seen') or sha(game) != row['game_sha256']:
                raise ValueError('ALFWorld game version mismatch: ' + row['task_id'])
    return rows


def checkpoint_complete(checkpoint, world_size=WORLD_SIZE):
    checkpoint = Path(checkpoint)
    expected = [checkpoint / 'data.pt', checkpoint / 'actor/config.json']
    expected += [checkpoint / 'actor' / f'{kind}_world_size_{world_size}_rank_{rank}.pt'
                 for rank in range(world_size) for kind in ('model', 'optim', 'extra_state')]
    missing = [str(p) for p in expected if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError('Incomplete resumable checkpoint: ' + ', '.join(missing))
    return checkpoint


def protocol(text):
    import re
    reasoning = re.search(r'<think>(.*?)</think>', text, re.S)
    strict = re.fullmatch(r'\s*<think>(.*?)</think>\s*<action>(.*?)</action>\s*',
                          text.replace('<|im_end|>', '').replace('<|endoftext|>', ''), re.S)
    return {'reasoning_open': '<think>' in text,
            'reasoning_nonempty_closed': bool(reasoning and reasoning[1].strip()),
            'strict_protocol': bool(strict and strict[1].strip() and strict[2].strip()
                                    and text.count('<think>') == text.count('</think>') == 1
                                    and text.count('<action>') == text.count('</action>') == 1)}
