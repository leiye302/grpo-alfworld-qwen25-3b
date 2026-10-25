"""Audit a clean checkout's allowlist, privacy, symlinks and source locks."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    '.env.example', '.gitattributes', '.github/workflows/check.yml', '.gitignore',
    'LICENSE', 'README.md', 'THIRD_PARTY.md', 'docs/REPRODUCTION.md',
    'configs/experiment.json', 'configs/models.json', 'configs/requirements.txt',
    'configs/seen140_tasks.json', 'configs/upstream.json', 'patches/runtime.patch',
    'scripts/check_release.py', 'scripts/eval.sh', 'scripts/run_both.sh', 'scripts/train.sh',
    'src/common.py', 'src/checkpoint_io.py', 'src/fixed_validation.py', 'src/report.py', 'src/run.py',
    'src/grpo_eval.py', 'src/setup.py', 'tests/test_contract.py', 'tests/test_checkpoint_io.py',
    'tests/test_native_runtime.py',
}
PRIVATE = [
    re.compile(r'[A-Za-z]:[/\\](?:Users[/\\]|app[/\\]|mnt[/\\])', re.I),
    re.compile(r'/mnt/[a-z0-9_.-]+/(?:test/)?VLA_test(?:/|\b)', re.I),
    re.compile(r'connect\.[a-z0-9]+\.seetacloud\.com', re.I),
    re.compile(r'(?:oc_sk_|github_pat_|ghp_)[A-Za-z0-9_-]{18,}'),
    re.compile(r'\bsk-[A-Za-z0-9_-]{24,}'),
    re.compile(r'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----'),
]


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True,
                          check=True).stdout


def audit():
    candidates = set(git('ls-files', '-z', '--cached', '--others', '--exclude-standard')
                     .decode('utf-8').strip('\0').split('\0'))
    if candidates != ALLOWED:
        raise RuntimeError(f'Release allowlist differs: {sorted(candidates ^ ALLOWED)}')
    for name in sorted(candidates):
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError('Release cannot contain external links: ' + name)
        content = path.read_text(encoding='utf-8')
        if any(pattern.search(content) for pattern in PRIVATE):
            raise RuntimeError('Private material in release file: ' + name)
    # Scan old text blobs as well: .gitignore alone cannot sanitize Git history.
    history = git('rev-list', '--objects', '--all').decode('utf-8').splitlines()
    checked = 0
    for record in history:
        object_id = record.split(' ', 1)[0]
        if git('cat-file', '-t', object_id).strip() != b'blob':
            continue
        payload = git('cat-file', 'blob', object_id)
        if len(payload) > 2 * 1024**2:
            raise RuntimeError('Unexpected large historical release blob')
        text = payload.decode('utf-8', errors='replace')
        if any(pattern.search(text) for pattern in PRIVATE):
            raise RuntimeError('Private material found in Git history; publication refused')
        checked += 1
    sys.path.insert(0, str(ROOT / 'src'))
    from setup import verify
    verify()
    print(f'PASS: {len(candidates)} allowlisted files, {checked} historical blobs, native source locks.')


if __name__ == '__main__':
    audit()
