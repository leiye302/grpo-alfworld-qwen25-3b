"""Four-GPU native GRPO launcher from public SDAR, with no distillation."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from common import ROOT, UPSTREAM, WORLD_SIZE, atomic_json, devices, load, sha, tasks
from checkpoint_io import CheckpointStore
from setup import verify


def base_model(size):
    return ROOT / 'models' / load('models.json')[size]['model_id'].split('/')[-1]


def assert_grpo_only(cfg):
    if cfg.algorithm.adv_estimator != 'grpo':
        raise RuntimeError('This release requires the native GRPO advantage estimator')
    for name in ('sdar', 'skillsd', 'rlsd', 'opsd', 'opd', 'local_hints'):
        if name in cfg.algorithm:
            raise RuntimeError('Unexpected distillation configuration: ' + name)
    for name in ('use_sdar_loss', 'use_sdl_loss', 'use_opd_loss'):
        if cfg.actor_rollout_ref.actor.get(name, False):
            raise RuntimeError('Pure GRPO must not enable auxiliary distillation: ' + name)
    if cfg.get('opd_teacher') is not None:
        raise RuntimeError('Pure GRPO must not configure an OPD Teacher')


def config(size, model, assets, run_dir, cpus=32, evaluate=False):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(UPSTREAM / 'verl/trainer/config'), version_base=None):
        cfg = compose(config_name='ppo_trainer')
    for key, value in load('experiment.json').items():
        OmegaConf.update(cfg, key, value, force_add=True)
    updates = {
        'actor_rollout_ref.model.path': str(model),
        'data.train_files': str(run_dir / 'train.parquet'),
        'data.val_files': str(run_dir / 'val.parquet'),
        'env.alfworld.fixed_eval_manifest': str(ROOT / 'configs/seen140_tasks.json'),
        'trainer.experiment_name': 'grpo_qwen25_' + size,
        'trainer.default_local_dir': str(run_dir / 'checkpoints'),
        'trainer.validation_data_dir': str(run_dir / 'validation_native'),
        'trainer.rollout_data_dir': str(run_dir / 'rollouts'),
        'trainer.val_only': evaluate,
        'ray_init.address': 'local', 'ray_init.num_gpus': WORLD_SIZE,
        'ray_init.num_cpus': cpus, 'ray_init.include_dashboard': False,
        'ray_init._temp_dir': None,
        'ray_init.object_store_memory': 2 * 1024**3,
        'ray_init.runtime_env.env_vars.PYTHONPATH': os.pathsep.join([str(ROOT / 'src'), str(UPSTREAM)]),
        'ray_init.runtime_env.env_vars.ALFWORLD_DATA': str(assets),
        'ray_init.runtime_env.env_vars.GRPO_RUN_DIR': str(run_dir),
    }
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    OmegaConf.resolve(cfg)
    assert_grpo_only(cfg)
    return cfg


def verify_model(model, size, evaluate):
    if not (model / 'config.json').is_file() or not list(model.glob('*.safetensors')):
        raise RuntimeError('Missing model weights; run setup.py --download-model first')
    if not evaluate:
        receipt = json.loads((model / 'download_receipt.json').read_text())
        expected = load('models.json')[size]
        if any(receipt[k] != expected[k] for k in ('model_id', 'revision')):
            raise RuntimeError('Training must start from the pinned original Instruct model')
        if not receipt['files']:
            raise RuntimeError('Empty base-model receipt')
        for name, info in receipt['files'].items():
            if sha(model / name) != info['sha256']:
                raise RuntimeError('Base-model hash mismatch: ' + name)


def data(run_dir):
    import pandas as pd
    for split, count in [('train', 16), ('val', 140)]:
        records = [{'data_source': 'text', 'prompt': [{'role': 'user', 'content': ''}],
                    'ability': 'agent', 'extra_info': {'split': split, 'index': i}}
                   for i in range(count)]
        pd.DataFrame(records).to_parquet(run_dir / (split + '.parquet'), index=False)


@contextlib.contextmanager
def run_lock(run_dir):
    import fcntl
    with (run_dir / 'run.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This output directory already has a running controller') from None
        yield


def launch(args):
    verify()
    gpu_ids = devices(args.gpus)
    assets = args.assets.resolve()
    model = (args.model or base_model(args.size)).resolve()
    run_dir = (args.run_dir or ROOT / 'runs' / ('grpo_' + args.size)).resolve()
    cfg = config(args.size, model, assets, run_dir, args.cpus, args.command == 'eval')
    if args.dry_run:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(cfg))
        return
    if sys.platform != 'linux':
        raise RuntimeError('Native GRPO training/evaluation requires Linux with CUDA')
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(gpu_ids), ALFWORLD_DATA=str(assets),
                      TOKENIZERS_PARALLELISM='false', VLLM_USE_V1='1', RAY_ADDRESS='local',
                      GRPO_RUN_DIR=str(run_dir))
    os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(UPSTREAM)])
    verify_model(model, args.size, args.command == 'eval')
    tasks(assets)
    binding = {'size': args.size, 'model': str(model), 'assets': str(assets),
               'recipe_sha256': sha(ROOT / 'configs/experiment.json'),
               'upstream_lock_sha256': sha(ROOT / 'configs/upstream.json'),
               'task_manifest_sha256': sha(ROOT / 'configs/seen140_tasks.json'),
               'world_size': WORLD_SIZE, 'command': args.command}
    if args.resume:
        if args.command != 'train' or json.loads((run_dir / 'binding.json').read_text()) != binding:
            raise RuntimeError('Resume binding differs from the original run')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(run_dir / 'binding.json', binding)
        data(run_dir)
    with run_lock(run_dir):
        if args.resume:
            # Disk-wide recovery and pruning must happen under the same controller lock.
            checkpoint = CheckpointStore(run_dir).recover()
            step = int(checkpoint.name.removeprefix('global_step_'))
            if not 0 < step < 150:
                raise RuntimeError('Training already completed or no valid intermediate update below 150')
            cfg.trainer.resume_mode = 'resume_path'
            cfg.trainer.resume_from_path = str(checkpoint)
        import torch
        if torch.cuda.device_count() != WORLD_SIZE:
            raise RuntimeError('Four allocated CUDA devices are required')
        for index in range(WORLD_SIZE):
            props = torch.cuda.get_device_properties(index)
            if props.total_memory < 70 * 1024**3:
                raise RuntimeError('This recipe targets four A100 80GB GPUs')
        runtime = Path(tempfile.mkdtemp(prefix='gp-', dir=os.environ.get('GRPO_RUNTIME_ROOT')))
        cfg.ray_init._temp_dir = str(runtime)
        from omegaconf import OmegaConf
        OmegaConf.save(cfg, run_dir / 'resolved_config.yaml')
        atomic_json(run_dir / 'runtime.json', {'ray': str(runtime), 'gpus': gpu_ids})
        sys.path[:0] = [str(UPSTREAM), str(ROOT / 'src')]
        os.chdir(UPSTREAM)
        # Same native task pool as the official launcher. Do not create an offline RL dataset.
        if args.command == 'train':
            import yaml
            from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
            native_config = yaml.safe_load((UPSTREAM / 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml').read_text())
            native_env = get_environment(native_config['env']['type'])(native_config, train_eval='train')
            training_games = {Path(x).resolve() for x in native_env.game_files}
            validation_games = {(assets / row['task_id']).resolve() for row in tasks()}
            if len(training_games) != 3553 or training_games & validation_games:
                raise RuntimeError('Expected native 3553 training games, disjoint from seen140')
            atomic_json(run_dir / 'training_pool.json', {'count': len(training_games),
                        'task_ids': sorted(p.relative_to(assets).as_posix() for p in training_games)})
        from verl.trainer.main_ppo import run_ppo
        import ray
        try:
            run_ppo(cfg)
            if args.command == 'train':
                CheckpointStore(run_dir).verify(run_dir / 'checkpoints/global_step_150')
            atomic_json(run_dir / 'completion.json', {'status': 'completed', 'command': args.command,
                        'updates': 150 if args.command == 'train' else 0, 'binding': binding})
        finally:
            ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'eval', 'export'])
    parser.add_argument('--size', choices=['1.5b', '3b'], required=True)
    parser.add_argument('--assets', type=Path, default=Path(os.environ.get('ALFWORLD_DATA', ROOT / 'data/alfworld')))
    parser.add_argument('--gpus', default=os.environ.get('GRPO_GPUS', '0,1,2,3'))
    parser.add_argument('--cpus', type=int, default=int(os.environ.get('GRPO_CPUS', '32')))
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--model', type=Path, help='Exported HF directory, evaluation only')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Compose and print config, no CUDA/model/data access')
    args = parser.parse_args()
    if args.cpus < 24:
        parser.error('At least 24 allocated logical CPUs are recommended for the pooled configuration')
    if args.model and args.command != 'eval':
        parser.error('--model is only accepted for evaluation; training uses the pinned base model')
    if args.command == 'export':
        verify()
        run_dir = (args.run_dir or ROOT / 'runs' / ('grpo_' + args.size)).resolve()
        with run_lock(run_dir):
            store = CheckpointStore(run_dir)
            target = run_dir / 'hf150'
            if target.exists():
                raise RuntimeError('Export destination already exists; inspect it rather than overwrite')
            checkpoint = store.recover()
            if checkpoint.name != 'global_step_150':
                raise RuntimeError('Export requires the verified final update 150 checkpoint')
            env = dict(os.environ, PYTHONPATH=str(UPSTREAM))
            subprocess.run([sys.executable, str(UPSTREAM / 'scripts/model_merger.py'), 'merge', '--backend', 'fsdp',
                            '--local_dir', str(checkpoint / 'actor'), '--target_dir', str(target)],
                           cwd=UPSTREAM, env=env, check=True)
            files = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
            atomic_json(run_dir / 'export_manifest.json', {'step': 150, 'files': files})
    else:
        if args.command == 'eval' and not args.model:
            parser.error('Evaluation requires --model /path/to/exported-hf-model')
        launch(args)


if __name__ == '__main__':
    main()
