import ast
from collections import defaultdict
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import WORLD_SIZE, checkpoint_complete, devices, load, protocol, tasks
from fixed_validation import fixed_validation_envs
from run import assert_grpo_only, config
from setup import verify


def test_native_source_and_task_locks():
    lock = verify()
    assert len(lock['unchanged_files']) >= 12
    assert 'verl/trainer/ppo/ray_trainer.py' in lock['unchanged_files']
    assert 'verl/workers/actor/dp_actor.py' in lock['unchanged_files']
    assert lock['native_launcher'] == 'examples/grpo_trainer/run_alfworld.sh'
    assert len(tasks()) == 140


@pytest.mark.parametrize('size', ['1.5b', '3b'])
def test_composed_native_grpo_recipe(size, tmp_path):
    cfg = config(size, tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    assert (cfg.trainer.n_gpus_per_node, cfg.ray_init.num_gpus) == (4, 4)
    assert cfg.ray_init.num_cpus == 32
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.6
    assert not cfg.actor_rollout_ref.rollout.enforce_eager
    assert not cfg.actor_rollout_ref.rollout.free_cache_engine
    assert not cfg.actor_rollout_ref.model.use_fused_kernels
    assert cfg.algorithm.adv_estimator == 'grpo'
    assert cfg.data.train_batch_size == 16 and cfg.env.rollout.n == 8
    assert cfg.actor_rollout_ref.rollout.n == 1
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 256
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 32
    assert cfg.actor_rollout_ref.actor.kl_loss_coef == 0.01
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0.001
    assert cfg.env.max_steps == 50 and cfg.env.history_length == 2
    assert cfg.data.max_response_length == 512 and cfg.data.max_prompt_length == 2048
    assert cfg.data.val_batch_size == 140
    assert cfg.trainer.total_training_steps == 150
    assert cfg.trainer.test_freq == 10 and cfg.trainer.save_freq == 20
    assert cfg.trainer.max_actor_ckpt_to_keep == 2
    for name in ('sdar', 'skillsd', 'rlsd', 'opsd', 'opd', 'local_hints'):
        assert name not in cfg.algorithm
    assert cfg.actor_rollout_ref.model.get('lora_rank', 0) == 0


@pytest.mark.parametrize('name', ['sdar', 'skillsd', 'rlsd', 'opsd', 'opd', 'local_hints'])
def test_distillation_config_is_rejected(name, tmp_path):
    from omegaconf import OmegaConf
    cfg = config('1.5b', tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    OmegaConf.update(cfg, 'algorithm.' + name, {}, force_add=True)
    with pytest.raises(RuntimeError, match='Unexpected distillation'):
        assert_grpo_only(cfg)


@pytest.mark.parametrize('name', ['use_sdar_loss', 'use_sdl_loss', 'use_opd_loss'])
def test_distillation_actor_flag_is_rejected(name, tmp_path):
    from omegaconf import OmegaConf
    cfg = config('3b', tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    OmegaConf.update(cfg, 'actor_rollout_ref.actor.' + name, True, force_add=True)
    with pytest.raises(RuntimeError, match='auxiliary distillation'):
        assert_grpo_only(cfg)


@pytest.mark.parametrize('spec', ['0,1', '0,1,2', '0,1,2,2', '0,1,2,3,4',
                                 '0,1,2,3,4,5,6,7', '', '0,1,2,GPU-xxx'])
def test_invalid_gpu_allocation(spec):
    with pytest.raises(ValueError):
        devices(spec)


def test_four_gpu_allocation():
    assert WORLD_SIZE == 4
    assert devices('0,1,2,3') == [str(i) for i in range(4)]
    assert devices('4,5,6,7') == [str(i) for i in range(4, 8)]


def test_partial_checkpoint_rejected(tmp_path):
    with pytest.raises(RuntimeError):
        checkpoint_complete(tmp_path)
    for rank in range(WORLD_SIZE):
        for kind in ('model', 'optim', 'extra_state'):
            path = tmp_path/'actor'/f'{kind}_world_size_{WORLD_SIZE}_rank_{rank}.pt'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'unit-test')
    (tmp_path/'actor/config.json').write_text('{}')
    (tmp_path/'data.pt').write_bytes(b'unit-test')
    assert checkpoint_complete(tmp_path) == tmp_path


def test_eight_rank_checkpoint_is_not_a_four_rank_resume(tmp_path):
    for rank in range(8):
        for kind in ('model', 'optim', 'extra_state'):
            path = tmp_path/'actor'/f'{kind}_world_size_8_rank_{rank}.pt'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'unit-test')
    (tmp_path/'actor/config.json').write_text('{}')
    (tmp_path/'data.pt').write_bytes(b'unit-test')
    with pytest.raises(RuntimeError, match='Incomplete resumable checkpoint'):
        checkpoint_complete(tmp_path)


def test_telemetry_does_not_accept_duplicate_actions():
    assert protocol('<think>Plan.</think><action>look</action>')['strict_protocol']
    assert not protocol('<think></think><action>look</action>')['strict_protocol']
    assert not protocol('<think>Plan.</think><action>look</action><action>look</action>')['strict_protocol']


def test_fixed_task_preserves_native_environment_and_checks_sha(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setenv('ALFWORLD_DATA', str(tmp_path))
    game = tmp_path/'json_2.1.1/valid_seen/task/trial/game.tw-pddl'
    game.parent.mkdir(parents=True)
    game.write_bytes(b'game')
    manifest = tmp_path/'manifest.json'
    row = {'task_id': game.relative_to(tmp_path).as_posix(), 'seed': 2026,
           'game_sha256': hashlib.sha256(b'game').hexdigest()}
    manifest.write_text(json.dumps([row]))
    base = SimpleNamespace(game_files=[str(game)], num_games=140)
    selected = fixed_validation_envs(base, manifest, 1, 1)
    assert selected[0].fixed_validation_seed == 2026
    assert base.num_games == 140 and selected[0] is not base
    row['game_sha256'] = '0'*64
    manifest.write_text(json.dumps([row]))
    with pytest.raises(ValueError):
        fixed_validation_envs(base, manifest, 1, 1)


def test_entry_is_real_main_ppo_without_teacher_training_path():
    source = (ROOT/'src/run.py').read_text()
    assert 'from verl.trainer.main_ppo import run_ppo' in source
    assert 'run_ppo(cfg)' in source
    assert 'run_sdar' not in source
    native_main = (ROOT/'.deps/sdar/verl/trainer/main_ppo.py').read_text()
    assert 'from grpo_eval import FixedSeenTrainer as RayPPOTrainer' in native_main
    assert 'TeacherPolicy' not in native_main
    assert 'SkillProvider' not in native_main
    tree = ast.parse((ROOT/'.deps/sdar/verl/trainer/ppo/ray_trainer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    fit_source = ast.unparse(fit)
    assert 'teacher_log_probs' not in fit_source
    assert '_compute_teacher_log_probs' not in fit_source
    assert 'teacher_forward' not in fit_source


def test_native_grpo_loss_has_real_gradient_and_optimizer_update():
    torch = pytest.importorskip('torch')
    import numpy as np
    source = ROOT/'.deps/sdar/verl/trainer/ppo/core_algos.py'
    tree = ast.parse(source.read_text())
    names = {'compute_grpo_outcome_advantage', 'agg_loss', 'compute_policy_loss'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == 3
    namespace = {'torch': torch, 'np': np, 'defaultdict': defaultdict,
                 'verl_F': SimpleNamespace(masked_mean=lambda v,m:(v*m).sum()/m.sum())}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    mask = torch.tensor([[1.,1.,0.]]*4)
    rewards = torch.tensor([[0.,10.,0.], [0.,0.,0.], [0.,10.,0.], [0.,0.,0.]])
    advantages, _ = namespace['compute_grpo_outcome_advantage'](
        rewards, mask, np.array(['a','a','b','b']), np.array(['a0','a1','b0','b1']))
    assert torch.all(advantages[[0,2],:2] > 0)
    assert torch.all(advantages[[1,3],:2] < 0)
    assert not advantages.requires_grad and torch.count_nonzero(advantages[:,2]) == 0
    student = torch.nn.Parameter(torch.full_like(mask, -1.0))
    old = student.detach().clone()
    optimizer = torch.optim.SGD([student], lr=0.1)
    loss = namespace['compute_policy_loss'](old, student, advantages, mask, cliprange=0.2)[0]
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad[0,0] < 0 and student.grad[1,0] > 0
    assert torch.count_nonzero(student.grad[:,2]) == 0
    optimizer.step()
    assert student[0,0] > old[0,0] and student[1,0] < old[1,0]
    assert torch.equal(student[:,2], old[:,2])


def test_worker_pool_preserves_native_seeds_order_and_results():
    source = ROOT/'.deps/sdar/agent_system/environments/env_package/alfworld/envs.py'
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in {'AlfworldWorker', 'AlfworldWorkerPool', 'AlfworldEnvs'}]

    class NativeEnvironment:
        def seed(self, seed): self.value = seed
        def reset(self): return [str(self.value)], {'admissible_commands': [['look']]}
        def step(self, actions):
            return [str(self.value)+actions[0]], [0], [False], {'admissible_commands': [['look']], 'won': [False]}

    class Base:
        def init_env(self, batch_size): return NativeEnvironment()

    class Method:
        def __init__(self, fn): self.remote = fn

    class Handle:
        def __init__(self, obj): self.obj = obj
        def __getattr__(self, name): return Method(getattr(self.obj, name))

    class Remote:
        def __init__(self, cls): self.cls = cls
        def remote(self, *args): return Handle(self.cls(*args))

    ray = SimpleNamespace(is_initialized=lambda: True, get=lambda x:x,
                          remote=lambda **kwargs: lambda cls:Remote(cls))
    namespace = {'gym': SimpleNamespace(Env=object), 'ray': ray, 'get_environment': lambda _: lambda *a,**k:Base(),
                 'load_config_file': lambda _: {'env': {'type': 'AlfredTWEnv'}},
                 'compute_reward': lambda info,multi_modal: 0}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), namespace)
    outputs = []
    for pool_size in (1, 2, 4):
        env = namespace['AlfworldEnvs']('unused', 17, 3, 2, {}, env_kwargs={'worker_pool_size':pool_size})
        outputs.append((copy.deepcopy(env.reset()), copy.deepcopy(env.step(['look']*6))))
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0][0][0] == ['17', '17', '18', '18', '19', '19']


def test_dry_run_needs_no_model_or_gpu(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'src/run.py'), 'train', '--size', '1.5b',
                             '--run-dir', str(tmp_path/'unused'), '--dry-run'], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path/'unused').exists()
    assert 'adv_estimator: grpo' in result.stdout
    assert 'gate_beta' not in result.stdout and 'skills_dir' not in result.stdout


def test_models_share_all_scientific_settings(tmp_path):
    from omegaconf import OmegaConf
    left = config('1.5b', tmp_path/'model15', tmp_path/'assets', tmp_path/'run15')
    right = config('3b', tmp_path/'model3', tmp_path/'assets', tmp_path/'run3')
    for cfg in (left, right):
        cfg.actor_rollout_ref.model.path = 'model'
        cfg.critic.model.tokenizer_path = 'model'
        cfg.reward_model.model.input_tokenizer = 'model'
        cfg.trainer.experiment_name = 'experiment'
        cfg.data.train_files = cfg.data.val_files = 'parquet'
        cfg.trainer.default_local_dir = cfg.trainer.validation_data_dir = cfg.trainer.rollout_data_dir = 'output'
        cfg.ray_init.runtime_env.env_vars.GRPO_RUN_DIR = 'output'
    assert OmegaConf.to_container(left, resolve=True) == OmegaConf.to_container(right, resolve=True)


def test_scientific_settings_match_official_grpo_launcher_except_gpu_count(tmp_path):
    from omegaconf import OmegaConf
    source = (ROOT/'.deps/sdar/examples/grpo_trainer/run_alfworld.sh').read_text()
    assignments = dict(re.findall(r'(?m)^\s*\+?([\w.]+)=([^\s\\]+)', source))
    cfg = config('3b', tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    keys = [
        'algorithm.adv_estimator', 'data.train_batch_size', 'data.max_prompt_length',
        'data.max_response_length', 'data.filter_overlong_prompts', 'data.truncation',
        'data.return_raw_chat', 'actor_rollout_ref.actor.optim.lr',
        'actor_rollout_ref.model.use_remove_padding',
        'actor_rollout_ref.actor.ppo_mini_batch_size',
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu',
        'actor_rollout_ref.actor.use_kl_loss', 'actor_rollout_ref.actor.kl_loss_coef',
        'actor_rollout_ref.actor.kl_loss_type',
        'actor_rollout_ref.model.enable_gradient_checkpointing',
        'actor_rollout_ref.actor.fsdp_config.param_offload',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu',
        'actor_rollout_ref.rollout.tensor_model_parallel_size',
        'actor_rollout_ref.rollout.gpu_memory_utilization',
        'actor_rollout_ref.rollout.enable_chunked_prefill',
        'actor_rollout_ref.rollout.enforce_eager',
        'actor_rollout_ref.rollout.free_cache_engine',
        'actor_rollout_ref.rollout.val_kwargs.temperature',
        'actor_rollout_ref.rollout.val_kwargs.do_sample',
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu',
        'actor_rollout_ref.ref.fsdp_config.param_offload',
        'actor_rollout_ref.actor.use_invalid_action_penalty',
        'actor_rollout_ref.actor.invalid_action_penalty_coef',
        'algorithm.use_kl_in_reward', 'env.env_name', 'env.seed', 'env.max_steps',
        'env.rollout.n', 'trainer.critic_warmup',
        'trainer.ray_wait_register_center_timeout', 'trainer.nnodes',
        'trainer.total_epochs', 'trainer.val_before_train',
    ]
    for key in keys:
        raw = assignments[key]
        if raw.startswith('$'):
            raw = assignments[raw[1:]]
        if raw.lower() in ('true', 'false'):
            expected = raw.lower() == 'true'
        elif raw.startswith("'"):
            expected = raw.strip("'")
        else:
            try:
                expected = float(raw)
            except ValueError:
                expected = raw
        assert OmegaConf.select(cfg, key) == expected, key


def test_native_gradient_accumulation_preserves_global_batch_on_four_gpus(tmp_path):
    cfg = config('3b', tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    source = ROOT/'.deps/sdar/verl/workers/fsdp_workers.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    normalization = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
        and ast.unparse(node.test) == 'self._is_actor'
        and 'self.config.actor.ppo_mini_batch_size *= self.config.rollout.n' in ast.unparse(node))
    results = {}
    for ranks in (4, 8):
        worker = SimpleNamespace(_is_actor=True, config=copy.deepcopy(cfg.actor_rollout_ref),
            device_mesh=SimpleNamespace(size=lambda: ranks), ulysses_sequence_parallel_size=1)
        exec(compile(ast.Module(body=[normalization], type_ignores=[]), str(source), 'exec'), {'self': worker})
        local = worker.config.actor.ppo_mini_batch_size
        micro = worker.config.actor.ppo_micro_batch_size_per_gpu
        assert ranks * local == cfg.actor_rollout_ref.actor.ppo_mini_batch_size
        results[ranks] = (local, micro, local // micro)
    assert results == {4: (64, 32, 2), 8: (32, 32, 1)}


def test_validation_uses_native_padding_and_runtime_directory_is_composable(tmp_path):
    source = (ROOT/'.deps/sdar/agent_system/multi_turn_rollout/rollout_loop.py').read_text()
    assert 'pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)' in source
    assert 'unpad_dataproto(batch_output_padded, pad_size=pad_size)' in source
    cfg = config('1.5b', tmp_path/'model', tmp_path/'assets', tmp_path/'run')
    cfg.ray_init._temp_dir = str(tmp_path/'ray')
    assert cfg.ray_init._temp_dir == str(tmp_path/'ray')


def test_evaluation_hook_preserves_native_success_and_all_tasks(tmp_path, monkeypatch):
    import importlib.util
    import types
    torch = pytest.importorskip('torch')
    rows = tasks()
    gamefiles = [str(tmp_path/row['task_id']) for row in rows]
    batch = SimpleNamespace(
        batch={'responses': torch.ones((140,2), dtype=torch.long),
               'attention_mask': torch.ones((140,4), dtype=torch.long)},
        non_tensor_batch={'gamefile': gamefiles, 'traj_uid': [str(i) for i in range(140)],
                          'turn_step': [0]*140, 'episode_rewards': [10.0]*70+[0.0]*70,
                          'is_action_valid': [True]*140})

    class NativeTrainer:
        def _validate(self):
            self.traj_collector.multi_turn_loop()
            return {'val/success_rate': 0.5}

    native_module = types.ModuleType('verl.trainer.ppo.ray_trainer')
    native_module.RayPPOTrainer = NativeTrainer
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.ray_trainer', native_module)
    spec = importlib.util.spec_from_file_location('tested_grpo_eval', ROOT/'src/grpo_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('ALFWORLD_DATA', str(tmp_path))
    monkeypatch.setenv('GRPO_RUN_DIR', str(tmp_path/'run'))
    trainer = module.FixedSeenTrainer()
    trainer.global_steps = 10
    original = lambda:batch
    trainer.traj_collector = SimpleNamespace(multi_turn_loop=original)
    trainer.tokenizer = SimpleNamespace(decode=lambda ids,**kw:'<think>Plan.</think><action>look</action>')
    assert trainer._validate() == {'val/success_rate':0.5}
    assert trainer.traj_collector.multi_turn_loop is original
    summary = json.loads((tmp_path/'run/evaluation/update_010/summary.json').read_text())
    assert summary['successes'] == 70 and len(summary['task_results']) == 140
    assert summary['reasoning_complete_rate'] == 1


def test_release_allowlist_and_privacy():
    result = subprocess.run([sys.executable, str(ROOT/'scripts/check_release.py')], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
