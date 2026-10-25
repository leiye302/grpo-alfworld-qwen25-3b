"""Regression checks for native execution defaults; no CUDA or training."""
import hashlib
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_native_execution_defaults():
    cfg = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    expected = {
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 2,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6,
        "actor_rollout_ref.rollout.enforce_eager": False,
        "actor_rollout_ref.rollout.free_cache_engine": False,
        "actor_rollout_ref.model.use_fused_kernels": False,
        "env.resources_per_worker.num_cpus": 0.1,
    }
    for key, value in expected.items():
        assert cfg[key] == value, key
    assert "env.alfworld.worker_pool_size" not in cfg
    assert not any("fused_kernel_options" in key for key in cfg)
    native = ROOT / ".deps/sdar/agent_system/environments/env_package/alfworld/envs.py"
    text = native.read_text(encoding="utf-8")
    assert "AlfworldWorkerPool" not in text
    assert "self.pool_size" not in text
    assert "worker.step.remote(actions[i])" in text


def test_runtime_temp_directory_can_be_assigned(tmp_path):
    from run import config
    cfg = config("3b", tmp_path/"model", tmp_path/"assets", tmp_path/"run")
    cfg.ray_init._temp_dir = str(tmp_path/"ray")
    assert cfg.ray_init._temp_dir == str(tmp_path/"ray")


def test_native_scheduler_cpu_budget():
    cfg = json.loads((ROOT/"configs/experiment.json").read_text(encoding="utf-8"))
    demand = ((cfg["data.train_batch_size"] * cfg["env.rollout.n"] + cfg["data.val_batch_size"])
              * cfg["env.resources_per_worker.num_cpus"] + cfg["trainer.n_gpus_per_node"] + 1)
    minimum = 32 if cfg["trainer.n_gpus_per_node"] == 4 else 40
    assert demand <= minimum
    source = (ROOT/"src/run.py").read_text(encoding="utf-8")
    assert "args.cpus < minimum_cpus" in source



def test_scientific_recipe_is_unchanged():
    cfg = json.loads((ROOT/"configs/experiment.json").read_text(encoding="utf-8"))
    runtime_fields = ('env.resources_per_worker.num_cpus', 'env.alfworld.worker_pool_size', 'actor_rollout_ref.rollout.tensor_model_parallel_size', 'actor_rollout_ref.rollout.gpu_memory_utilization', 'actor_rollout_ref.rollout.enforce_eager', 'actor_rollout_ref.rollout.free_cache_engine', 'actor_rollout_ref.model.use_fused_kernels', 'actor_rollout_ref.model.fused_kernel_options.impl_backend')
    for key in runtime_fields:
        cfg.pop(key, None)
    normalized = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(normalized).hexdigest() == "8dce1059d7a419037245098f7a0e13eaf5b4b58f7f47c1721ee61ef9f2b2d6b0"
