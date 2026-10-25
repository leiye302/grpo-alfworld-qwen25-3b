# Reproduction contract: SDAR's pure GRPO baseline

## Identity and immutable source

Source: https://github.com/ZJU-REAL/SDAR at
`29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
Native launcher: `examples/grpo_trainer/run_alfworld.sh`.
Native entry: `verl.trainer.main_ppo.run_ppo`.
Native loop: `verl.trainer.ppo.ray_trainer.RayPPOTrainer.fit`.

This is **pure GRPO with its original reference KL/entropy/invalid-action
handling**, not GRPO+OPSD at beta=0 or with a zero distillation coefficient.
There is no privileged-context Teacher, skill retrieval, skill input, OPD,
SDAR loss, RLSD advantage reweighting, HINT-SD, RETHINK, Analyzer, SFT or API.
The shared upstream checkout contains code for other algorithms, but this
entry never instantiates their trainers. A fixed base reference remains for
the original reference KL; it is not a distillation Teacher.

## Exact recipe

Both pinned Qwen2.5 Instruct backbones start independently from their
corresponding original public model. One update globally samples 16 task
groups, eight episodes each. Episodes contain a variable number of decision
rows; retain native adjustment/grouping and advantage statistics.

Retain native GRPO advantage normalization, clipped policy loss, response
mask and token-mean. No extra reasoning/action mask is introduced. LR=1e-6,
PPO global decision-row mini-batch256, micro32/GPU, one PPO epoch;
reference KL=0.01 low_var_kl; entropy=0.001; invalid-action penalty=0.1.

Interaction: max50 environment steps, history2, prompt2048, response512,
training temperature1, top_p1, top_k=-1. This standalone 3B release has four
FSDP ranks and two TP=2 vLLM rollout replicas, memory0.6, eager=false, free_cache=false.
The global batch does not increase with GPU count. env.rollout.n=8 while
actor_rollout_ref.rollout.n remains one, as required by upstream.

Relative to the eight-GPU 07ff835 release, experiment.json changes only
trainer.n_gpus_per_node to four. Default GPU IDs, Ray allocation, rank-file
checks and checkpoint binding also use four; default CPUs are32. The global
mini-batch256 and per-GPU micro32 remain unchanged: native normalization gives
local64 rows/rank and gradient accumulation, rather than local32 on eight
ranks. Retain native decision-row batch adjustment and loss aggregation.
Hardware topology changes are not a promise of bitwise numerical identity.

## Explicit adaptations

- Add the public 1.5B backbone and fixed public revisions for both sizes.
- Portable public source/model/environment preparation, project-relative paths.
- Fix validation to the included seen140 game identities, SHA hashes and
  seed2026+i, at update0 and every ten updates. Official launcher defaults
  to128 validation entries and every five updates; this difference is explicit.
- Stop at150 updates, save20/final150, keep newest two complete checkpoints,
  local JSON metrics and final HF export. Official launcher disables saves.
- Pool16 independent original text environments per Ray actor; preserve
  environment seeds, transitions, identities and result order.
- Evaluation/logging subclass, with the original training loop called through
  super. Only its import alias is changed in main_ppo. A storage-only mixin
  wraps native save/load to replace unsafe pre-save rotation with transactional
  publication and disk-derived retention after restarts.

No source implementing advantages, policy gradients, Actor backward, FSDP,
reward, response generation, observation history, prompt or action projection
is modified. Twelve core file hashes and the complete small runtime patch
are checked before launch. No authored machine's conda path, proxy, wandb key
or private filesystem is inherited from the official shell script.

## Clean-machine preparation and runtime acceptance

Use README installation steps in a fresh Linux/Python3.12 environment, with
the pinned public downloads and dependency constraints. `setup.py --framework`
only requires Git and Python's standard library; it does not require the
author's installed environments or caches. No training trajectories need
collection or download. The parquet records only declare modality/batch size;
the native ALFWorld environment supplies the actual online tasks.

Run source verification, `scripts/check_release.py`, CPU tests, and both
dry-run commands before GPU work. On actual hardware confirm:

1. Four 80GB GPUs, four FSDP ranks, two TP2 rollout replicas, intended model/revision.
2. Real environment reset, model response, parsed action, execution and feedback.
3. Finite rewards/advantages/loss/grad norm and actual parameter updates.
4. No Teacher scoring, skill injection or auxiliary distillation metrics.
5. First save contains every rank's model/optimizer/extra state plus data.pt,
   Actor configuration/tokenizer, and a verified CHECKPOINT_COMPLETE.json.
   Confirm native GPU load succeeds before relying on unattended recovery.

CPU CI executes native tensor-level GRPO advantage/loss/backward tests and
configuration/source contracts. It is not proof of GPU memory fit, NCCL,
vLLM startup, full RL training or benchmark scores. Any runtime adaptation
must be recorded; do not silently alter sampling count, loss or task budget.

## Evaluation and recovery

The same seen140 manifest is used for both sizes and the public comparison
repos. One rollout/task, temperature0.4, same limits and seeds. Native success
and action execution are untouched. Additional reasoning/strict-format metrics
are telemetry, never gates. Preserve token IDs and special-token text.

Report initial base performance and update150 with successes/140, success
rate, task-level outcomes and actual training/validation curves. Do not use
the test set to select the best checkpoint as the final result. An independently
launched exported-model evaluation starts controller numbering at0; record
its actual training source step through the export manifest.

Resume requires the same model, source, config, task-manifest and four-rank
binding. Checkpoints include optimizer, scheduler/RNG extra and dataloader,
but do not snapshot TextWorld environment iterators. Exact uninterrupted vs
resumed rollout identity is not promised. First-save failure has no complete
intermediate resume point; do not silently call a new run a continuation.
An eight-rank run/checkpoint cannot be resumed by this four-rank recipe;
preserve old output directories and use a new run directory for a fresh run.

The native FSDP serializer and loader are unchanged. The storage wrapper writes
all ranks into a same-filesystem hidden staging directory, with upstream's
pre-save/in-memory pruning disabled for that call only. It flushes files and
directories, requires all 12 rank shards plus data/config/tokenizer files,
checks torch.save ZIP structure/CRCs without unpickling, and seals every file's
size/SHA256 and the run-binding hash in CHECKPOINT_COMPLETE.json. The complete
directory is atomically renamed into global_step_N; a full SHA reread must
succeed before atomically updating latest and deleting older verified saves.
On a failed new save, old complete saves and the canonical pointer survive.

Both saving and --resume scan checkpoint directories numerically, retaining
the newest two verified same-binding saves, not a process-local saved-path
list. Resume repairs a stale/missing/broken pointer and pins the native loader
to the exact verified directory. Legacy native saves inside the same bound run
are structurally/CRC/hash validated and sealed on first recovery. Partial,
corrupt, linked or foreign-manifest directories and interrupted staging trees
remain evidence; they are neither valid resume points nor automatic deletion
targets. Thus there can be extra *unverified* evidence on disk after failures.
Recovery/export/saving use the same exclusive run lock. Exports verify their
source, and hf150 remains outside resumable-checkpoint rotation.

Checksum reads add storage I/O only. Keep enough free space for two existing
complete saves plus a third in progress; never delete a good save to make room
for an unverified one. CPU tests cover four-rank small-file layouts, rejection
of eight-rank saves/bindings, failure injection and real torch.save/load
roundtrips, not four-GPU FSDP/NCCL recovery.

Budget at least200GiB free, preferably250GiB, for this one model, newest two
complete checkpoints, public resources and save/export peaks. This is
an estimate, not a measured minimum. No checkpoints or run logs ship publicly.
