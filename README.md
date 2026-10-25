# GRPO on ALFWorld — Qwen2.5-3B, 8×A100 80GB

独立的 **Qwen2.5-3B-Instruct 纯 GRPO** 仓库，采用 [SDAR 公开实现](https://github.com/ZJU-REAL/SDAR) 的原生 GRPO，不含 OPSD／Teacher／RETHINK／Analyzer／SFT。

从双尺寸仓库的已验证提交 `9e6de9d13b12f16008a50fb6886388b25ab0d535` 拆分。**Python 源码、依赖、上游锁、训练配方、评测任务和测试代码全部保持原样**；只将本仓库的 shell 启动入口固定为 3b，并调整本 README。共享底层工具保留原通用参数接口，避免重写训练代码；按下列本仓库入口运行即可，不会顺序训练另一个尺寸。

## 1. 固定方法与配置

| 项目 | 本仓库 |
| --- | --- |
| 初始化 | Qwen/Qwen2.5-3B-Instruct，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| 上游 | ZJU-REAL/SDAR，`29dcb12a55d8fc17ba6ae03fd0a69048f43051f3` |
| 原生入口 | `examples/grpo_trainer/run_alfworld.sh` → `verl.trainer.main_ppo` |
| GPU | 每次使用完整 8×A100 80GB；八 rank FSDP，rollout TP=2 |
| 训练 | 150 updates；全局16任务×8轨迹；LR=1e-6 |
| PPO | 全局决策行mini-batch256，micro32/GPU，epoch1，原生token-mean |
| 正则 | 原生reference KL=0.01 low_var_kl；entropy=0.001；非法动作惩罚=0.1 |
| 交互 | max steps50，history2，prompt2048，response512；训练temperature1 |
| 验证 | update0及每10步；固定seen140；一任务一rollout，temperature0.4 |
| 保存 | 每20步和最终150；新断点完整保存及校验后，才删除旧断点；保留最近两份 |

全局采样量不随八卡放大。保留原生优势、response mask、policy loss、Actor backward、FSDP、动作执行和奖励；没有 Teacher 蒸馏，也不是将 OPSD 系数设为0来模拟 GRPO。完整方法边界见 [复现合同](docs/REPRODUCTION.md)，精确参数见 [experiment.json](configs/experiment.json)。

## 2. 新服务器准备

只依赖公开仓库、公开固定版本模型和 ALFWorld 资源，不依赖维护者本机、私有断点或私人数据。需要 Linux、Python3.12、兼容CUDA12.8的驱动和八张已获配 A100 80GB；建议64个可用逻辑CPU及256–512GiB主机内存。存盘要留够旧两份完整断点加第三份新写入的峰值空间。

```bash
git clone https://github.com/leiye302/grpo-alfworld-qwen25-3b.git
cd grpo-alfworld-qwen25-3b
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
python src/setup.py --framework
python -m pip install -r configs/requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e .deps/sdar -c configs/requirements.txt
python -m pip check

cp .env.example .env
# 如有需要，编辑自己获配的八张卡、CPU数和ALFWorld资源路径。
set -a
source .env
set +a
python src/setup.py --download-model 3b
alfworld-download
python src/setup.py --verify
python scripts/check_release.py
python -m pytest tests -q
```

FlashAttention需要匹配的wheel，或本机CUDA Toolkit/nvcc编译。不要另跑上游未约束的全量依赖安装而升级环境。模型和游戏身份均固定并核验SHA；没有大权重或私有数据随代码上传。

## 3. 运行本尺寸

```bash
# 只检查配置，不占GPU。
bash scripts/train.sh --dry-run

# 正式训练：本入口已固定 3b，无需另传 --size。
bash scripts/train.sh

# 训练完成后导出。
python src/run.py export --size 3b

# 中断后：确认旧控制器/worker已退出，再续跑。
bash scripts/train.sh --resume
```

建议在 tmux 或调度器内运行。也可用 `bash scripts/run_both.sh` 自动完成本尺寸训练、导出和曲线汇总；这个文件名为兼容原仓库保留，**在本仓库只跑 3b，不跑另一尺寸**。不要在同一组卡上重复启动。

原生Ray负责启动八rank和TP2副本，不加外层torchrun，不连接其他作业，不执行整机 `ray stop`。默认拒绝覆盖旧输出，同一run有排他锁。首轮真机确认环境交互、有限reward/loss/grad norm、真实参数更新及完整存盘；CPU测试不替代GPU验收。

## 4. 统一评测和输出

固定 [seen140_tasks.json](configs/seen140_tasks.json)：valid_seen的140个唯一任务，seed=2026+i；max steps50、history2、response512、temperature0.4。这是统一的seen140主实验协议，不声称逐项沿用官方128条验证。保持原生执行/奖励，额外格式指标仅作遥测，不作训练门禁。

```bash
bash scripts/eval.sh --model "$PWD/runs/grpo_3b/hf150" \
  --run-dir "$PWD/runs/grpo_3b_final_eval"
python src/report.py
```

训练与初始基模成绩保存在 `runs/grpo_3b/evaluation/update_000`，其后每10步同一组任务；完整结果在 `metrics.jsonl`、逐任务summary/decisions和最终150步目录。报告固定update150，不从测试集挑最高断点。独立评测控制器从0编号，实际权重步数以导出清单为准。

## 5. 保存与续跑保证

沿用已修复的事务式保存：八rank先写私有暂存目录，检查model/optimizer/extra、data.pt、配置/tokenizer、torch.save归档CRC及逐文件SHA256，生成 `CHECKPOINT_COMPLETE.json`；原子发布并重新核验后，才更新latest并清理旧的。

每次保存和 `--resume` 均扫描磁盘，按update编号保留最近两份已校验的同run完整断点，不依赖进程内列表；修复缺失/损坏/过时指针。失败的新存盘不会先删旧断点。损坏、未完成、不同绑定或带链接的目录保留作证据，不加载也不自动删除，因此故障证据可能额外占空间。无完整断点时明确报错，不偷偷从0重跑。最终HF导出不参与续训断点轮转。

原生断点包含optimizer、scheduler/rank RNG和dataloader，但不保存TextWorld环境迭代器，因此不承诺中断前后rollout逐位相同。完整SHA校验只增加存盘I/O，不改变训练设置。

## 6. 验证与许可

GitHub Actions从干净Linux检出公开获取上游，运行源码/配方锁、原生GRPO梯度测试、固定任务、存盘故障注入、真实CPU torch.save/load往返和启动检查。**没有在8×A100上运行正式训练或GPU恢复验收，没有预填成绩。**

包装代码/上游框架采用Apache-2.0；模型遵守各自许可（1.5B为Apache-2.0，3B为Qwen Research License）。资源许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。
