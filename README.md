# Train LLM From Scratch

从零开始训练大语言模型（LLM）的完整流程，包含**预训练（Pre-training）** 和 **指令微调（SFT）** 两个阶段。

基于 Qwen2.5-1.5B 架构，使用 Hugging Face Transformers Trainer 训练，SwanLab 记录实验日志。

## 项目结构

```
.
├── pretrain.py           # 预训练脚本 — 从随机权重开始训练语言模型
├── finetune.py           # 指令微调脚本 — 在预训练模型上做 SFT
├── download_model.py     # 下载 Qwen2.5-1.5B 模型
├── download_dataset.py   # 下载 BelleGroup 对话数据集
├── pretrain.sh           # 预训练启动脚本
├── finetune.sh           # 微调启动脚本
├── finetune_quick.sh     # 微调快速验证脚本（3步，约1-2分钟）
├── ds_config_zero2.json  # DeepSpeed ZeRO-2 配置（仅 CUDA 环境）
├── requirements.txt      # Python 依赖
└── data/
    └── BelleGroup/       # 微调数据集
```

## 运行平台

| 平台 | 支持 | 说明 |
|------|------|------|
| **AutoDL / Linux + CUDA GPU** | ✅ 推荐 | 使用 DeepSpeed ZeRO-2 + bf16，训练速度快 |
| **macOS (Apple Silicon MPS)** | ✅ 支持 | 无 DeepSpeed，使用 fp16 + gradient checkpointing，batch_size=1 |

> macOS 上安装依赖时，`deepspeed`（CUDA only）和 `torchdata`（已废弃）可能报错。忽略即可，本项目在 macOS 上不使用它们。

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install 'swanlab[dashboard]'  # SwanLab 本地模式需要
```

### 2. 下载模型与数据

```bash
# 下载 Qwen2.5-1.5B 模型（约 3GB）
python3 download_model.py

# 下载 BelleGroup 对话数据集（约 600MB）
python3 download_dataset.py
```

### 3. 快速验证（推荐先跑这个）

完整训练在 macOS 上耗时较长，建议先用快速验证确认环境配置正确：

```bash
./finetune_quick.sh
```

快速验证会：
- 只跑 3 个训练步（约 1-2 分钟）
- 使用短序列长度（`block_size=256`）节省显存
- 不保存 checkpoint，不连接 SwanLab
- 每步打印 loss，验证前向/反向传播和优化器正常工作

### 4. 完整指令微调（SFT）

快速验证通过后，跑完整训练：

```bash
./finetune.sh
```

预计耗时：macOS M1 Max 上约 17-33 小时，CUDA GPU 上约 1-2 小时。

脚本参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_name_or_path` | 基础模型路径 | `autodl-tmp/qwen-1.5b` |
| `--train_files` | 训练数据路径 | `./data/BelleGroup/train_3.5M_CN.json` |
| `--per_device_train_batch_size` | 每设备 batch size | `1`（macOS）/ `16`（CUDA） |
| `--gradient_accumulation_steps` | 梯度累积步数 | `8`（macOS）/ `4`（CUDA） |
| `--learning_rate` | 学习率 | `1e-4` |
| `--num_train_epochs` | 训练轮数 | `3` |
| `--block_size` | 最大序列长度 | `2048` |
| `--fp16` | 半精度训练 | macOS 启用，CUDA 建议用 `--bf16` |
| `--gradient_checkpointing` | 梯度检查点（省显存） | 启用 |

### 5. 从零预训练

> ⚠️ 需要准备预训练语料（JSONL 格式，每行一个 `{"text": "..."}`）。

```bash
# 修改 pretrain.sh 中的 --train_files 指向你的数据
./pretrain.sh
```

`pretrain.py` 支持两种模式：

- **`--config_name`**：从随机权重初始化新模型（指定模型架构配置）
- **`--model_name_or_path`**：在已有模型上继续预训练

## 数据处理说明

### 预训练（pretrain.py）

- 读取 JSONL 文件 → tokenize → 拼接成长文本 → 按 `block_size` 切块
- 所有 token 参与 loss 计算（标准语言模型）
- 支持多进程预处理（`--preprocessing_num_workers`）

### 指令微调（finetune.py）

- 读取对话 JSON（BelleGroup 格式）→ 构建 Qwen 格式 prompt 模板
- 使用 `IGNORE_TOKEN_ID` 屏蔽 user / system 部分的 loss
- **只计算 assistant 回答部分的 loss**
- 默认只取前 10,000 条数据

## 数据集格式

### 预训练数据格式（JSONL）

```jsonl
{"text": "今天天气真好，适合出去走走。"}
{"text": "在深度学习中，Transformer 是最重要的模型架构之一。"}
```

### 微调数据格式（JSONL）

```json
{"conversations": [
    {"from": "human", "value": "中国的首都是哪里？"},
    {"from": "assistant", "value": "中国的首都是北京。"}
]}
```

## 高级配置

### CUDA 环境（AutoDL / Linux）

如果使用 CUDA GPU，建议改用 DeepSpeed：

```bash
# 在 CUDA 环境下使用 deepspeed 启动（替换 finetune.sh 内容）
deepspeed finetune.py \
    --model_name_or_path autodl-tmp/qwen-1.5b \
    --train_files ./data/BelleGroup/train_3.5M_CN.json \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --output_dir autodl-tmp/output/sft \
    --eval_strategy no \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --warmup_steps 200 \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 2048 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero2.json \
    --report_to swanlab
```

> CUDA 环境不需要 `PYTORCH_MPS_HIGH_WATERMARK_RATIO` 和 `--fp16`，改用 `--bf16` 和 DeepSpeed。

### macOS 内存优化

当前 `finetune.sh` 已包含以下优化：
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` — 允许 MPS 使用更多内存
- `--fp16` — 半精度训练减少内存占用
- `--gradient_checkpointing` — 以计算换内存
- `batch_size=1` + `gradient_accumulation_steps=8` — 小 batch 大累积

## 训练结果

微调完成后，模型 checkpoints 保存在 `--output_dir` 指定的目录（默认 `autodl-tmp/output/sft/`）：

```
autodl-tmp/output/sft/
├── checkpoint-100/
│   ├── config.json
│   ├── model.safetensors
│   └── training_args.bin
├── checkpoint-200/
├── ...
└── final_model/          # trainer.save_model() 保存的最终模型
    ├── config.json
    ├── model.safetensors
    └── tokenizer_config.json
```

> 快速验证模式（`finetune_quick.sh`）已在 M1 Max 上验证通过，完整训练流程可正常运行。
> 完整 3 epoch 训练因 macOS 内存限制尚未跑完，建议在 CUDA GPU 环境（如 AutoDL）上运行全量训练。

## 实验跟踪

使用 SwanLab 记录训练指标（loss、学习率等），默认本地模式：

```bash
# 查看本地 SwanLab 日志
swanlab watch ./swanlog
```

如需同步云端，配置 API Key：

```python
swanlab.login(api_key="your_api_key")
```

## 参考

- [从零开始构建大模型 - Datawhale](https://github.com/datawhalechina/happy-llm)
- [Qwen2.5 官方文档](https://github.com/QwenLM/Qwen2.5)
- [Hugging Face Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [BelleGroup 对话数据集](https://huggingface.co/datasets/BelleGroup/train_3.5M_CN)
- [swanlab](https://github.com/SwanHubX/SwanLab)
