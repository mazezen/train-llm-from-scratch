# 快速验证模式 — 只跑 3 步，验证训练流程是否正常
# 运行时间约 1-2 分钟（M1 Max）

export TENSORBOARD_LOGGING_DIR=autodl-tmp/output/sft_quick/logs
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python finetune.py \
    --model_name_or_path autodl-tmp/qwen-1.5b \
    --train_files ./data/BelleGroup/train_3.5M_CN.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --output_dir autodl-tmp/output/sft_quick \
    --eval_strategy no \
    --learning_rate 1e-4 \
    --max_steps 3 \
    --warmup_steps 1 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy no \
    --seed 12 \
    --block_size 256 \
    --dataloader_pin_memory False \
    --gradient_checkpointing \
    --fp16 \
    --report_to none
    # --max_steps 3: 只跑 3 步，约 1-2 分钟
    # --gradient_accumulation_steps 1: 不累积，更快出结果
    # --block_size 256: 短序列，省显存
    # --dataloader_pin_memory False: MPS 不支持 pin_memory，关掉避免警告
