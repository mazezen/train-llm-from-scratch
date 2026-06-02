# 快速验证模式 — 只跑 3 步，验证预训练流程是否正常
# 运行时间约 1-2 分钟（M1 Max）

export TENSORBOARD_LOGGING_DIR=autodl-tmp/output/pretrain_quick/logs
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python pretrain.py \
    --config_name autodl-tmp/qwen-1.5b \
    --tokenizer_name autodl-tmp/qwen-1.5b \
    --train_files ./data/pretrain_test.jsonl \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --output_dir autodl-tmp/output/pretrain_quick \
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
    --preprocessing_num_workers 1 \
    --gradient_checkpointing \
    --fp16 \
    --report_to none
    # --max_steps 3: 只跑 3 步，约 1-2 分钟
    # --gradient_accumulation_steps 1: 不累积，更快出结果
    # --block_size 256: 短序列，省显存
    # --train_files ./data/pretrain_test.jsonl: 10 条测试数据
