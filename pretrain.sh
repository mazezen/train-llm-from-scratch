export TENSORBOARD_LOGGING_DIR=autodl-tmp/output/pretrain/logs
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python pretrain.py \
    --config_name autodl-tmp/qwen-1.5b \
    --tokenizer_name autodl-tmp/qwen-1.5b \
    --train_files ./autodl-tmp/dataset/pretrain_data/mobvoi_seq_monkey_general_open_corpus.jsonl \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --output_dir autodl-tmp/output/pretrain \
    --eval_strategy  no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --preprocessing_num_workers 10 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 2048 \
    --gradient_checkpointing \
    --fp16 \
    --report_to swanlab
    # Note: removed --deepspeed (CUDA only), adapted for macOS.

    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \
