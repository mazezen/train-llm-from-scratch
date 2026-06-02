export TENSORBOARD_LOGGING_DIR=autodl-tmp/output/sft/logs
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python finetune.py \
    --model_name_or_path autodl-tmp/qwen-1.5b \
    --train_files ./data/BelleGroup/train_3.5M_CN.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --output_dir autodl-tmp/output/sft \
    --eval_strategy  no \
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
    --gradient_checkpointing \
    --fp16 \
    --report_to swanlab
    # Note: removed --deepspeed (CUDA only). Using gradient_checkpointing + fp16 + batch_size=1
    # to fit 1.5B model on M1 Max 32GB unified memory.
    
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \
