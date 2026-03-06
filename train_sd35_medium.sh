PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT


accelerate launch --num_processes=8 --multi_gpu --mixed_precision bf16 --main_process_port $PORT main.py \
    --model_name "medium" \
    --train_batch_size 8 \
    --learning_rate 5e-6 \
    --learning_rate_cls 5e-6 \
    --num_boundaries 4 \
    --scales 64 80 96 128 \
    --boundaries 0 7 14 18 28 \
    --do_dmd_loss \
    --do_gan_loss \
    --do_mmd_loss \
    --cfg_teacher 7.0 \
    --cls_blocks 11 \
    --mmd_blocks 11 \
    --max_train_steps 3000 \
    --apply_lora_to_attn_projections \
    --apply_lora_to_mlp_projections \
    --seed 42 \
    --gradient_checkpointing \
    --resume_from_checkpoint "latest"
