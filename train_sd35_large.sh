PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT


accelerate launch --num_processes=8 --multi_gpu --mixed_precision bf16 --main_process_port $PORT main.py \
    --model_name "large" \
    --train_batch_size 4 \
    --learning_rate 5e-6 \
    --learning_rate_cls 5e-6 \
    --num_boundaries 4 \
    --scales 64 80 96 128 \
    --boundaries 0 7 14 18 28 \
    --do_dmd_loss \
    --do_gan_loss \
    --do_mmd_loss \
    --cfg_teacher 4.5 \
    --cls_blocks 20 \
    --mmd_blocks 20 \
    --max_train_steps 4000 \
    --apply_lora_to_attn_projections \
    --apply_lora_to_mlp_projections \
    --seed 42 \
    --gradient_checkpointing \
    --resume_from_checkpoint "latest"
