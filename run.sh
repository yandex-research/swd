ACCELERATE_CONFIG="configs/default_config.yaml"
PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT

MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
DATASET_PATH=<YOUR PATH>


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 --multi_gpu --mixed_precision bf16 --main_process_port $PORT main.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_dataloader_config_path=$DATASET_PATH \
    --train_batch_size=2 \
    --gradient_checkpointing \
    --checkpointing_steps=5000 \
    --learning_rate=5e-6 \
    --num_boundaries=4 \
    --scales="64,80,96,128" \
    --boundaries="0,7,14,18,28" \
    --do_pixels_downscale \
    --stochastic_case \
    --num_discriminator_upds=3 \
    --num_discriminator_layers=4 \
    --seed=42 \
    --output_dir="results" \
    --lora_rank=64 \
    --cls_blocks=11 \
    --pdm_blocks=11 \
    --cfg_teacher=4.5 \
    --cfg_fake=1.0 \
    --apply_lora_to_attn_projections \
    --apply_lora_to_mlp_projections \
    --apply_lora_to_ada_norm_projections \
    --apply_lora_to_embedders \
    --apply_lora_to_proj_out \
    --validation_steps=5 \
    --evaluation_steps=10 \
    --coco_ref_stats_path stats/fid_stats_mscoco256_val.npz \
    --inception_path stats/pt_inception-2015-12-05-6726825d.pth \
    --max_train_steps=1000 \
    --resume_from_checkpoint=latest \
    --max_eval_samples=100 \
    --pickscore_model_name_or_path yuvalkirstain/PickScore_v1 \
    --clip_model_name_or_path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
    --do_dmd \
    --do_gan_loss \
    --do_pdm_loss \
    --pdm_kernel='linear' \
    --pdm_sigma=100 \
    --scheduler_shift=3 \
    --pdm_real \
    --fake_diffusion_flow_pred
