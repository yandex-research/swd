import argparse
import os

from distill_sd3_scalewise import train

# Parse arguments
# ----------------------------------------------------------------------------------------------------------------------
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--size_interpolation",
        type=str,
        default="nearest",
        choices=["nearest", "bilinear", "bicubic", "fft"],
        help="Interpolation mode for size switch.",
    )
    parser.add_argument(
        "--train_dataloader_config_path",
        type=str,
        default=None,
        help="Path to train_dataloader yaml config.",
    )
    parser.add_argument(
        "--clip_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pickscore_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--coco_ref_stats_path",
        type=str,
        default="stats/fid_stats_mscoco256_val.npz",
    )
    parser.add_argument(
        "--mjhq_ref_stats_path",
        type=str,
        default="stats/fid_stats_mjhq256_val.npz",
    )
    parser.add_argument(
        "--inception_path",
        type=str,
        default="stats/pt_inception-2015-12-05-6726825d.pth",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run fine-tuning validation every X steps. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=5000,
        help="Number of samples for metric calculation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=2, help="Batch size (per device) for the evaluation."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_cls",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None
    )
    parser.add_argument(
        "--apply_lora_to_attn_projections",
        action="store_true",
        help=("Whether to apply LoRA to attention projections in attention."),
    )
    parser.add_argument(
        "--apply_lora_to_add_projections",
        action="store_true",
        help=("Whether to apply LoRA to add projections in attention."),
    )
    parser.add_argument(
        "--apply_lora_to_mlp_projections",
        action="store_true",
        help=("Whether to apply LoRA to mlp projections."),
    )
    parser.add_argument(
        "--apply_lora_to_ada_norm_projections",
        action="store_true",
        help=("Whether to apply LoRA to AdaLN projections."),
    )
    parser.add_argument(
        "--apply_lora_to_timestep_projections",
        action="store_true",
        help=("Whether to apply LoRA to timesteps projections."),
    )
    parser.add_argument(
        "--apply_lora_to_proj_out",
        action="store_true",
        help=("Whether to apply LoRA to proj_out."),
    )
    parser.add_argument(
        "--apply_lora_to_embedders",
        action="store_true",
        help=("Whether to apply LoRA to embedders."),
    )
    parser.add_argument(
        "--offload_text_encoders",
        action="store_true",
        help="Whether to offload text encoders on training.",
    ) 
    parser.add_argument(
        "--num_boundaries",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num_discriminator_upds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_discriminator_layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument(
        "--previous_best_ps",
        type=float,
        default=0.218,
    )
    parser.add_argument(
        "--cfg_teacher",
        type=float,
        default=4.5,
    )
    parser.add_argument(
        "--cfg_fake",
        type=float,
        default=4.5,
    )
    parser.add_argument(
        "--gen_cls_loss_weight",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--guidance_cls_loss_weight",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--do_pixels_downscale",
        action="store_true",
    )
    parser.add_argument(
        "--do_gan_loss",
        action="store_true",
    )
    parser.add_argument(
        "--do_pdm_loss",
        action="store_true",
    )
    parser.add_argument(
        "--do_pdm_v2",
        action="store_true",
    )
    parser.add_argument(
        "--pdm_sigma",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--pdm_kernel",
        type=str,
        default='linear',
    )
    parser.add_argument(
        "--do_dmd_loss",
        action="store_true",
    )
    parser.add_argument(
        "--stochastic_case",
        action="store_true",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--boundaries",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cls_blocks",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pdm_blocks",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--n_steps_fake_dmd",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--pdm_noise_start_idx",
        type=int,
        default=18,
    )
    parser.add_argument(
        "--pdm_noise_end_idx",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--pdm_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--scheduler_shift",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--pdm_real",
        action="store_true",
    )
    parser.add_argument(
        "--fake_diffusion_flow_pred",
        action="store_true",
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.scales:
        args.scales = [int(x) for x in args.scales.split(",")]

    if args.boundaries:
        args.boundaries = [int(x) for x in args.boundaries.split(",")]

    if args.cls_blocks:
        args.cls_blocks = [int(x) for x in args.cls_blocks.split(",")]
    
    if args.pdm_blocks:
        args.pdm_blocks = [int(x) for x in args.pdm_blocks.split(",")]
    else:
        args.pdm_blocks = args.cls_blocks

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.train_dataloader_config_path is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args
# ----------------------------------------------------------------------------------------------------------------------

# Input spot
if __name__ == "__main__":
    args = parse_args()
    train(args)
