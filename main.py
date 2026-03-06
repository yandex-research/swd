import argparse
import os

from train import train


# Parse arguments
# ----------------------------------------------------------------------------------------------------------------------
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["medium", "large"],
        default="medium",
        required=True,
        help="SD3.5 model size",
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

    # Training
    parser.add_argument(
        "--seed", type=int, default=0, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
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
        help="Initial learning rate for fake model (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=10.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=25,
        help="Interval (in training steps) between loss logging.",
    )

    # LoRA
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--lora_alpha", type=int, default=None)
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
        "--cfg_teacher",
        type=float,
        default=4.5,
    )
    parser.add_argument(
        "--cfg_fake",
        type=float,
        default=1.0,
    )
    # Scale-wise
    parser.add_argument(
        "--num_boundaries",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=28,
    )
    parser.add_argument(
        "--boundaries",
        nargs="+",
        default=[],
        type=int,
        help="Interval boundaries (diffusion timesteps) based on `--num_timesteps`",
    )
    parser.add_argument("--scales", nargs="+", default=[], type=int)
    parser.add_argument(
        "--scheduler_shift",
        type=int,
        default=3,
    )
    # DMD loss
    parser.add_argument(
        "--do_dmd_loss",
        action="store_true",
    )
    parser.add_argument(
        "--n_steps_fake_dmd",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--dmd_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--dmd_noise_start_idx",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--dmd_noise_end_idx",
        type=int,
        default=28,
    )
    # GAN loss
    parser.add_argument(
        "--do_gan_loss",
        action="store_true",
    )
    parser.add_argument(
        "--num_discriminator_layers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--cls_blocks",
        nargs="+",
        default=[],
        type=int,
        help="Block index(es) from which features are taken for the GAN classification loss.",
    )
    parser.add_argument(
        "--gen_cls_loss_weight",
        type=float,
        default=5e-3,
        help="Weight of the GAN classification loss in the generator.",
    )
    parser.add_argument(
        "--disc_cls_loss_weight",
        type=float,
        default=1e-2,
        help="Weight of the GAN classification loss in the discriminator.",
    )
    # MMD loss
    parser.add_argument(
        "--do_mmd_loss",
        action="store_true",
    )
    parser.add_argument(
        "--mmd_blocks",
        nargs="+",
        default=[],
        type=int,
        help="Block index(es) from which features are taken for the MMD loss.",
    )
    parser.add_argument(
        "--mmd_rbf_sigma",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--mmd_kernel",
        type=str,
        default="linear",
    )
    parser.add_argument(
        "--mmd_noise_start_idx",
        type=int,
        default=18,
    )
    parser.add_argument(
        "--mmd_noise_end_idx",
        type=int,
        default=28,
    )
    parser.add_argument(
        "--mmd_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--do_batch_mmd",
        action="store_true",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter.",
    )

    # Checkpointing
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
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
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    # Sampling
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=500,
        help="Interval (in training steps) between metric calculations.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=250,
        help="Interval (in training steps) between validation image generations.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--coco_ref_stats_path",
        type=str,
        default="src/stats/fid_stats_mscoco256_val.npz",
    )
    parser.add_argument(
        "--mjhq_ref_stats_path",
        type=str,
        default="src/stats/fid_stats_mjhq256_val.npz",
    )
    parser.add_argument(
        "--inception_path",
        type=str,
        default="src/stats/pt_inception-2015-12-05-6726825d.pth",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=5008,
        help="Number of samples for metric calculation",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation.",
    )

    # Distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if len(args.mmd_blocks) == 0:
        args.mmd_blocks = args.cls_blocks

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
