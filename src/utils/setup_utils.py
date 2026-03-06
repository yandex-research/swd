import copy
import os
import random

import numpy as np
import torch
from accelerate.logging import get_logger

logger = get_logger(__name__)


def set_tf32(tf32=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if tf32 else "highest")


def seed_everything(seed, rank, world_size):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed = seed * world_size + rank
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------------------------------
def load_if_exist(args, accelerator, transformer, is_student):
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if is_student:
                transformer.load_adapter(
                    os.path.join(path, "model"),
                    adapter_name="resume",
                    is_trainable=True,
                )
                transformer.set_adapter("resume")
            else:
                transformer.teacher_transformer.load_adapter(
                    os.path.join(path, "fake/lora"),
                    adapter_name="resume",
                    is_trainable=True,
                )
                transformer.set_adapter("resume")
                transformer.cls_pred_branch.load_state_dict(
                    torch.load(os.path.join(path, "fake", "cls_head.pt"))
                )
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
    else:
        initial_global_step = 0

    return initial_global_step


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_3rd_party(args, accelerator):
    # Scheduler and math around the number of training steps.

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        # Tracker config doesn't support lists
        tracker_config.pop("scales")
        tracker_config.pop("cls_blocks")
        tracker_config.pop("mmd_blocks")
        tracker_config.pop("boundaries")
        accelerator.init_trackers("scalewise", config=tracker_config)


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def saving(transformer, args, accelerator, global_step, is_student):
    if is_student:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}/model")
    else:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}/fake")
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving to {save_path}")
    if is_student:
        transformer = accelerator.unwrap_model(transformer)
        transformer.save_pretrained(save_path)
    else:
        transformer = accelerator.unwrap_model(transformer)
        transformer.teacher_transformer.save_pretrained(save_path + "/lora")
        torch.save(
            transformer.cls_pred_branch.state_dict(),
            os.path.join(save_path, "cls_head.pt"),
        )
    logger.info(f"Saved state to {save_path}")


# ----------------------------------------------------------------------------------------------------------------------
