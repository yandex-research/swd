import numpy as np
import torch
from accelerate.logging import get_logger
from transformers import AutoModel, AutoProcessor

from src.evaluation.fid_score_in_memory import calculate_fid

logger = get_logger(__name__)


########################################################################################################################
#                                      METRICS: CS, PICKSCORE, FID                                                     #
########################################################################################################################


@torch.inference_mode()
def calc_pick_and_clip_scores(model, image_inputs, text_inputs, batch_size=8):
    assert len(image_inputs) == len(text_inputs["input_ids"])
    assert len(text_inputs.keys()) == 2

    scores = torch.zeros(len(image_inputs))
    for i in range(0, len(image_inputs), batch_size):
        image_batch = image_inputs[i : i + batch_size]
        text_batch = {
            "input_ids": text_inputs["input_ids"][i : i + batch_size],
            "attention_mask": text_inputs["attention_mask"][i : i + batch_size],
        }
        # embed
        with torch.amp.autocast("cuda"):
            image_embs = model.get_image_features(image_batch).pooler_output
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        with torch.amp.autocast("cuda"):
            text_embs = model.get_text_features(**text_batch).pooler_output
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores[i : i + batch_size] = (text_embs * image_embs).sum(-1)
    return scores.cpu()


@torch.no_grad()
def calculate_scores(args, images, prompts, ref_stats_path=None, device="cuda"):
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    clip_model = (
        AutoModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        .eval()
        .to(device)
    )
    pickscore_model = (
        AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
    )

    image_inputs = processor(
        images=images,
        return_tensors="pt",
    )[
        "pixel_values"
    ].to(device)

    text_inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    logger.info("Evaluating PickScore...")
    pick_score = calc_pick_and_clip_scores(
        pickscore_model, image_inputs, text_inputs
    ).mean()
    logger.info("Evaluating CLIP ViT-H-14 score...")
    clip_score = calc_pick_and_clip_scores(clip_model, image_inputs, text_inputs).mean()

    if ref_stats_path is not None:
        logger.info("Evaluating FID score...")
        fid_score = calculate_fid(
            images, ref_stats_path, inception_path=args.inception_path
        )

    return pick_score, clip_score, fid_score
