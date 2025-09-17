import torch
import numpy as np

from src.utils.fid_score_in_memory import calculate_fid
from accelerate.logging import get_logger
from transformers import AutoProcessor, AutoModel
#import ImageReward as RM

logger = get_logger(__name__)


########################################################################################################################
#                                      METRICS: CS, PICKSCORE, FID                                                     #
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
@torch.inference_mode()
def calc_pick_and_clip_scores(model, image_inputs, text_inputs, batch_size=8):
    assert len(image_inputs) == len(text_inputs["input_ids"])
    assert len(text_inputs.keys()) == 2

    scores = torch.zeros(len(image_inputs))
    for i in range(0, len(image_inputs), batch_size):
        image_batch = image_inputs[i : i + batch_size]
        text_batch = {
            "input_ids": text_inputs["input_ids"][i : i + batch_size],
            "attention_mask": text_inputs["attention_mask"][i : i + batch_size]
        }
        # embed
        with torch.amp.autocast("cuda"):
            image_embs = model.get_image_features(image_batch)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        with torch.amp.autocast("cuda"):
            text_embs = model.get_text_features(**text_batch)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores[i:i + batch_size] = (text_embs * image_embs).sum(-1)
    return scores.cpu()
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def calculate_scores(
    args,
    images,
    prompts,
    ref_stats_path=None,
    device='cuda'
):
    processor = AutoProcessor.from_pretrained(args.clip_model_name_or_path)
    clip_model = AutoModel.from_pretrained(args.clip_model_name_or_path).eval().to(device)
    pickscore_model = AutoModel.from_pretrained(args.pickscore_model_name_or_path).eval().to(device)
    #imagereward_model = RM.load("ImageReward-v1.0").eval().to(device)

    image_inputs = processor(
        images=images,
        return_tensors="pt",
    )['pixel_values'].to(device)

    text_inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    logger.info("Evaluation ImageReward...")
    image_reward = torch.zeros(1)
    #for prompt, image in zip(prompts, images):
    #    image_reward += imagereward_model.score(prompt, [image])
    #image_reward /= len(prompts)

    logger.info("Evaluating PickScore...")
    pick_score = calc_pick_and_clip_scores(pickscore_model, image_inputs, text_inputs).mean()
    logger.info("Evaluating CLIP ViT-H-14 score...")
    clip_score = calc_pick_and_clip_scores(clip_model, image_inputs, text_inputs).mean()
    
    if ref_stats_path is not None:
        logger.info("Evaluating FID score...")
        fid_score = calculate_fid(images, ref_stats_path, inception_path=args.inception_path)

    return image_reward, pick_score, clip_score, fid_score
# ----------------------------------------------------------------------------------------------------------------------


@torch.inference_mode()
def calculate_image_reward_score(
    images, prompts, device='cuda',
    batch_size=8, image_reward_path="ImageReward-v1.0",
):
    model = ImageReward.load(image_reward_path, device=device).eval()

    scores = []
    for i in range(0, len(prompts), batch_size):
        # text encode
        with torch.cuda.amp.autocast():
            text_input = model.blip.tokenizer(prompts[i: i + batch_size],
                                              padding='max_length', truncation=True,
                                              max_length=35, return_tensors="pt",
                                              ).to(device)

            processed_images = torch.stack([model.preprocess(image).to(device)
                                            for image in images[i: i + batch_size]])
            image_embeds = model.blip.visual_encoder(processed_images)

            # text encode cross attention with image
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            text_output = model.blip.text_encoder(text_input.input_ids,
                                                attention_mask=text_input.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True,
                                                )

        txt_features = text_output.last_hidden_state[:, 0].float() # (feature_dim)
        rewards = model.mlp(txt_features)
        rewards = (rewards - model.mean) / model.std

        scores.extend(rewards[:, 0].tolist())
    
    return np.mean(scores)
# ----------------------------------------------------------------------------------------------------------------------
