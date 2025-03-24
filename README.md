# Scale-wise Distillation of Diffusion Models

<a href='https://arxiv.org/pdf/2503.16397'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 
<a href='https://yandex-research.github.io/swd/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; 
<a href="https://huggingface.co/spaces/dbaranchuk/Scale-wise-Distillation">
	    <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Generation-orange'></a>
<a href="https://gist.github.com/dbaranchuk/c29ea632ae7563299b5131bb5d1e24e6">
	    <img src='https://img.shields.io/badge/ComfyUI-SwD Large-blue'></a> &nbsp;

<p  align="center">
⚡️ <b> SwD is twice as fast as leading distillation methods </b> <br> 
⚡️ <b> SwD surpasses leading distillation methods within the same computational budget </b>
<p >

<p align="center">
<img src="assets/main.jpg" width="1080px"/>
</p>

## 🔧 TODO
- [ ] Training code
- [ ] FLUX
- [x] ComfyUI support
- [x] Inference with SD3.5

## 💡 Quick introduction
The paper introduces Scale-wise Distillation (SwD), a novel framework for accelerating 
diffusion models (DMs) by progressively increasing spatial resolution during the generation process. 
SwD achieves significant speedups (2.5× to 10×) compared to full-resolution models while maintaining or even improving image quality.


<p align="center">
<img src="assets/swd.png" width="1080px"/>
</p>

The human evaluation shows that SwD is highly competitive and often outperforms the baselines.
SwD generates images with higher complexity compared to leading approaches.

<p align="center">
<img src="assets/sbs.png" width="1080px"/>
</p>


## 🔥 Inference

### HF 🤗 Models
We release two versions of SwD, Medium (2B) and Large (8B), distilled from SD3.5. <br>
SwD requires two key hyperparameters: <b> scales </b> and <b> sigmas </b>. 
- The scales hyperparameter defines the spatial resolution at which predictions are performed during the generation
process. It specifies the sequence of resolutions (e.g., starting from a lower resolution like 256×256 
and progressively increasing to the target resolution, such as 1024×1024).
- The sigmas hyperparameter controls the noise levels applied at each step of the diffusion process.
It is equivalent to diffusion timesteps.

| Model                                                                  | Scales                                                 | Sigmas                                      
|:-----------------------------------------------------------------------|:-------------------------------------------------------|:--------------------------------------------|
| [SwD 2B, 6 steps](https://huggingface.co/yresearch/swd-medium-6-steps) | 32, 48, 64, 80, 96, 128                                | 1.0000, 0.9454, 0.8959, 0.7904, 0.7371, 0.6022, 0.0000 |
| [SwD 2B, 4 steps](https://huggingface.co/yresearch/swd-medium-4-steps) | 32, 64, 96, 128                                        | 1.0000, 0.9454, 0.7904, 0.6022, 0.0000                                              |
| [SwD 8B, 6 steps](https://huggingface.co/yresearch/swd-large-6-steps)  | 32, 48, 64, 80, 96, 128                                | 1.0000, 0.9454, 0.8959, 0.7904, 0.7371, 0.6022, 0.0000 |
| [SwD 8B, 4 steps](https://huggingface.co/yresearch/swd-large-4-steps)  | 64, 80, 96, 128                                        | 1.0000, 0.8959, 0.7371, 0.6022, 0.0000              |


Upgrade to the latest version of the [🧨 diffusers](https://github.com/huggingface/diffusers)
and [🧨 peft](https://github.com/huggingface/peft)
```
pip install -U diffusers
pip install -U peft
```

and then you can run
<br> (Probably, you will need to specify the visible device: %env CUDA_VISIBLE_DEVICES=0, for correct loading of LoRAs.)
```py
import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large",
                                                torch_dtype=torch.float16,
                                                custom_pipeline='quickjkee/swd_pipeline')
pipe = pipe.to("cuda")
lora_path = 'yresearch/swd-large-6-steps'
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer,
    lora_path,
).to("cuda")

generator = torch.Generator().manual_seed(0)
prompt = 'a cat reading a newspaper'
sigmas = [1.0000, 0.9454, 0.8959, 0.7904, 0.7371, 0.6022, 0.0000]
scales = [32, 48, 64, 80, 96, 128]

images = pipe(
    prompt,
    sigmas=torch.tensor(sigmas).to('cuda'),
    timesteps=torch.tensor(sigmas[:-1]).to('cuda') * 1000,
    scales=scales,
    guidance_scale=0.0,
    height=int(scales[0] * 8),
    width=int(scales[0] * 8),
    generator=generator,
).images
```
<p align="center">
<img src="assets/cat.jpg" width="512px"/>
</p>

## 🔧 Training

Coming soon!

## Citation

```bibtex
@article{starodubcev2025swd,
  title={Scale-wise Distillation of Diffusion Models},
  author={Nikita Starodubcev and Denis Kuznedelev and Artem Babenko and Dmitry Baranchuk},
  journal={arXiv preprint arXiv:2503.16397},
  year={2025}
}
```
