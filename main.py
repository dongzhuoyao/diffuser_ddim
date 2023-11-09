from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DDIMInverseScheduler,
    StableDiffusionPix2PixZeroPipeline,
)
import torch

# load models
# model_id = "stabilityai/stable-diffusion-2-1"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pix2pix = StableDiffusionPix2PixZeroPipeline(
    pipe.vae,
    pipe.text_encoder,
    pipe.tokenizer,
    pipe.unet,
    pipe.scheduler,
    pipe.feature_extractor,
    None,
    DDIMInverseScheduler.from_config(pipe.scheduler.config),
    None,
    None,
    False,
)
pipe.to("cuda")
pix2pix.to("cuda")

# some random input
generator = torch.manual_seed(123456)
randomNoise = torch.randn((1, 4, 64, 64))

# generate a image from the given random input
someimg = pipe(
    "A cat sitting on a table",
    latents=randomNoise,
    generator=generator,
    num_inference_steps=50,
).images[0]
someimg.save("someimg.png")
