from diffusers import (
    DDIMPipeline,
)  # , DDIMScheduler, DDIMInverseScheduler, DDIMInversion
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from torch import autocast, inference_mode


model_id = "google/ddpm-celebahq-256"
# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id)

image = ddim(num_inference_steps=200)

save_image(
    image,
    f"sampled.png",
    nrow=int(np.sqrt(image.shape[0])),
    normalize=True,
    value_range=(-1, 1),
)

print(ddim.scheduler.timesteps)
# print(ddim.scheduler.timesteps.flip(0))
print(image.min(), image.max(), image.mean(), image.std())
# ddim.timesteps = DDIMScheduler
with autocast("cuda"), inference_mode():
    for i, e in enumerate(tqdm(ddim.scheduler.timesteps[1:-1])):
        image = ddim.scheduler.reverse_step(
            ddim.unet(image, e).sample, e, image
        ).next_sample
        print("e", e)
        print(image.min(), image.max(), image.mean(), image.std())
save_image(
    image,
    f"noise.png",
    nrow=int(np.sqrt(image.shape[0])),
    normalize=True,
    value_range=(-1, 1),
)
image_ = image.clone()
print(ddim.scheduler.timesteps)
with autocast("cuda"), inference_mode():
    for i, e in enumerate(tqdm(ddim.scheduler.timesteps)):
        image_ = ddim.scheduler.step(ddim.unet(image_, e).sample, e, image_).prev_sample
        print(e)
        print(image_.min(), image_.max(), image_.mean(), image_.std())
save_image(
    image_,
    f"recovered.png",
    nrow=int(np.sqrt(image.shape[0])),
    normalize=True,
    value_range=(-1, 1),
)
