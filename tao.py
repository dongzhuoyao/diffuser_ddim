from diffusers import DDIMPipeline
import PIL.Image
import numpy as np

# load model and scheduler
pipe = DDIMPipeline.from_pretrained("fusing/ddim-lsun-bedroom")

# run pipeline in inference (sample random noise and denoise)
image = pipe(eta=0.0, num_inference_steps=50)

# process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# save image
image_pil.save("test.png")
