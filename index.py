!pip install diffusers transformers accelerate torch safetensors
!pip install diffusers==0.30.2 transformers==4.44.2 accelerate==0.34.2 safetensors==0.4.5 torch --upgrade
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from IPython.display import display

model_id = "runwayml/stable-diffusion-v1-5"

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

prompt = "moon"

image = pipe(prompt).images[0]

display(image)
image.save("generated_image.png")
print("✅ Image saved as generated_image.png")
