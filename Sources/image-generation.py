from diffusers import AutoPipelineForText2Image

import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

while True:

    print("\n-------------------\n")

    prompt = input('Prompt: ')

    print("\n-------------------\n")

    image = pipeline(prompt).images[0]

    print("\n-------------------\n")

    filename = input("Filename: ")

    image.save(f"images/output-{filename}.jpg")