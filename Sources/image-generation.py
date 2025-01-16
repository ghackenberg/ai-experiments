from diffusers import AutoPipelineForText2Image

import torch

# pipeline configuration

#model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
model = "stabilityai/sdxl-turbo"
torch_dtype = torch.float16
variant = "fp16"

pipeline = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch_dtype, variant=variant).to("cuda")

# image generation loop

while True:

    # read user input

    print("\n-------------------\n")

    prompt = input('Prompt: ')

    # generate image

    print("\n-------------------\n")

    image = pipeline(prompt).images[0]

    # save image

    print("\n-------------------\n")

    filename = input("Filename: ")

    image.save(f"images/output-{filename}.jpg")