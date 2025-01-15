import torch

from transformers import pipeline

# pipeline configuration

task = "text-generation"
model = "meta-llama/Llama-3.2-1B-Instruct"
torch_dtype = torch.bfloat16
device_map = "auto"

pipe = pipeline(task, model=model, torch_dtype=torch_dtype, device_map=device_map)

# chat loop

messages = []

while True:

    # read user input

    print("\n-------------------\n")

    content = input("You: ")

    print("\n-------------------\n")

    messages.append({"role": "user", "content": content})

    # generate bot response

    outputs = pipe(messages, max_new_tokens=4096, pad_token_id=pipe.tokenizer.eos_token_id)

    message = outputs[0]["generated_text"][-1]

    print(f"Bot: {message["content"]}")

    messages.append(message)