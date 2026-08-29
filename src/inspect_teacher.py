import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "roneneldan/TinyStories-8M"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

model.eval()

print(model)
print("Parameters:", sum(p.numel() for p in model.parameters()))

prompt = "Once upon a time there was a little"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))