from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./merged-model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "### Question: Explain quantum computing in simple terms.\n### Answer:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=120,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
