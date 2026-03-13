from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
import torch

path = "models/tinyllama_base"

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

save_file(state_dict, f"{path}/model.safetensors")

print("Safetensors created.")
