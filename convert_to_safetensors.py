from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
import torch

models = [
    "models/tinyllama_chat",
    "models/tinyllama_base"
]

for path in models:
    print("Loading:", path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)

    tensors = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    save_file(tensors, f"{path}/model.safetensors")
    print("Converted:", path)
