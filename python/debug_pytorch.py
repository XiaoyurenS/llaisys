from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)

inputs = torch.tensor([[1,2,3]], dtype=torch.long)
with torch.no_grad():
    out = model(input_ids=inputs).logits  # [1, T, vocab]
logits = out[0, -1, :8]
print(logits)