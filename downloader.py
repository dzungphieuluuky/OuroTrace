import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Download model to a local folder (bypasses folder name issues)
print("⬇️ Downloading model to local folder 'ouro_model'...")
local_model_path = snapshot_download(
    repo_id="ByteDance/Ouro-1.4B",
    local_dir="ouro_model",
    local_dir_use_symlinks=False,  # Ensures actual files are downloaded
)
print(f"✅ Downloaded to: {local_model_path}")

# 2. Load Tokenizer from local folder
print("⏳ Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 3. Load Model from local folder
print("⏳ Loading Model...")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",
    torch_dtype="auto",  # Auto-selects optimal dtype (e.g., float16 on GPU)
)
print("✅ Model loaded successfully!")

# 4. Run Experiment Loop (with use_cache=False to avoid cache setter error)
inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,  # Slight variation; set False for deterministic
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=False,  # Disables KV caching to prevent property setter issue; loops still run
)
print("\nGenerated Output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
