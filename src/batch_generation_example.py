    """
    This is example file for batch generation taken from https://huggingface.co/docs/transformers/main/en/continuous_batching
    """
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

SAMPLE_SIZE = 20

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    attn_implementation="sdpa_paged",
    device_map="cuda",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", padding_side="left")

dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
dataset = dataset.select(range(SAMPLE_SIZE))
tokenized_datasets = dataset.map(lambda x: tokenizer(x["question"]), batched=True)
simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]

generation_config = GenerationConfig(
    max_new_tokens=32,
    use_cuda_graph=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False,
    max_batch_tokens=512,
)

batch_outputs = model.generate_batch(
    inputs=simple_batch_inputs,
    generation_config=generation_config,
)

for request_id, output in batch_outputs.items():
    generated_text = tokenizer.decode(output.generated_tokens, skip_special_tokens=True)
    print(f"Request {request_id} output: {generated_text}")