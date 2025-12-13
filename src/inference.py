import argparse
import re
import time
import warnings
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration and Data ---

# NOTE: Since you provided two different versions of the data with slightly different prompts, 
# we use the version that includes the 'Let's think step by step' trigger.
EVAL_DATA = [
    {"n_ops": 4,  "prompt": "711 + 234 + 666 + 474 =", "target": "2085"},
    {"n_ops": 8,  "prompt": "733 + 588 + 375 + 582 + 46 + 350 + 762 + 989 =", "target": "4425"},
    {"n_ops": 32, "prompt": "30 + 900 + 881 + 874 + 875 + 55 + 518 + 311 + 937 + 981 + 584 + 756 + 657 + 455 + 162 + 505 + 266 + 933 + 295 + 784 + 979 + 913 + 337 + 430 + 444 + 760 + 503 + 577 + 750 + 694 + 465 + 365 =", "target": "18976"},
    {"n_ops": 16, "prompt": "909 + 872 + 91 + 696 + 853 + 752 + 130 + 39 + 446 + 672 + 612 + 301 + 588 + 779 + 314 + 38 =", "target": "8092"},
    {"n_ops": 16, "prompt": "636 + 186 + 977 + 145 + 133 + 112 + 243 + 289 + 563 + 554 + 90 + 823 + 581 + 459 + 855 + 698 =", "target": "7344"},
]


# --- Model Loading Function ---

def load_model(model_path: str, ut_steps: int, float_precision: torch.dtype) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads the Ouro model and tokenizer with specified UT steps."""
    
    # Load config and set UT steps
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.total_ut_steps = ut_steps
    
    # -1.0 ensures the model always uses the full UT steps for stable evaluation
    config.early_exit_threshold = -1.0 
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model in {float_precision}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        torch_dtype=float_precision,
        trust_remote_code=True
    )
    model.eval()

    print(f"Model loaded! Using device: {model.device}")
    print(f"UT steps: {model.config.total_ut_steps}")
    
    return model, tokenizer


# --- Inference Function ---

@torch.no_grad()
def predict_with_reasoning(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_tokens: int) -> Tuple[str, str]:
    """Runs inference and extracts reasoning and the final answer."""
    
    # Use a prompt template that encourages CoT and a clean final answer format
    prompt_template = (
        "{prompt} Let's think step by step to find the total sum. "
        "The final answer is strictly the number after 'The total sum is:'."
    )
    full_prompt = prompt_template.format(prompt=prompt)
    
    messages = [{"role": "user", "content": full_prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False  # Required for Ouro (Looped Transformer)
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 1. Isolate the Assistant's full response (which includes thinking)
    full_response = response.split("<|assistant|>")[-1].strip()
    
    # 2. Try to extract the final number using the clean format (best practice)
    final_match = re.search(r"The total sum is:\s*(\d+)", full_response)
    
    if final_match:
        predicted_number = final_match.group(1)
    else:
        # Fallback extraction: find the last number in the full thought process
        numbers = re.findall(r'\d+', full_response)
        predicted_number = numbers[-1] if numbers else "0"

    return full_response, predicted_number


# --- Main Evaluation Function ---

def run_evaluation(args: argparse.Namespace):
    """Main function to run the model evaluation."""
    
    # 1. Load Model
    precision = torch.bfloat16 if args.bf16 else torch.float16
    model, tokenizer = load_model(args.model_path, args.ut_steps, precision)

    # 2. Run Evaluation
    print("\n" + "─" * 70)
    print(f"Evaluating {args.model_path} ({args.ut_steps} UT steps, {precision})")
    print("─" * 70)
    
    correct = 0
    total_start = time.time()

    for i, item in enumerate(EVAL_DATA, 1):
        start = time.time()
        
        # Call the inference function
        full_thought, pred = predict_with_reasoning(item["prompt"], model, tokenizer, args.max_tokens)
        
        elapsed = time.time() - start
        status = "CORRECT" if pred == item["target"] else "WRONG"
        if status == "CORRECT":
            correct += 1
            
        print(f"**{i}. [{item['n_ops']:2} ops]** | Target: {item['target']} | Pred: {pred} | Status: {status} | Time: {elapsed:.2f}s")
        
        # Optional: Print Reasoning Thoughts
        if args.print_reasoning:
            print("\n   🧠 **REASONING THOUGHTS:**")
            # Limit the output of the full thought to avoid extremely long prints
            print(f"   {full_thought[:500]}..." if len(full_thought) > 500 else f"   {full_thought}")
            print("\n" + "—" * 30 + "\n")

    # 3. Print Final Metrics
    total_time = time.time() - total_start
    print("─" * 70)
    print(f"Accuracy: {correct}/{len(EVAL_DATA)} = {100*correct/len(EVAL_DATA):.1f}%")
    print(f"Total time: {total_time:.2f}s → Avg: {total_time/len(EVAL_DATA):.2f}s per prompt")
    print("All done!")


# --- Command Line Argument Setup ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the Ouro Looped Transformer with configurable parameters.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required Arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="ByteDance/Ouro-1.4B-Thinking",
        help="Path to the pre-trained Ouro model directory (e.g., /kaggle/working/ouro_model_thinking)"
    )

    # Configurable Model Parameters
    parser.add_argument(
        "--ut_steps",
        type=int,
        default=4,
        help="Number of Recurrent Steps (Universal Transformer steps) to use. Default is 4 (R4)."
    )
    
    # Generation/Evaluation Parameters
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate for the thought and final answer. Use a high number (e.g., 512-1024) for long reasoning."
    )
    
    # Output Parameters
    parser.add_argument(
        "--print_reasoning",
        action="store_true",
        help="If set, the full thought process will be printed for each evaluation item."
    )
    
    # Performance Parameters
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use torch.bfloat16 precision instead of torch.float16 for potential stability on supported GPUs."
    )

    args = parser.parse_args()
    run_evaluation(args)