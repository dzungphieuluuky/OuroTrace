import pandas as pd
import re
import torch
import random
from tqdm.auto import tqdm
from typing import List, Dict
from .data_generator import create_reasoning_primitives_data, format_5_shot_prompt

def load_and_process_results(file_path: str):
    """Load results CSV and add derived features"""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame()

    # Feature engineering
    if "generated_tokens" not in df.columns:
        df["generated_tokens"] = df["full_response"].apply(
            lambda x: len(re.findall(r"\S+", str(x))) if pd.notna(x) else 0
        )

    # Type conversion
    df["is_correct"] = df["is_correct"].astype(bool)
    df["ut_steps"] = pd.to_numeric(df["ut_steps"], errors="coerce").astype("Int64")

    return df

def run_holistic_evaluation(model, tokenizer, config: dict):
    """
    Runs the full evaluation suite:
    1. Custom Reasoning Primitives (Depth-k Var Assign) - Running locally
    2. Standard Benchmarks (QA/Math) - via lm-evaluation-harness
    
    Returns:
        List[Dict]: Results for each evaluation instance
    """
    holistic_results = []

    # --- PART 1: CUSTOM REASONING PRIMITIVES ---
    print("\n" + "=" * 60)
    print("🧠 Running Reasoning Primitives (5-shot)")
    print("=" * 60)

    # Generate data
    primitives = create_reasoning_primitives_data(config)
    
    if not primitives:
        print("⚠️ No reasoning primitives configured. Skipping.")
    else:
        # Determine template format from config
        template_format = config.get("reasoning_primitives", {}).get("template_format", "plain")
        
        for task_name, samples in primitives.items():
            print(f"\n📋 Task: {task_name} ({len(samples)} samples)")
            correct = 0
            
            for item in tqdm(samples, desc=f"  {task_name}", leave=False):
                # Apply 5-shot formatting
                prompt = format_5_shot_prompt(samples, item, template_format=template_format)
                
                # Generate response (greedy decoding for consistency)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,  # Answer should be short (just a number)
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,  # Greedy for reproducibility
                    )
                
                # Extract generated text
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Parse answer (look for last "Answer:" due to few-shot context)
                if "Answer:" in full_text:
                    generated = full_text.split("Answer:")[-1].strip()
                elif "Response:" in full_text:
                    generated = full_text.split("Response:")[-1].strip()
                elif "Assistant:" in full_text:
                    generated = full_text.split("Assistant:")[-1].strip()
                else:
                    # Fallback: take everything after the prompt
                    generated = full_text[len(prompt):].strip()
                
                # Clean up generated answer (remove extra text)
                generated = generated.split()[0] if generated.split() else generated
                
                # Check correctness
                is_correct = generated == item["expected_answer"]
                if is_correct:
                    correct += 1
                
                holistic_results.append({
                    "task_category": "Reasoning Primitive",
                    "task_name": task_name,
                    "prompt": prompt,  # Log last 100 chars for debugging
                    "prediction": generated,
                    "target": item["expected_answer"],
                    "is_correct": is_correct,
                })
            
            accuracy = correct / len(samples) if samples else 0.0
            print(f"    ✅ Accuracy: {accuracy:.2%} ({correct}/{len(samples)})")

    # --- PART 2: STANDARD BENCHMARKS (lm-eval) ---
    print("\n" + "=" * 60)
    print("📚 Running Standard Benchmarks (lm-evaluation-harness)")
    print("=" * 60)

    try:
        import lm_eval
        from lm_eval import evaluator

        # Standard benchmark tasks
        standard_tasks = [
            # Closed Book QA
            "triviaqa",
            "nq_open",
            "webqs",
            # Open Book QA
            "squadv2",
            "drop",
            "coqa",
            # Math Word Problems
            "gsm8k",
            "svamp",
            "asdiv",
        ]

        print(f"📝 Configured tasks: {', '.join(standard_tasks)}")
        print("⚠️ Note: This may take significant time and download large datasets.")

        # Only run if explicitly enabled (to avoid long eval times)
        if config.get("ENABLE_HEAVY_BENCHMARKS", False):
            print("\n🚀 Starting benchmark evaluation...")
            
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model.name_or_path},dtype=float16",
                tasks=standard_tasks,
                num_fewshot=5,  # 5-shot evaluation
                batch_size=config.get("eval_batch_size", 4),
            )

            # Log results
            print("\n📊 Benchmark Results:")
            for task, res in results["results"].items():
                # Extract accuracy (different tasks use different metric keys)
                acc = res.get("acc,none") or res.get("acc") or res.get("exact_match")
                if acc is not None:
                    print(f"  • {task}: {acc:.2%}")
                    holistic_results.append({
                        "task_category": "Standard Benchmark",
                        "task_name": task,
                        "is_correct": acc,  # Store accuracy directly
                    })
                else:
                    print(f"  • {task}: No accuracy metric found")
        else:
            print("⏩ Skipping heavy benchmarks (set ENABLE_HEAVY_BENCHMARKS=True to run)")

    except ImportError:
        print("⚠️ 'lm-evaluation-harness' not installed. Skipping Standard Benchmarks.")
        print("ℹ️  Install with: pip install lm-eval")

    return holistic_results