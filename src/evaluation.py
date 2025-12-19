import pandas as pd
import re
import torch
import random
from tqdm.auto import tqdm
from .data_generator import create_reasoning_primitives_data, format_5_shot_prompt

def analyze_experiment_results(accuracy_results: list, perplexity_results: list = None):
    """Generate summary statistics dataframe"""
    if not accuracy_results:
        return pd.DataFrame()

    df = pd.DataFrame(accuracy_results)

    # Group by UT steps and task
    summary = (
        df.groupby(["ut_steps", "task_type"])
        .agg(
            {
                "is_correct": ["mean", "count", "std"],
                "generation_time": ["mean", "min", "max"],
                "generated_tokens": ["mean"],
            }
        )
        .round(3)
    )

    # Flatten columns
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    return summary


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
    """
    holistic_results = []

    # --- PART 1: CUSTOM REASONING PRIMITIVES ---
    print("\n" + "=" * 40 + "\n🧠 Running Reasoning Primitives (5-shot)\n" + "=" * 40)

    # Generate data
    primitives = create_reasoning_primitives_data(config)

    for task_name, samples in primitives.items():
        print(f"  Task: {task_name} ({len(samples)} samples)")
        correct = 0

        for item in tqdm(samples, desc=task_name, leave=False):
            # Apply 5-shot formatting
            prompt = format_5_shot_prompt(samples, item)

            # Simple Generation (Greedy)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )

            # Extract output (naive splitting on 'Answer:')
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # We look for the *last* "Answer:" because of few-shot context
            if "Answer:" in full_text:
                generated = full_text.split("Answer:")[-1].strip()
            else:
                generated = full_text.strip()

            # Check correctness
            if generated == item["expected_answer"]:
                correct += 1

            holistic_results.append(
                {
                    "task_category": "Reasoning Primitive",
                    "task_name": task_name,
                    "prompt": prompt[-50:] + "...",  # Log truncation
                    "prediction": generated,
                    "target": item["expected_answer"],
                    "is_correct": generated == item["expected_answer"],
                }
            )

        acc = correct / len(samples) if samples else 0.0
        print(f"    ✅ Accuracy: {acc:.2%}")

    # --- PART 2: STANDARD BENCHMARKS (lm-eval) ---
    print("\n" + "=" * 40 + "\n📚 Running Standard Benchmarks (lm-eval)\n" + "=" * 40)

    # Check if library is available
    try:
        import lm_eval
        from lm_eval import evaluator

        # Map the 4 slices to lm-eval task names
        # Note: Task names depend on specific lm-eval version.
        # These are common identifiers.
        standard_tasks = [
            # Closed Book QA
            "triviaqa",
            "nq_open",
            "webqs",
            # Open Book QA
            "squadv2",
            "drop",
            "coqa",
            # Math Word Problems (5-shot usually handled by harness config)
            "gsm8k",
            "svamp",
            "asdiv",
        ]

        print(f"Attempting to run tasks: {standard_tasks}")
        print("⚠️ Note: This may take a long time and download large datasets.")

        # We wrap this in a try-block because running 19 benchmarks
        # in a script can be heavy.
        if config.get("ENABLE_HEAVY_BENCHMARKS", False):
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model.name_or_path},dtype=float16",
                tasks=standard_tasks,
                num_fewshot=5,  # Global 5-shot setting
                batch_size=4,
            )

            # Log results
            for task, res in results["results"].items():
                acc = res.get("acc,none") or res.get("acc")
                print(f"  {task}: {acc:.2%}")
                holistic_results.append(
                    {
                        "task_category": "Standard Benchmark",
                        "task_name": task,
                        "is_correct": acc,  # Storing score directly
                    }
                )
        else:
            print("⏩ Skipping heavy benchmarks (ENABLE_HEAVY_BENCHMARKS=False)")

    except ImportError:
        print("⚠️ 'lm-evaluation-harness' not installed. Skipping Standard Benchmarks.")
        print("ℹ️ To run: pip install lm-evaluation-harness")

    return holistic_results