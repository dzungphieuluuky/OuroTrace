import pandas as pd
import re
import torch
import random
from tqdm.auto import tqdm
from .data import create_reasoning_primitives_data, format_5_shot_prompt

def analyze_experiment_results(accuracy_results):
    if not accuracy_results: return pd.DataFrame()
    df = pd.DataFrame(accuracy_results)
    summary = df.groupby(['ut_steps', 'task_type']).agg({
        'is_correct': ['mean', 'count', 'std'],
        'generation_time': ['mean', 'min', 'max'],
        'generated_tokens': ['mean']
    }).round(3)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary

def run_holistic_evaluation(model, tokenizer, config):
    results = []
    print("\n" + "="*40 + "\n🧠 Running Reasoning Primitives (5-shot)\n" + "="*40)
    primitives = create_reasoning_primitives_data(config)
    
    for task_name, samples in primitives.items():
        print(f"  Task: {task_name} ({len(samples)})")
        correct = 0
        for item in tqdm(samples, desc=task_name, leave=False):
            prompt = format_5_shot_prompt(samples, item)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            
            full = tokenizer.decode(out[0], skip_special_tokens=True)
            gen = full.split("Answer:")[-1].strip() if "Answer:" in full else full.strip()
            
            if gen == item['expected_answer']: correct += 1
            results.append({'task': task_name, 'correct': gen == item['expected_answer']})
        print(f"    ✅ Acc: {correct/len(samples):.2%}")

    if config.get('ENABLE_HEAVY_BENCHMARKS', False):
        try:
            import lm_eval
            from lm_eval import evaluator
            print("\nRunning lm-eval benchmarks...")
            # ... lm-eval code ...
        except ImportError:
            print("lm-eval not installed.")
            
    return results