import time
import wandb
import gc
import torch
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Any, Optional, Tuple
from .evaluation_metrics import (
    OuroMetrics,
    analyze_experiment_results,
    PaperComplianceChecker
)

# Import utilities (adjust paths as needed)
from .utils import generate_test_id
from .data_generator import (
    create_test_datasets, 
    create_perplexity_data, 
    load_and_preprocess_data
)
from .new_model import SafeOuroThinkingExperiment
from .evaluation import run_holistic_evaluation


def run_batch_experiment(config: dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Run experiment with automatic batch/compile optimization based on UT steps.
    
    AUTO-OPTIMIZATION RULES:
    - UT Steps = 1: Enable batching + torch.compile (fast path)
    - UT Steps > 1: Disable batching + torch.compile (stability)
    
    Args:
        config: Configuration dictionary with MODEL, DATA, EVAL_SETTINGS, etc.
    
    Returns:
        Tuple of (accuracy_results, perplexity_results, holistic_results)
    """
    # 1. Initialize W&B
    use_wandb = config.get("WANDB", {}).get("enabled", False)
    run = None

    if use_wandb:
        wb_conf = config["WANDB"]
        print(f"🔗 Initializing W&B (timeout: {wb_conf.get('timeout', 30)}s)...")

        try:
            run = wandb.init(
                project=wb_conf.get("project", "ouro-trace"),
                entity=wb_conf.get("entity", None),
                name=wb_conf.get("run_name", f"run_{int(time.time())}"),
                config=config,
                mode=wb_conf.get("mode", "online"),
                settings=wandb.Settings(
                    start_timeout=wb_conf.get("timeout", 30), 
                    _disable_stats=True
                ),
            )
            print("✅ W&B initialized")
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}. Continuing offline.")
            use_wandb = False
            run = None

    # 2. Extract and Display Configuration
    model_config = config["MODEL"]
    model_path = model_config["path"]
    ut_steps_list = config["INFERENCE_STEPS"]
    data_config = config["DATA"]
    eval_settings = config["EVAL_SETTINGS"]
    optimization_config = config.get("OPTIMIZATION", {})

    print(f"\n{'='*70}")
    print(f"🔧 EXPERIMENT CONFIGURATION")
    print(f"{'='*70}")
    print(f"Model Path: {model_path}")
    print(f"UT Steps to Test: {ut_steps_list}")
    print(f"Data Type: {model_config.get('dtype', torch.bfloat16)}")
    print(f"4-bit Quantization: {model_config.get('use_4bit_quant', True)}")
    print(f"Torch Compile: {model_config.get('use_torch_compile', True)}")
    print(f"Max Batch Size: {optimization_config.get('max_batch_size', 4)}")
    print(f"Max New Tokens: {optimization_config.get('max_new_tokens', 256)}")
    print(f"Batching: {optimization_config.get('enable_batch', True)}")
    print(f"Calculate Perplexity: {eval_settings.get('calculate_perplexity', True)}")
    print(f"Early Exit: {eval_settings.get('early_exit_threshold', 1.0)}")
    print(f"{'='*70}\n")

    
    # 3. Setup Experiment Handler
    experiment = SafeOuroThinkingExperiment(
        model_path,
        dtype=config["MODEL"].get("dtype", torch.bfloat16),
        use_4bit_quant=config["MODEL"].get("use_4bit_quant", True),
        use_torch_compile=config["MODEL"].get("use_torch_compile", True),
        max_batch_size=optimization_config.get("max_batch_size", 4),
        max_new_tokens=optimization_config.get("max_new_tokens", 256),
    )

    torch.manual_seed(42)
    print(f"🎲 Random seed set to 42")

    # 4. Prepare Test Datasets
    print(f"\n{'='*70}")
    print(f"📦 LOADING TEST DATASETS")
    print(f"{'='*70}")
    
    if data_config.get("load_existing", False):
        print(f"Loading from: {data_config['data_file_path']}")
        test_datasets = load_and_preprocess_data(data_config["data_file_path"])
        print(f"✅ Loaded existing data")
    else:
        print("⚙️ Generating new test datasets...")
        test_datasets = create_test_datasets(data_config)
        print(f"✅ Generated test datasets")
    
    # Print dataset summary
    print(f"\nDataset Summary:")
    for task_type, items in test_datasets.items():
        print(f"   {task_type:12s}: {len(items):4d} samples")
    print(f"{'='*70}\n")

    # Check experiment compliance with paper
    checker = PaperComplianceChecker()
    
    task_alignment = checker.check_task_alignment(list(test_datasets.keys()))
    ut_coverage = checker.check_ut_steps_coverage(ut_steps_list)
    
    print(f"\n{'='*70}")
    print(f"📋 PAPER COMPLIANCE CHECK")
    print(f"{'='*70}")
    print(f"Task Alignment: {task_alignment}")
    print(f"UT Steps Coverage: {ut_coverage}")
    print(f"{'='*70}\n")

    # 5. Prepare Perplexity Data (if needed)
    perplexity_results = []
    perplexity_data = []
    
    if eval_settings.get("calculate_perplexity", False):
        print(f"📚 Preparing perplexity evaluation data...")
        raw_ppl_data = create_perplexity_data(eval_settings["ppl_num_samples"])
        perplexity_data = ["\n\n".join(raw_ppl_data)]
        print(f"✅ Prepared {eval_settings['ppl_num_samples']} samples for PPL\n")

    all_results = []
    holistic_results = []

    # 6. Main Experiment Loop (over different UT steps)
    for ut_step_idx, ut_steps in enumerate(ut_steps_list):
        print(f"\n{'='*70}")
        print(f"🧪 EXPERIMENT {ut_step_idx + 1}/{len(ut_steps_list)}: UT Steps = {ut_steps}")
        print(f"{'='*70}\n")

        # AUTO-OPTIMIZATION: Determine if batching should be enabled
        enable_batch = optimization_config.get("enable_batch", True)
        
        print(f"⚙️  AUTO-OPTIMIZATION SETTINGS:")
        print(f"   Batch Processing: {'✅ ENABLED' if enable_batch else '❌ DISABLED'}")
        print(f"   Torch Compile: {'✅ ENABLED' if model_config.get('use_torch_compile', True) else '❌ DISABLED'}")
        print(f"   NOTE: torch.compile is not the culprit, batching with generate(), not with generate_batch() function.")
        print()

        # Load model with specific UT steps configuration
        try:
            model, tokenizer, model_config, config_dict = experiment.load_model_with_ut_steps(ut_steps)
        except Exception as e:
            print(f"❌ Failed to load model with UT steps={ut_steps}: {e}")
            continue

        # Build task templates (only once)
        if not hasattr(experiment, "_templates_precomputed"):
            print("🔧 Building task templates...")
            experiment._build_task_templates(tokenizer)
            experiment._templates_precomputed = True
            print("✅ Task templates built\n")
            print()

        # A. PERPLEXITY EVALUATION
        if perplexity_data:
            print(f"{'='*70}")
            print(f"📉 PERPLEXITY EVALUATION")
            print(f"{'='*70}\n")
            
            try:
                ppl, avg_loss = experiment.calculate_perplexity(
                    model,
                    tokenizer,
                    perplexity_data,
                    ut_steps,
                    max_length=eval_settings.get("ppl_max_length", 2048),
                    stride=eval_settings.get("ppl_stride", 512),
                )
                
                perplexity_results.append({
                    "ut_steps": ut_steps,
                    "perplexity": ppl,
                    "avg_loss": avg_loss
                })
                
                print(f"\n✅ Perplexity Results:")
                print(f"   Perplexity: {ppl:.4f}")
                print(f"   Avg Loss:   {avg_loss:.4f}\n")

                if use_wandb:
                    wandb.log({
                        "perplexity": ppl,
                        "val_loss": avg_loss,
                        "ut_steps": ut_steps
                    })
            
            except Exception as e:
                print(f"⚠️ Perplexity calculation failed: {e}\n")

        # B. ACCURACY & PERFORMANCE EVALUATION
        print(f"{'='*70}")
        print(f"🎯 ACCURACY EVALUATION")
        print(f"{'='*70}\n")
        
        for task_type, items in test_datasets.items():
            if not items:
                print(f"⚠️ Skipping {task_type} - no test items\n")
                continue
            
            print(f"\n{'─'*70}")
            print(f"📝 Task: {task_type.upper()}")
            print(f"{'─'*70}")
            print(f"Total Samples: {len(items)}")
            
            task_results = []
            task_start_time = time.time()

            # Determine optimal batch size for this task (only if batch enabled)
            batch_size = 1
            if enable_batch:
                task_batch_limits = {
                    "n_ary": 8,
                    "p_hop": 4,
                    "igsm": 2
                }
                batch_size = min(
                    task_batch_limits.get(task_type, 1),
                    experiment.max_batch_size
                )
                print(f"Batch Size: {batch_size}")
                print(f"Strategy: Batched Processing")
            else:
                print(f"Batch Size: 1 (Sequential)")
                print(f"Strategy: Sequential Processing")
            
            print()

            # Process items in batches or sequentially using unified predict()
            if batch_size > 1 and len(items) >= batch_size:
                # BATCHED PROCESSING
                num_batches = (len(items) + batch_size - 1) // batch_size if enable_batch else 1
                print(f"Running {num_batches} batches...")
                
                for batch_idx in tqdm(
                    range(0, len(items), batch_size),
                    desc=f"   {task_type}",
                    leave=False,
                    total=num_batches
                ):
                    batch_items = items[batch_idx : batch_idx + batch_size]
                    prompts = [item["prompt"] for item in batch_items]

                    try:
                        # Use unified predict() with list of prompts
                        batch_outputs = experiment.predict(
                            user_inputs=prompts,
                            task_type=task_type,
                            model=model,
                            tokenizer=tokenizer,
                            ut_steps=ut_steps,
                            enable_batch=enable_batch,
                        )

                        # Process each output
                        for output, item in zip(batch_outputs, batch_items):
                            result_entry = _create_result_entry(
                                output, item, task_type, ut_steps
                            )
                            task_results.append(result_entry)
                            all_results.append(result_entry)
                            print(pd.DataFrame([result_entry])[['test_input', 'full_response']])
                            experiment.monitor_and_maybe_abort(result_entry, task_type)
                    
                    except Exception as e:
                        print(f"\n⚠️ Batch {batch_idx//batch_size + 1} failed: {e}")
                        # Fallback to sequential for this batch
                        for item in batch_items:
                            try:
                                output = experiment.predict(
                                    user_inputs=item["prompt"],
                                    task_type=task_type,
                                    model=model,
                                    tokenizer=tokenizer,
                                    ut_steps=ut_steps,
                                    enable_batch=False,
                                )
                                result_entry = _create_result_entry(
                                    output, item, task_type, ut_steps
                                )
                                task_results.append(result_entry)
                                all_results.append(result_entry)
                                print(pd.DataFrame([result_entry])[['test_input', 'full_response']])
                                experiment.monitor_and_maybe_abort(result_entry, task_type)
                            except Exception as e2:
                                print(f"⚠️ Item failed: {e2}")
                                error_result = {
                                    "prediction": "ERROR",
                                    "full_response": str(e2),
                                    "generation_time": 0,
                                    "generated_tokens": 0,
                                    "input_tokens": 0,
                                    "is_degenerate": False,
                                }
                                result_entry = _create_result_entry(
                                    error_result, item, task_type, ut_steps
                                )
                                task_results.append(result_entry)
                                all_results.append(result_entry)
                                print(pd.DataFrame([result_entry])[['test_input', 'full_response']])
                                experiment.monitor_and_maybe_abort(result_entry, task_type)
            else:
                # SEQUENTIAL PROCESSING
                print(f"Processing {len(items)} items sequentially...")
                
                for item in tqdm(items, desc=f"   {task_type}", leave=False):
                    try:
                        # Use unified predict() with single prompt
                        output = experiment.predict(
                            user_inputs=item["prompt"],
                            task_type=task_type,
                            model=model,
                            tokenizer=tokenizer,
                            ut_steps=ut_steps,
                            enable_batch=False,
                        )
                        result_entry = _create_result_entry(
                            output, item, task_type, ut_steps
                        )
                        task_results.append(result_entry)
                        all_results.append(result_entry)
                        print(pd.DataFrame([result_entry])[['test_input', 'full_response']])
                        experiment.monitor_and_maybe_abort(result_entry, task_type)
                    except Exception as e:
                        print(f"⚠️ Item failed: {e}")
                        error_result = {
                            "prediction": "ERROR",
                            "full_response": str(e),
                            "generation_time": 0,
                            "generated_tokens": 0,
                            "input_tokens": 0,
                            "is_degenerate": False,
                        }
                        result_entry = _create_result_entry(
                            error_result, item, task_type, ut_steps
                        )
                        task_results.append(result_entry)
                        all_results.append(result_entry)
                        print(pd.DataFrame([result_entry])[['test_input', 'full_response']])
                        experiment.monitor_and_maybe_abort(result_entry, task_type)
            
            # Log and display task summary
            _log_task_summary(
                task_results, task_type, ut_steps, task_start_time, use_wandb
            )

            # Display sample results (first 5)
            _display_sample_results(task_results, task_type)

        # C. HOLISTIC EVALUATION (if enabled)
        if config.get("reasoning_primitives") or config.get("ENABLE_HEAVY_BENCHMARKS"):
            print(f"\n{'='*70}")
            print(f"🎯 HOLISTIC EVALUATION")
            print(f"{'='*70}\n")
            
            try:
                holistic_eval = run_holistic_evaluation(model, tokenizer, config)
                holistic_eval['ut_steps'] = ut_steps
                holistic_results.append(holistic_eval)
                print(f"✅ Holistic evaluation completed\n")
            except Exception as e:
                print(f"⚠️ Holistic evaluation failed: {e}\n")

        # Cleanup GPU memory
        print(f"{'='*70}")
        print(f"🧹 Cleaning up GPU memory...")
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"✅ GPU memory freed")
        print(f"{'='*70}\n")

    # 7. Final Summary
    print(f"\n{'='*70}")
    print(f"📊 FINAL EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")
    
    if all_results:
        df_all = pd.DataFrame(all_results)
        
        # Check for garbage outputs
        if 'is_degenerate' in df_all.columns:
            num_garbage = df_all['is_degenerate'].sum()
            if num_garbage > 0:
                print(f"⚠️ WARNING: {num_garbage} garbage/degenerate outputs detected\n")
        
        print("📈 Overall Accuracy by Task Type:")
        print(f"{'─'*70}")
        accuracy_by_task = df_all.groupby('task_type')['is_correct'].agg(['mean', 'count'])
        accuracy_by_task.columns = ['Accuracy', 'N']
        accuracy_by_task['Accuracy'] = (accuracy_by_task['Accuracy'] * 100).round(2)
        accuracy_by_task['Accuracy'] = accuracy_by_task['Accuracy'].apply(lambda x: f"{x:.2f}%")
        print(accuracy_by_task)
        print()
        
        print("📈 Accuracy by UT Steps:")
        print(f"{'─'*70}")
        accuracy_by_steps = df_all.groupby('ut_steps')['is_correct'].agg(['mean', 'count'])
        accuracy_by_steps.columns = ['Accuracy', 'N']
        accuracy_by_steps['Accuracy'] = (accuracy_by_steps['Accuracy'] * 100).round(2)
        accuracy_by_steps['Accuracy'] = accuracy_by_steps['Accuracy'].apply(lambda x: f"{x:.2f}%")
        print(accuracy_by_steps)
        print()
        
        print("📈 Accuracy by Task Type and UT Steps:")
        print(f"{'─'*70}")
        accuracy_pivot = df_all.pivot_table(
            values='is_correct',
            index='task_type',
            columns='ut_steps',
            aggfunc='mean'
        ) * 100
        print(accuracy_pivot.round(2))
        print()
    
    if perplexity_results:
        print("📉 Perplexity by UT Steps:")
        print(f"{'─'*70}")
        df_ppl = pd.DataFrame(perplexity_results)
        print(df_ppl.to_string(index=False))
        print()

    # 8. Paper-Aligned Metrics Analysis
    if all_results:
        print(f"\n{'='*70}")
        print(f"📊 PAPER-ALIGNED ANALYSIS")
        print(f"{'='*70}\n")
        
        # Determine model size
        model_path = config["MODEL"]["path"]
        if "1.4" in model_path.lower():
            model_size_b = 1.4
            model_name = "Ouro-1.4B"
            if "thinking" in model_path.lower():
                model_name = "Ouro-1.4B-Thinking"
        elif "2.6" in model_path.lower():
            model_size_b = 2.6
            model_name = "Ouro-2.6B"
            if "thinking" in model_path.lower():
                model_name = "Ouro-2.6B-Thinking"
        else:
            model_size_b = 1.4  # default
            model_name = "Ouro"
        
        # Run analysis
        try:
            paper_metrics = analyze_experiment_results(
                all_results,
                model_name=model_name,
                model_size_b=model_size_b,
                save_plots=True
            )
            
            # Save metrics to CSV
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for metric_name, df in paper_metrics.items():
                if not df.empty:
                    filename = f"./results_{timestamp}/{metric_name}_{timestamp}.csv"
                    df.to_csv(filename, index=False)
                    print(f"✅ Saved {metric_name} to {filename}")
        
        except Exception as e:
            print(f"⚠️ Paper metrics analysis failed: {e}")

    # 9. Close W&B
    if use_wandb and run:
        print(f"{'='*70}")
        print("🔗 Finalizing W&B...")
        wandb.finish()
        print("✅ W&B session closed")
        print(f"{'='*70}\n")
    
    # Save results to csv files 
    save_results(all_results, perplexity_results, holistic_results, output_dir=f"./results_{timestamp}", timestamp=timestamp)

    # Save config file into yaml file
    save_config(config, output_dir=f"./results_{timestamp}", timestamp=timestamp)

    return all_results, perplexity_results, holistic_results


def _create_result_entry(
    result: Dict[str, Any], 
    item: Dict[str, Any], 
    task_type: str, 
    ut_steps: int
) -> Dict[str, Any]:
    """Create a standardized result entry with correctness evaluation."""
    pred = str(result.get("prediction", "ERROR")).strip().lower()
    target = str(item["expected_answer"]).strip().lower()

    # Determine correctness based on task type
    is_correct = False
    
    if task_type == "p_hop":
        is_correct = (pred == target)
    elif task_type in ["n_ary", "igsm"]:
        try:
            pred_num = float(pred)
            target_num = float(target)
            is_correct = abs(pred_num - target_num) < 0.001
        except (ValueError, TypeError):
            is_correct = (pred == target)
    else:
        is_correct = (pred == target)

    return {
        "task_type": task_type,
        "difficulty": item.get("difficulty", "unknown"),
        "test_input": item["prompt"],
        "expected_answer": item["expected_answer"],
        "prediction": result.get("prediction", "ERROR"),
        "is_correct": is_correct,
        "test_id": generate_test_id(
            task_type, 
            item.get("difficulty", ""), 
            item["prompt"]
        ),
        "ut_steps": ut_steps,
        "full_response": result.get("full_response", ""),
        "generation_time": result.get("generation_time", 0.0),
        "generated_tokens": result.get("generated_tokens", 0),
        "input_tokens": result.get("input_tokens", 0),
        "is_degenerate": result.get("is_degenerate", False),
    }


def _log_task_summary(
    results: List[Dict[str, Any]], 
    task_type: str, 
    ut_steps: int, 
    start_time: float, 
    use_wandb: bool
) -> None:
    """Log summary statistics for a task."""
    if not results:
        print(f"\n   ⚠️ No results to summarize for {task_type}\n")
        return

    # Calculate metrics
    num_samples = len(results)
    num_correct = sum(r["is_correct"] for r in results)
    num_degenerate = sum(r.get("is_degenerate", False) for r in results)
    accuracy = num_correct / num_samples if num_samples > 0 else 0.0
    
    total_gen_time = sum(r.get("generation_time", 0) for r in results)
    avg_gen_time = total_gen_time / num_samples if num_samples > 0 else 0.0
    
    total_tokens = sum(r.get("generated_tokens", 0) for r in results)
    avg_tokens = total_tokens / num_samples if num_samples > 0 else 0.0
    
    total_duration = time.time() - start_time
    throughput = num_samples / total_duration if total_duration > 0 else 0.0

    # Print summary
    print(f"\n{'─'*70}")
    print(f"📊 Summary for {task_type.upper()}")
    print(f"{'─'*70}")
    print(f"Accuracy:            {accuracy*100:6.2f}% ({num_correct}/{num_samples})")
    if num_degenerate > 0:
        print(f"Garbage Outputs:     {num_degenerate:6d} ({num_degenerate/num_samples*100:.1f}%)")
    print(f"Avg Gen Time:        {avg_gen_time:6.3f}s")
    print(f"Avg Tokens:          {avg_tokens:6.1f}")
    print(f"Total Duration:      {total_duration:6.1f}s")
    print(f"Throughput:          {throughput:6.2f} samples/sec")
    print(f"{'─'*70}\n")

    # Log to W&B
    if use_wandb:
        try:
            wandb.log({
                f"{task_type}/accuracy": accuracy,
                f"{task_type}/num_degenerate": num_degenerate,
                f"{task_type}/avg_generation_time": avg_gen_time,
                f"{task_type}/avg_tokens": avg_tokens,
                f"{task_type}/throughput": throughput,
                f"{task_type}/num_samples": num_samples,
                "ut_steps": ut_steps,
            })
        except Exception as e:
            print(f"   ⚠️ Failed to log to W&B: {e}")


def _display_sample_results(results: List[Dict[str, Any]], task_type: str, num_samples: int = 10) -> None:
    """Display sample results for inspection."""
    if not results:
        return
    
    print(f"📋 Sample Results for {task_type.upper()} (first {num_samples}):")
    print(f"{'─'*70}")
    
    df_sample = pd.DataFrame(results).head(num_samples)
    display_cols = ['test_input', 'full_response', 'generated_tokens', 'is_correct']
    
    # Add degenerate flag if present
    if 'is_degenerate' in df_sample.columns:
        display_cols.append('is_degenerate')
    
    # Truncate long text for display
    for col in ['test_input', 'full_response', 'generated_tokens']:
        if col in df_sample.columns:
            df_sample[col] = df_sample[col].astype(str).str[:60]
    
    print(df_sample[display_cols].to_string(index=False))
    print()


# Optional: Save results function
def save_results(
    all_results: List[Dict],
    perplexity_results: List[Dict],
    holistic_results: List[Dict],
    timestamp: Optional[str] = None,
    output_dir: str = "./results"
) -> None:
    """Save experiment results to CSV files."""
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if all_results:
        all_file = os.path.join(output_dir, f"all_{timestamp}.csv")
        pd.DataFrame(all_results).to_csv(all_file, index=False)
        print(f"✅ Saved all results to {all_file}")
    
    if perplexity_results:
        ppl_file = os.path.join(output_dir, f"perplexity_{timestamp}.csv")
        pd.DataFrame(perplexity_results).to_csv(ppl_file, index=False)
        print(f"✅ Saved perplexity results to {ppl_file}")
    
    if holistic_results:
        holistic_file = os.path.join(output_dir, f"holistic_{timestamp}.csv")
        pd.DataFrame(holistic_results).to_csv(holistic_file, index=False)
        print(f"✅ Saved holistic results to {holistic_file}")

def save_config(
    config: dict,
    timestamp: Optional[str] = None,
    output_dir: str = "./results"
) -> None:
    """Save experiment configuration to a YAML file."""
    import os
    import yaml
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # sanitize config before saving
    clean = {}
    for k, v in config.items():
        if isinstance(v, dict):
            clean[k] = {kk: str(vv) for kk, vv in v.items()}
        else:
            clean[k] = str(v)
            
    config_file = os.path.join(output_dir, f"config_{timestamp}.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(clean, f)
    print(f"✅ Saved config to {config_file}")