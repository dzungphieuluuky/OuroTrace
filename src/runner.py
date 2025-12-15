import time
import wandb
import gc
import torch
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Any, Optional, Tuple

# Import utilities (adjust paths as needed)
from .utils import generate_test_id
from .data_generator import (
    create_test_datasets, 
    create_perplexity_data, 
    load_and_preprocess_data
)
from .model import OuroBatchExperiment
from .evaluation import run_holistic_evaluation


def run_batch_experiment(config: dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Run experiment with batching support and W&B logging.
    
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
                project=wb_conf.get("project", "ouro-looped-transformer"),
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

    # 2. Extract Configuration
    model_path = config["MODEL"]["path"]
    ut_steps_list = config["INFERENCE_STEPS"]
    data_config = config["DATA"]
    eval_settings = config["EVAL_SETTINGS"]
    optimization_config = config.get("OPTIMIZATION", {})

    # 3. Setup Experiment Handler
    experiment = OuroBatchExperiment(
        model_path,
        dtype=config["MODEL"].get("dtype", torch.float16),
        use_4bit_quant=config["MODEL"].get("use_4bit_quant", True),
        use_torch_compile=config["MODEL"].get("use_torch_compile", False),
        max_batch_size=optimization_config.get("max_batch_size", 4),
        max_new_tokens=optimization_config.get("max_new_tokens", 512),  # Reduced default
    )

    torch.manual_seed(42)
    print(f"🎲 Random seed set to 42")

    # 4. Prepare Test Datasets
    print("\n📦 Loading test datasets...")
    if data_config.get("load_existing", False):
        test_datasets = load_and_preprocess_data(data_config["data_file_path"])
        print(f"✅ Loaded existing data from {data_config['data_file_path']}")
    else:
        print("⚙️ Generating new test datasets...")
        test_datasets = create_test_datasets(data_config)
        print(f"✅ Generated test datasets")
    
    # Print dataset summary
    for task_type, items in test_datasets.items():
        print(f"   - {task_type}: {len(items)} samples")

    # 5. Prepare Perplexity Data (if needed)
    perplexity_results = []
    perplexity_data = []
    
    if eval_settings.get("calculate_perplexity", False):
        print("\n📚 Preparing perplexity evaluation data...")
        raw_ppl_data = create_perplexity_data(eval_settings["ppl_num_samples"])
        perplexity_data = ["\n\n".join(raw_ppl_data)]
        print(f"✅ Prepared {eval_settings['ppl_num_samples']} samples for PPL")

    all_results = []
    holistic_results = []

    # 6. Main Experiment Loop (over different UT steps)
    for ut_step_idx, ut_steps in enumerate(ut_steps_list):
        print(f"\n{'='*70}")
        print(f"🧪 EXPERIMENT {ut_step_idx + 1}/{len(ut_steps_list)}: UT Steps = {ut_steps}")
        print(f"{'='*70}\n")

        # Load model with specific UT steps configuration
        try:
            model, tokenizer, model_config, config_dict = experiment.load_model_with_ut_steps(
                ut_steps, eval_settings.get("early_exit_threshold", 1.0)
            )
        except Exception as e:
            print(f"❌ Failed to load model with UT steps={ut_steps}: {e}")
            continue

        # Verify configuration
        if hasattr(model.config, 'total_ut_steps'):
            actual_steps = model.config.total_ut_steps
            print(f"✅ Model loaded with UT steps: {actual_steps}")
            
            if actual_steps != ut_steps:
                print(f"⚠️ WARNING: Config mismatch! Requested={ut_steps}, Actual={actual_steps}")
        else:
            print(f"⚠️ WARNING: Model config missing 'total_ut_steps' attribute")

        # Build task templates (only once)
        if not hasattr(experiment, "_templates_precomputed"):
            print("🔧 Building task templates...")
            experiment._build_task_templates(tokenizer)
            experiment._templates_precomputed = True

        # A. PERPLEXITY EVALUATION
        if perplexity_data:
            print(f"\n📉 Calculating perplexity...")
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
                
                print(f"✅ Perplexity: {ppl:.4f} | Avg Loss: {avg_loss:.4f}")

                if use_wandb:
                    wandb.log({
                        "perplexity": ppl,
                        "val_loss": avg_loss,
                        "ut_steps": ut_steps
                    })
            
            except Exception as e:
                print(f"⚠️ Perplexity calculation failed: {e}")

        # B. ACCURACY & PERFORMANCE EVALUATION
        enable_batch = optimization_config.get("enable_batch", True)
        
        for task_type, items in test_datasets.items():
            if not items:
                print(f"⚠️ Skipping {task_type} - no test items")
                continue
            
            print(f"\n📝 Evaluating task: {task_type.upper()} ({len(items)} samples)")
            task_results = []
            task_start_time = time.time()

            # Determine optimal batch size for this task
            batch_size = 1
            if enable_batch and hasattr(model, 'generate_batch'):
                task_batch_limits = {
                    "n_ary": 8,
                    "p_hop": 4,
                    "igsm": 2
                }
                batch_size = min(
                    task_batch_limits.get(task_type, 1),
                    experiment.max_batch_size
                )
                print(f"   Using batch size: {batch_size}")
            else:
                print(f"   Using sequential processing (batch_size=1)")

            # Process items in batches or sequentially
            if batch_size > 1 and len(items) >= batch_size:
                # BATCHED PROCESSING
                print(f"   🔄 Running batched inference...")
                
                for batch_idx in tqdm(
                    range(0, len(items), batch_size),
                    desc=f"   {task_type}",
                    leave=False
                ):
                    batch_items = items[batch_idx : batch_idx + batch_size]
                    prompts = [item["prompt"] for item in batch_items]

                    try:
                        batch_outputs = experiment.batch_predict_with_metrics(
                            prompts=prompts,
                            task_type=task_type,
                            model=model,
                            tokenizer=tokenizer,
                            ut_steps=ut_steps
                        )

                        # Process each output in the batch
                        for output, item in zip(batch_outputs, batch_items):
                            result_entry = _create_result_entry(
                                output, item, task_type, ut_steps
                            )
                            task_results.append(result_entry)
                            all_results.append(result_entry)
                    
                    except Exception as e:
                        print(f"⚠️ Batch {batch_idx} failed: {e}")
                        # Fallback to sequential for this batch
                        for item in batch_items:
                            try:
                                output = experiment.predict_with_metrics_optimized(
                                    user_input=item["prompt"],
                                    task_type=task_type,
                                    model=model,
                                    tokenizer=tokenizer,
                                    ut_steps=ut_steps
                                )
                                result_entry = _create_result_entry(
                                    output, item, task_type, ut_steps
                                )
                                task_results.append(result_entry)
                                all_results.append(result_entry)
                            except Exception as e2:
                                print(f"⚠️ Item failed: {e2}")
                                # Create error entry
                                error_result = {
                                    "prediction": "ERROR",
                                    "full_response": str(e2),
                                    "generation_time": 0,
                                    "generated_tokens": 0,
                                    "input_tokens": 0,
                                }
                                result_entry = _create_result_entry(
                                    error_result, item, task_type, ut_steps
                                )
                                task_results.append(result_entry)
                                all_results.append(result_entry)
            
            else:
                # SEQUENTIAL PROCESSING
                print(f"   🔄 Running sequential inference...")
                
                for item in tqdm(items, desc=f"   {task_type}", leave=False):
                    try:
                        output = experiment.predict_with_metrics_optimized(
                            user_input=item["prompt"],
                            task_type=task_type,
                            model=model,
                            tokenizer=tokenizer,
                            ut_steps=ut_steps
                        )
                        result_entry = _create_result_entry(
                            output, item, task_type, ut_steps
                        )
                        task_results.append(result_entry)
                        all_results.append(result_entry)
                    
                    except Exception as e:
                        print(f"⚠️ Item failed: {e}")
                        error_result = {
                            "prediction": "ERROR",
                            "full_response": str(e),
                            "generation_time": 0,
                            "generated_tokens": 0,
                            "input_tokens": 0,
                        }
                        result_entry = _create_result_entry(
                            error_result, item, task_type, ut_steps
                        )
                        task_results.append(result_entry)
                        all_results.append(result_entry)

            # Log task summary
            _log_task_summary(
                task_results, task_type, ut_steps, task_start_time, use_wandb
            )

            # Print sample results
            if task_results:
                print(f"\n   📊 Sample results from {task_type}:")
                df_sample = pd.DataFrame(task_results).head(20)
                display_cols = ['test_input', 'expected_answer', 'prediction', 'is_correct']
                display_cols = [c for c in display_cols if c in df_sample.columns]
                print(df_sample[display_cols].to_string(index=False))
                print()

        # C. HOLISTIC EVALUATION (if enabled)
        if config.get("reasoning_primitives") or config.get("ENABLE_HEAVY_BENCHMARKS"):
            print(f"\n🎯 Running holistic evaluation...")
            try:
                holistic_eval = run_holistic_evaluation(model, tokenizer, config)
                holistic_eval['ut_steps'] = ut_steps
                holistic_results.append(holistic_eval)
                print(f"✅ Holistic evaluation completed")
            except Exception as e:
                print(f"⚠️ Holistic evaluation failed: {e}")

        # Cleanup GPU memory
        print(f"\n🧹 Cleaning up GPU memory...")
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"✅ GPU memory freed")

    # 7. Final Summary
    print(f"\n{'='*70}")
    print(f"📊 EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")
    
    if all_results:
        df_all = pd.DataFrame(all_results)
        
        print("Overall Accuracy by Task Type:")
        accuracy_by_task = df_all.groupby('task_type')['is_correct'].agg(['mean', 'count'])
        accuracy_by_task.columns = ['Accuracy', 'N']
        accuracy_by_task['Accuracy'] = (accuracy_by_task['Accuracy'] * 100).round(2)
        print(accuracy_by_task)
        print()
        
        print("Accuracy by UT Steps:")
        accuracy_by_steps = df_all.groupby('ut_steps')['is_correct'].agg(['mean', 'count'])
        accuracy_by_steps.columns = ['Accuracy', 'N']
        accuracy_by_steps['Accuracy'] = (accuracy_by_steps['Accuracy'] * 100).round(2)
        print(accuracy_by_steps)
        print()
    
    if perplexity_results:
        print("Perplexity by UT Steps:")
        df_ppl = pd.DataFrame(perplexity_results)
        print(df_ppl.to_string(index=False))
        print()

    # 8. Close W&B
    if use_wandb and run:
        print("🔗 Finalizing W&B...")
        wandb.finish()
        print("✅ W&B session closed")

    return all_results, perplexity_results, holistic_results


def _create_result_entry(
    result: Dict[str, Any], 
    item: Dict[str, Any], 
    task_type: str, 
    ut_steps: int
) -> Dict[str, Any]:
    """
    Create a standardized result entry with correctness evaluation.
    
    Args:
        result: Prediction result from model
        item: Test item with prompt and expected answer
        task_type: Type of task (n_ary, p_hop, igsm)
        ut_steps: Number of UT steps used
    
    Returns:
        Standardized result dictionary
    """
    pred = str(result.get("prediction", "ERROR")).strip().lower()
    target = str(item["expected_answer"]).strip().lower()

    # Determine correctness based on task type
    is_correct = False
    
    if task_type == "p_hop":
        # For p_hop, exact string match (case-insensitive)
        is_correct = (pred == target)
    
    elif task_type in ["n_ary", "igsm"]:
        # For numeric tasks, try numeric comparison with tolerance
        try:
            pred_num = float(pred)
            target_num = float(target)
            is_correct = abs(pred_num - target_num) < 0.001
        except (ValueError, TypeError):
            # Fallback to string comparison if not numeric
            is_correct = (pred == target)
    
    else:
        # Default: string comparison
        is_correct = (pred == target)

    # Build result entry
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
        "prompt_idx": result.get("prompt_idx", 0),
    }


def _log_task_summary(
    results: List[Dict[str, Any]], 
    task_type: str, 
    ut_steps: int, 
    start_time: float, 
    use_wandb: bool
) -> None:
    """
    Log summary statistics for a task.
    
    Args:
        results: List of result dictionaries
        task_type: Type of task
        ut_steps: Number of UT steps
        start_time: Task start time
        use_wandb: Whether to log to W&B
    """
    if not results:
        print(f"   ⚠️ No results to summarize for {task_type}")
        return

    # Calculate metrics
    num_samples = len(results)
    num_correct = sum(r["is_correct"] for r in results)
    accuracy = num_correct / num_samples if num_samples > 0 else 0.0
    
    total_gen_time = sum(r.get("generation_time", 0) for r in results)
    avg_gen_time = total_gen_time / num_samples if num_samples > 0 else 0.0
    
    total_tokens = sum(r.get("generated_tokens", 0) for r in results)
    avg_tokens = total_tokens / num_samples if num_samples > 0 else 0.0
    
    total_duration = time.time() - start_time

    # Print summary
    print(f"\n   📊 Task Summary for {task_type}:")
    print(f"      Accuracy: {accuracy:.2%} ({num_correct}/{num_samples})")
    print(f"      Avg Generation Time: {avg_gen_time:.3f}s")
    print(f"      Avg Tokens: {avg_tokens:.1f}")
    print(f"      Total Duration: {total_duration:.1f}s")
    print(f"      Throughput: {num_samples/total_duration:.2f} samples/sec")

    # Log to W&B
    if use_wandb:
        try:
            wandb.log({
                f"{task_type}/accuracy": accuracy,
                f"{task_type}/avg_generation_time": avg_gen_time,
                f"{task_type}/avg_tokens": avg_tokens,
                f"{task_type}/throughput": num_samples / total_duration if total_duration > 0 else 0,
                f"{task_type}/num_samples": num_samples,
                "ut_steps": ut_steps,
            })
        except Exception as e:
            print(f"   ⚠️ Failed to log to W&B: {e}")


# Optional: Add a function to save results to disk
def save_results(
    all_results: List[Dict],
    perplexity_results: List[Dict],
    holistic_results: List[Dict],
    output_dir: str = "./results"
) -> None:
    """Save experiment results to CSV files."""
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if all_results:
        accuracy_file = os.path.join(output_dir, f"accuracy_{timestamp}.csv")
        pd.DataFrame(all_results).to_csv(accuracy_file, index=False)
        print(f"✅ Saved accuracy results to {accuracy_file}")
    
    if perplexity_results:
        ppl_file = os.path.join(output_dir, f"perplexity_{timestamp}.csv")
        pd.DataFrame(perplexity_results).to_csv(ppl_file, index=False)
        print(f"✅ Saved perplexity results to {ppl_file}")
    
    if holistic_results:
        holistic_file = os.path.join(output_dir, f"holistic_{timestamp}.csv")
        pd.DataFrame(holistic_results).to_csv(holistic_file, index=False)
        print(f"✅ Saved holistic results to {holistic_file}")