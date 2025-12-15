import time
import wandb
import gc
import torch
import random
from tqdm.auto import tqdm
from .utils import generate_test_id
from .data_generator import create_test_datasets, create_perplexity_data, load_and_preprocess_data
from .model import OuroBatchExperiment
from .evaluation import run_holistic_evaluation

def run_batch_experiment(config: dict):
    """
    OPTIMIZED: Run experiment with UT-step aware optimizations.
    Returns: (accuracy_results, perplexity_results)
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
                    start_timeout=wb_conf.get("timeout", 30), _disable_stats=True
                ),
            )
            print("✅ W&B initialized")
        except Exception as e:
            print(f"⚠️ W&B failed: {e}. Continuing offline.")
            use_wandb = False
            run = None

    # 2. Extract Config with optimizations
    model_path = config["MODEL"]["path"]
    ut_steps_list = config["INFERENCE_STEPS"]
    data_config = config["DATA"]
    eval_settings = config["EVAL_SETTINGS"]
    
    # NEW: Optimization settings
    opt_config = config.get("OPTIMIZATION", {})
    enable_dynamic_batching = opt_config.get("enable_dynamic_batching", True)
    enable_memory_optimization = opt_config.get("enable_memory_optimization", True)

    # 3. Setup Experiment with optimizations
    experiment = OuroBatchExperiment(
        model_path,
        dtype=config["MODEL"]["dtype"],
        use_4bit_quant=config["MODEL"].get("use_4bit_quant", True),
        use_torch_compile=config["MODEL"].get("use_torch_compile", True),
        max_batch_size=opt_config.get("max_batch_size", 4),
        max_new_tokens=opt_config.get("max_new_token", 1024),
    )

    torch.manual_seed(42)

    # 4. Prepare Data
    if data_config["load_existing"]:
        test_datasets = load_and_preprocess_data(data_config["data_file_path"])
    else:
        print("Generating new test datasets...")
        test_datasets = create_test_datasets(data_config)

    perplexity_results = []
    perplexity_data = []
    if eval_settings["calculate_perplexity"]:
        raw_ppl_data = create_perplexity_data(eval_settings["ppl_num_samples"])
        perplexity_data = ["\n\n".join(raw_ppl_data)]

    all_results = []

    # 5. Main Loop with optimizations
    for ut_steps in ut_steps_list:
        print(f"\n{'=' * 60}\n🧪 EXPERIMENT: UT Steps = {ut_steps}\n{'=' * 60}")
        
        # NEW: Pre-warm CUDA for better performance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Warm up CUDA
            warmup_tensor = torch.randn(1024, 1024, device='cuda')
            torch.matmul(warmup_tensor, warmup_tensor)
            del warmup_tensor

        # Load fresh model
        model, tokenizer, _, _ = experiment.load_model_with_ut_steps(
            ut_steps, eval_settings["early_exit_threshold"]
        )
        
        # NEW: Apply model-level optimizations immediately
        if hasattr(experiment, '_optimize_model_for_ut_steps'):
            model = experiment._optimize_model_for_ut_steps(model, ut_steps)

        if not hasattr(experiment, "_templates_precomputed"):
            experiment._build_task_templates(tokenizer)
            experiment._templates_precomputed = True

        # A. Perplexity
        if perplexity_data:
            print(f"📉 Calculating PPL...")
            ppl, avg_loss = experiment.calculate_perplexity(
                model,
                tokenizer,
                perplexity_data,
                ut_steps,
                max_length=eval_settings["ppl_max_length"],
                stride=eval_settings["ppl_stride"],
            )
            perplexity_results.append(
                {"ut_steps": ut_steps, "perplexity": ppl, "avg_loss": avg_loss}
            )
            print(f"✅ PPL: {ppl:.4f} | Loss: {avg_loss:.4f}")

            if use_wandb:
                wandb.log(
                    {"perplexity": ppl, "val_loss": avg_loss, "ut_steps": ut_steps}
                )

        # B. Accuracy & Time with dynamic batching
        enable_batch = opt_config.get("enable_batch", True)

        for task_type, items in test_datasets.items():
            print(f"\n📝 Task: {task_type} ({len(items)} samples)")
            task_results = []
            start_time = time.time()

            # NEW: Dynamic batch sizing based on UT steps
            if enable_batch:
                # Base limits per task
                base_limits = {"n_ary": 8, "p_hop": 4, "igsm": 2}
                base_batch = min(base_limits.get(task_type, 1), experiment.max_batch_size)
                
                # NEW: Adjust for UT steps
                if ut_steps > 1 and enable_dynamic_batching:
                    # Reduce batch size for multi-step to avoid OOM
                    batch_size = max(1, base_batch // ut_steps)
                    print(f"  Dynamic batch: {base_batch} → {batch_size} (UT{ut_steps})")
                else:
                    batch_size = base_batch
            else:
                batch_size = 1

            # Process Loop with optimizations
            if batch_size > 1 and len(items) >= 2:
                # OPTIMIZED: Batched Processing with memory management
                for i in range(0, len(items), batch_size):
                    batch_items = items[i : i + batch_size]
                    prompts = [item["prompt"] for item in batch_items]
                    
                    # NEW: Clear memory between batches for multi-UT steps
                    if ut_steps > 1 and enable_memory_optimization:
                        if hasattr(experiment, '_cleanup_between_batches'):
                            experiment._cleanup_between_batches()

                    batch_out = experiment.batch_predict_with_metrics(
                        prompts, task_type, model, tokenizer, ut_steps
                    )

                    for res, item in zip(batch_out, batch_items):
                        res_entry = _create_result_entry(res, item, task_type, ut_steps)
                        task_results.append(res_entry)
                        all_results.append(res_entry)
                        
                    # NEW: Progress indicator for large UT steps
                    if ut_steps > 1 and (i // batch_size) % 5 == 0:
                        processed = min(i + batch_size, len(items))
                        print(f"  Progress: {processed}/{len(items)} samples")
            else:
                # OPTIMIZED: Sequential Processing with cleanup
                for idx, item in enumerate(tqdm(items, desc=f"  {task_type}", leave=False)):
                    # NEW: Periodic cleanup during sequential processing
                    if ut_steps > 1 and enable_memory_optimization:
                        if idx % 10 == 0 and hasattr(experiment, '_cleanup_between_batches'):
                            experiment._cleanup_between_batches()
                    
                    res = experiment.predict_with_metrics_optimized(
                        item["prompt"], task_type, model, tokenizer, ut_steps
                    )
                    res_entry = _create_result_entry(res, item, task_type, ut_steps)
                    task_results.append(res_entry)
                    all_results.append(res_entry)

            # Logging
            _log_task_summary(task_results, task_type, ut_steps, start_time, use_wandb)

        # C. Holistic Evaluation (optional)
        if config.get("reasoning_primitives") or config.get("ENABLE_HEAVY_BENCHMARKS"):
            holistic_results = run_holistic_evaluation(model, tokenizer, config)

        # OPTIMIZED Cleanup
        del model, tokenizer
        if hasattr(experiment, '_cleanup_between_batches'):
            experiment._cleanup_between_batches()
        else:
            torch.cuda.empty_cache()
            gc.collect()
            
        # NEW: Clear caches between UT step experiments
        if hasattr(experiment, '_model_cache'):
            experiment._model_cache.clear()
        if hasattr(experiment, '_template_cache'):
            experiment._template_cache.clear()

    # Final W&B Close
    if use_wandb and run:
        wandb.finish()

    return all_results, perplexity_results, holistic_results if 'holistic_results' in locals() else []

# NEW: Update config processing to include optimizations
def _create_result_entry(result, item, task_type, ut_steps):
    """Helper to format result dictionary with timing info"""
    pred = str(result["prediction"]).strip().lower()
    target = str(item["expected_answer"]).strip().lower()

    is_correct = False
    if task_type == "p_hop":
        is_correct = pred == target
    else:
        try:
            is_correct = abs(float(pred) - float(target)) < 0.001
        except:
            is_correct = pred == target
    
    # NEW: Add performance metrics
    tokens_per_second = result["generated_tokens"] / result["generation_time"] if result["generation_time"] > 0 else 0

    return {
        "task_type": task_type,
        "difficulty": item.get("difficulty", "unknown"),
        "test_input": item["prompt"],
        "expected_answer": item["expected_answer"],
        "is_correct": is_correct,
        "test_id": generate_test_id(
            task_type, item.get("difficulty", ""), item["prompt"]
        ),
        "ut_steps": ut_steps,
        "tokens_per_second": tokens_per_second,  # NEW: Performance metric
        "efficiency_ratio": tokens_per_second / ut_steps if ut_steps > 0 else 0,  # NEW
        **result,
    }

# FIX: Add missing _log_task_summary function
def _log_task_summary(results, task_type, ut_steps, start_time, use_wandb):
    """Log summary statistics for a task"""
    if not results:
        return

    acc = sum(r["is_correct"] for r in results) / len(results)
    avg_time = sum(r["generation_time"] for r in results) / len(results)
    duration = time.time() - start_time

    print(f"    📊 Acc={acc:.2%} | Time/Sample={avg_time:.3f}s | Total={duration:.1f}s")
    
    # Calculate tokens per second
    total_tokens = sum(r.get("generated_tokens", 0) for r in results)
    total_time = sum(r.get("generation_time", 0) for r in results)
    if total_time > 0:
        tokens_per_sec = total_tokens / total_time
        print(f"    ⚡ Tokens/sec={tokens_per_sec:.1f} | Efficiency={tokens_per_sec/ut_steps:.1f} tokens/sec/UT")

    if use_wandb:
        wandb.log(
            {
                f"{task_type}/accuracy": acc,
                f"{task_type}/avg_time": avg_time,
                f"{task_type}/tokens_per_second": tokens_per_sec if total_time > 0 else 0,
                "ut_steps": ut_steps,
            }
        )