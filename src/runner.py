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
    Run experiment with batching support and W&B logging.
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

    # 2. Extract Config
    model_path = config["MODEL"]["path"]
    ut_steps_list = config["INFERENCE_STEPS"]
    data_config = config["DATA"]
    eval_settings = config["EVAL_SETTINGS"]

    # 3. Setup Experiment
    experiment = OuroBatchExperiment(
        model_path,
        dtype=config["MODEL"]["dtype"],
        use_4bit_quant=config["MODEL"].get("use_4bit_quant", True),
        use_torch_compile=config["MODEL"].get("use_torch_compile", True),
        max_batch_size=config.get("OPTIMIZATION", {}).get("max_batch_size", 4),
        max_new_tokens=config.get("OPTIMIZATION", {}).get("max_new_token", 1024),
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

    # 5. Main Loop
    for ut_steps in ut_steps_list:
        print(f"\n{'=' * 60}\n🧪 EXPERIMENT: UT Steps = {ut_steps}\n{'=' * 60}")

        # Load fresh model
        model, tokenizer, _, _ = experiment.load_model_with_ut_steps(
            ut_steps, eval_settings["early_exit_threshold"]
        )

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

        # B. Accuracy & Time
        enable_batch = config.get("OPTIMIZATION", {}).get("enable_batch", True)

        for task_type, items in test_datasets.items():
            print(f"\n📝 Task: {task_type} ({len(items)} samples)")
            task_results = []
            start_time = time.time()

            # Determine batch strategy
            batch_size = 1
            if enable_batch:
                limits = {"n_ary": 8, "p_hop": 4, "igsm": 2}
                batch_size = min(limits.get(task_type, 1), experiment.max_batch_size)

            # Process Loop
            if batch_size > 1 and len(items) >= 2:
                # Batched Processing
                for i in range(0, len(items), batch_size):
                    batch_items = items[i : i + batch_size]
                    prompts = [item["prompt"] for item in batch_items]

                    batch_out = experiment.batch_predict_with_metrics(
                        prompts, task_type, model, tokenizer, ut_steps
                    )

                    for res, item in zip(batch_out, batch_items):
                        res_entry = _create_result_entry(res, item, task_type, ut_steps)
                        task_results.append(res_entry)
                        all_results.append(res_entry)
            else:
                # Sequential Processing
                for item in tqdm(items, desc=f"  {task_type}", leave=False):
                    res = experiment.predict_with_metrics_optimized(
                        item["prompt"], task_type, model, tokenizer, ut_steps
                    )
                    res_entry = _create_result_entry(res, item, task_type, ut_steps)
                    task_results.append(res_entry)
                    all_results.append(res_entry)

            # Logging
            _log_task_summary(task_results, task_type, ut_steps, start_time, use_wandb)

        # C. Holistic Evaluation
        if config.get("reasoning_primitives") or config.get("ENABLE_HEAVY_BENCHMARKS"):
            holistic_results = run_holistic_evaluation(model, tokenizer, config)
            # You can process holistic_results further or save them separately if needed

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # Final W&B Close
    if use_wandb and run:
        wandb.finish()

    return all_results, perplexity_results

def _create_result_entry(result, item, task_type, ut_steps):
    """Helper to format result dictionary"""
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
        **result,
    }

def _log_task_summary(results, task_type, ut_steps, start_time, use_wandb):
    if not results:
        return

    acc = sum(r["is_correct"] for r in results) / len(results)
    avg_time = sum(r["generation_time"] for r in results) / len(results)
    duration = time.time() - start_time

    print(f"    📊 Acc={acc:.2%} | Time/Sample={avg_time:.3f}s | Total={duration:.1f}s")

    if use_wandb:
        wandb.log(
            {
                f"{task_type}/accuracy": acc,
                f"{task_type}/avg_time": avg_time,
                "ut_steps": ut_steps,
            }
        )