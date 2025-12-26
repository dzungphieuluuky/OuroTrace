import os
import time
import wandb
import gc
import torch
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Any, Optional, Tuple, Union

from .data_generator import create_reasoning_primitives_data, format_5_shot_prompt
from .utils import (
    save_results,
    save_simple_reasoning_results,
    save_perplexity_results,
    save_reasoning_primitives_results,
    save_heavy_benchmark_results,
    save_config,
    generate_test_id,
)
from .evaluation_analysis import (
    EnhancedOuroMetrics,
    analyze_experiment_results,
    PaperComplianceChecker,
)

# Import utilities (adjust paths as needed)
from .output_monitor import OutputQualityMonitor, ExperimentFailureException
from .data_generator import (
    create_test_datasets,
    create_perplexity_data,
    load_and_preprocess_data,
)
from .new_model import (
    OuroExperiment,
    SafeOptimizations,
)


def run_reasoning_primitives_evaluation(model, tokenizer, config: dict):
    """
    Runs the full evaluation suite:
    1. Custom Reasoning Primitives (Depth-k Var Assign) - Running locally
    2. Standard Benchmarks (QA/Math) - via lm-evaluation-harness

    Returns:
        List[Dict]: Results for each evaluation instance
    """
    reasoning_primitives_results = []

    # --- PART 1: CUSTOM REASONING PRIMITIVES ---
    print("\n" + "=" * 60)
    print("Running Reasoning Primitives (5-shot)")
    print("=" * 60)

    # Generate data
    primitives = create_reasoning_primitives_data(config = config)

    if not primitives:
        print("Warning: No reasoning primitives configured. Skipping.")
    else:
        # Determine template format from config
        template_format = config.get("reasoning_primitives", {}).get(
            "template_format", "chat"
        )

        for task_name, samples in primitives.items():
            print(f"\nTask: {task_name} ({len(samples)} samples)")
            correct = 0

            for item in tqdm(samples, desc=f"  {task_name}", leave=False):
                # Apply 5-shot formatting
                prompt = format_5_shot_prompt(
                    task_samples = samples,
                    current_sample = item, 
                    template_format = template_format
                )

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
                    generated = full_text[len(prompt) :].strip()

                # Clean up generated answer (remove extra text)
                generated = generated.split()[0] if generated.split() else generated

                # Check correctness
                is_correct = generated == item["expected_answer"]
                if is_correct:
                    correct += 1

                reasoning_primitives_results.append(
                    {
                        "task_category": "Reasoning Primitive",
                        "task_name": task_name,
                        "prompt": prompt,
                        "prediction": generated,
                        "target": item["expected_answer"],
                        "is_correct": is_correct,
                    }
                )

            accuracy = correct / len(samples) if samples else 0.0
            print(f"    Accuracy: {accuracy:.2%} ({correct}/{len(samples)})")

    return reasoning_primitives_results


def run_benchmark_evaluation(model, tokenizer, config: dict):
    """
    Runs standard benchmarks using lm-evaluation-harness.
    
    This function evaluates the model on standard NLP benchmarks including
    closed-book QA, open-book QA, and math word problems.
    
    Returns:
        List[Dict]: Results for each benchmark task
    """
    print("\n" + "=" * 60)
    print("Running Standard Benchmarks (lm-evaluation-harness)")
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
            # "coqa",
            # Math Word Problems
            "gsm8k",
            # "svamp",
            "asdiv",
        ]

        print(f"Configured tasks: {', '.join(standard_tasks)}")
        print("Note: This may take significant time and download large datasets.")
        benchmark_results = []
        # Only run if explicitly enabled (to avoid long eval times)
        if config.get("ENABLE_HEAVY_BENCHMARKS", False):
            print("\nStarting benchmark evaluation...")

            results = evaluator.simple_evaluate(
                model="hf",
                model_args=(
                    f"pretrained={model.name_or_path},"
                    "dtype=bfloat16,"
                    "trust_remote_code=True,"
                ),
                tasks=standard_tasks,
                num_fewshot=5,  # 5-shot evaluation
                batch_size=config.get("eval_batch_size", 4),
                verbosity="yes",
            )

            # Log results
            print("\nBenchmark Results:")
            for task, res in results["results"].items():
                # Extract accuracy (different tasks use different metric keys)
                res_keys = list(res.keys())
                print(f"DEBUG: {task} result keys: {res_keys}")
                acc = res.get("acc,none") or res.get("acc") or res.get("exact_match")
                try:
                    print(f"  • {task}: {acc:.2%}")
                    benchmark_results.append(
                        {
                            "task_category": "Standard Benchmark",
                            "task_name": task,
                            "is_correct": acc,  # Store accuracy directly
                        }
                    )
                except Exception as e:
                    print(f"Exception: {e}")
                    print(f" Saving all results for {task}")
                    benchmark_results.append(
                        {
                            "task_category": "Standard Benchmark",
                            "task_name": task,
                            "results": res,
                        }
                    )
        else:
            print(
                "Skipping heavy benchmarks (set ENABLE_HEAVY_BENCHMARKS = True to run)"
            )

    except ImportError:
        print("Warning: 'lm-evaluation-harness' not installed. Skipping Standard Benchmarks.")
        print("Info: Install with: pip install lm-eval")

    return benchmark_results


def run_experiment(config: dict) -> list[List[Dict]]:
    """Run batch experiment based on the provided configuration.

    Args:
        config (dict): Experiment configuration dictionary.

    Returns:
        list[List[Dict]]: Simple reasoning results, perplexity results, reasoning primitives results, benchmark results.
    """
    # 1. Initialize W&B
    use_wandb = config.get("WANDB", {}).get("enabled", False)
    run = None

    if use_wandb:
        wb_conf = config["WANDB"]
        print(f"Initializing W&B (timeout: {wb_conf.get('timeout', 30)}s)...")

        try:
            run = wandb.init(
                project=wb_conf.get("project", "ouro-trace"),
                entity=wb_conf.get("entity", None),
                name=wb_conf.get("run_name", f"run_{int(time.time())}"),
                config=config,
                mode=wb_conf.get("mode", "online"),
                settings=wandb.Settings(_disable_stats=True),
            )
            print("W&B initialized successfully")
        except Exception as e:
            print(f"Warning: W&B initialization failed: {e}. Continuing offline.")
            use_wandb = False
            run = None

    # 2. Extract and Display Configuration
    model_config = config["MODEL"]
    model_path = model_config["path"]
    ut_steps_list = config["INFERENCE_STEPS"]
    data_config = config["DATA"]
    eval_settings = config["EVAL_SETTINGS"]
    optimization_config = config.get("OPTIMIZATION", {})

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"Model Path: {model_path}")
    print(f"UT Steps to Test: {ut_steps_list}")
    print(f"Data Type: {model_config.get('dtype', torch.bfloat16)}")
    print(f"4-bit Quantization: {model_config.get('use_4bit_quant', True)}")
    print(f"Torch Compile: {model_config.get('use_torch_compile', True)}")
    print(f"Max Batch Size: {optimization_config.get('max_batch_size', 8)}")
    print(f"Max New Tokens: {optimization_config.get('max_new_tokens', 256)}")
    print(f"Batching: {optimization_config.get('enable_batch', True)}")
    print(f"Calculate Perplexity: {eval_settings.get('calculate_perplexity', True)}")
    print(f"Early Exit: {eval_settings.get('early_exit_threshold', 1.0)}")
    print(f"{'=' * 70}\n")

    # 3. Setup Experiment Handler
    experiment = OuroExperiment(
        model_path,
        dtype=config["MODEL"].get("dtype", torch.bfloat16),
        use_4bit_quant=config["MODEL"].get("use_4bit_quant", True),
        use_torch_compile=config["MODEL"].get("use_torch_compile", True),
        max_batch_size=optimization_config.get("max_batch_size", 8),
        max_new_tokens=optimization_config.get("max_new_tokens", 256),
    )

    torch.manual_seed(42)
    print(f"Random seed set to 42")

    # 4. Prepare Test Datasets
    print(f"\n{'=' * 70}")
    print(f"LOADING TEST DATASETS")
    print(f"{'=' * 70}")

    if data_config.get("load_existing", False):
        print(f"Loading from: {data_config['data_file_path']}")
        test_datasets = load_and_preprocess_data(data_config["data_file_path"])
        print(f"Loaded existing data successfully")
    else:
        print("Generating new test datasets...")
        test_datasets = create_test_datasets(data_config)
        print(f"Generated test datasets successfully")

    # Print dataset summary
    print(f"\nDataset Summary:")
    for task_type, items in test_datasets.items():
        print(f"   {task_type:12s}: {len(items):4d} samples")
    print(f"{'=' * 70}\n")

    # 5. Initialization of result storage
    perplexity_data = []
    perplexity_results = []
    simple_reasoning_results = []
    reasoning_primitives_results = []
    benchmark_results = []

    if eval_settings.get("calculate_perplexity", False):
        print(f"Preparing perplexity evaluation data...")
        raw_ppl_data = create_perplexity_data(eval_settings["ppl_num_samples"])
        perplexity_data = ["\n\n".join(raw_ppl_data)]
        print(f"Prepared {eval_settings['ppl_num_samples']} samples for PPL\n")


    # 6. Setup output directory and periodic saving
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # use .. to save results outside OuroTrace directory for easy downloading on kaggle/colab
    output_dir = f"../results_{timestamp}_UT_{'-'.join(map(str, ut_steps_list))}"
    os.makedirs(output_dir, exist_ok=True)

    # Save config ONCE at the start
    save_config(config, output_dir=output_dir, experiment=experiment)

    # save results periodically to prevent accidental collapse on long runs
    periodic_save_interval = config.get(
        "PERIODIC_SAVE_INTERVAL", 300
    )  # default 5 minutes
    last_save_time = time.time()

    try:
        # 6. Main Experiment Loop (over different UT steps)
        for ut_step_idx, ut_steps in enumerate(ut_steps_list):
            print(f"\n{'=' * 70}")
            print(
                f"EXPERIMENT {ut_step_idx + 1}/{len(ut_steps_list)}: UT Steps = {ut_steps}"
            )
            print(f"{'=' * 70}\n")

            # AUTO-OPTIMIZATION: Determine if batching should be enabled
            enable_batch = optimization_config.get("enable_batch", True)

            # Load model with specific UT steps configuration
            try:
                model, tokenizer, model_config, config_dict = (
                    experiment.load_model_with_ut_steps(ut_steps)
                )
            except Exception as e:
                print(f"Failed to load model with UT steps={ut_steps}: {e}")
                continue

            # Build task templates (only once)
            if not hasattr(experiment, "_templates_precomputed"):
                print("Building task templates...")
                experiment._build_task_templates(tokenizer)
                experiment._templates_precomputed = True
                print("Task templates built successfully\n")
                save_config(config, output_dir=output_dir, experiment=experiment)
                print("Experiment configuration saved with task templates\n")
                print()

            # A. PERPLEXITY EVALUATION
            if perplexity_data:
                print(f"{'=' * 70}")
                print(f"PERPLEXITY EVALUATION")
                print(f"{'=' * 70}\n")

                try:
                    with torch.inference_mode():
                        ppl, avg_loss = experiment.calculate_perplexity(
                            model,
                            tokenizer,
                            perplexity_data,
                            ut_steps,
                            max_length=eval_settings.get("ppl_max_length", 2048),
                            stride=eval_settings.get("ppl_stride", 512),
                        )

                    perplexity_results.append(
                        {"ut_steps": ut_steps, "perplexity": ppl, "avg_loss": avg_loss}
                    )

                    print(f"\nPerplexity Results:")
                    print(f"   Perplexity: {ppl:.4f}")
                    print(f"   Avg Loss:   {avg_loss:.4f}\n")

                    if use_wandb:
                        wandb.log(
                            {
                                "perplexity": ppl,
                                "val_loss": avg_loss,
                                "ut_steps": ut_steps,
                            }
                        )

                except Exception as e:
                    print(f"Warning: Perplexity calculation failed: {e}\n")
                now = time.time()
                if now - last_save_time >= periodic_save_interval:
                    save_results(
                        simple_reasoning_results = simple_reasoning_results,
                        perplexity_results = perplexity_results,
                        reasoning_primitives_results = reasoning_primitives_results,
                        benchmark_results = benchmark_results,
                        output_dir = output_dir,
                        overwrite = True,
                    )
                    last_save_time = now

            # B. ACCURACY & PERFORMANCE EVALUATION
            print(f"{'=' * 70}")
            print(f"ACCURACY EVALUATION")
            print(f"{'=' * 70}\n")

            for task_type, items in test_datasets.items():
                if not items:
                    print(f"Warning: Skipping {task_type} - no test items\n")
                    continue

                print(f"\n{'─' * 70}")
                print(f"Task: {task_type.upper()}")
                print(f"{'─' * 70}")
                print(f"Total Samples: {len(items)}")

                task_results = []
                task_start_time = time.time()

                # Determine optimal batch size for this task (only if batch enabled)
                batch_size = 1
                if enable_batch:
                    task_batch_limits = optimization_config.get(
                        "task_batch_size", {"n_ary": 8, "p_hop": 4, "igsm": 2}
                    )
                    batch_size = min(
                        task_batch_limits.get(task_type, 1), experiment.max_batch_size
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
                    num_batches = (
                        (len(items) + batch_size - 1) // batch_size
                        if enable_batch
                        else 1
                    )
                    print(f"Running {num_batches} batches...")

                    for batch_idx in tqdm(
                        range(0, len(items), batch_size),
                        desc=f"   {task_type}",
                        leave=False,
                        total=num_batches,
                    ):
                        batch_items = items[batch_idx : batch_idx + batch_size]
                        prompts = [item["prompt"] for item in batch_items]

                        try:
                            # Use unified predict() with list of prompts
                            with torch.inference_mode():
                                batch_outputs = experiment.predict(
                                    user_inputs=prompts,
                                    task_type=task_type,
                                    model=model,
                                    tokenizer=tokenizer,
                                    ut_steps=ut_steps,
                                )

                            # Process each output
                            for output, item in zip(batch_outputs, batch_items):
                                result_entry = _create_result_entry(
                                    output, item, task_type, ut_steps
                                )
                                task_results.append(result_entry)
                                simple_reasoning_results.append(result_entry)
                                print(
                                    pd.DataFrame([result_entry])[
                                        [
                                            "test_input",
                                            "full_response",
                                            "generated_tokens",
                                        ]
                                    ]
                                )
                                experiment.monitor_and_maybe_abort(
                                    result_entry, task_type
                                )

                        except Exception as e:
                            print(
                                f"\nWarning: Batch {batch_idx // batch_size + 1} failed: {e}"
                            )
                            # Fallback to sequential for this batch
                            for item in batch_items:
                                try:
                                    with torch.inference_mode():
                                        output = experiment.predict(
                                            user_inputs=item["prompt"],
                                            task_type=task_type,
                                            model=model,
                                            tokenizer=tokenizer,
                                            ut_steps=ut_steps,
                                        )
                                    result_entry = _create_result_entry(
                                        output, item, task_type, ut_steps
                                    )
                                    task_results.append(result_entry)
                                    simple_reasoning_results.append(result_entry)
                                    print(
                                        pd.DataFrame([result_entry])[
                                            [
                                                "test_input",
                                                "full_response",
                                                "generated_tokens",
                                            ]
                                        ]
                                    )
                                    experiment.monitor_and_maybe_abort(
                                        result_entry, task_type
                                    )
                                except Exception as e2:
                                    print(f"Warning: Item failed: {e2}")
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
                                    simple_reasoning_results.append(result_entry)
                                    print(
                                        pd.DataFrame([result_entry])[
                                            [
                                                "test_input",
                                                "full_response",
                                                "generated_tokens",
                                            ]
                                        ]
                                    )
                                    experiment.monitor_and_maybe_abort(
                                        result_entry, task_type
                                    )
                        now = time.time()
                        if now - last_save_time >= periodic_save_interval:
                            save_results(
                                simple_reasoning_results = simple_reasoning_results,
                                perplexity_results = perplexity_results,
                                reasoning_primitives_results = reasoning_primitives_results,
                                benchmark_results = benchmark_results,
                                output_dir = output_dir,
                                overwrite = True,
                            )
                            last_save_time = now

                else:
                    # SEQUENTIAL PROCESSING
                    print(
                        f"Batch size < 1 or not enough items, processing sequentially."
                    )
                    print(f"Processing {len(items)} items sequentially...")

                    for item in tqdm(items, desc=f"   {task_type}", leave=False):
                        try:
                            # Use unified predict() with single prompt
                            with torch.inference_mode():
                                output = experiment.predict(
                                    user_inputs=item["prompt"],
                                    task_type=task_type,
                                    model=model,
                                    tokenizer=tokenizer,
                                    ut_steps=ut_steps,
                                )
                            result_entry = _create_result_entry(
                                output, item, task_type, ut_steps
                            )
                            task_results.append(result_entry)
                            simple_reasoning_results.append(result_entry)
                            print(
                                pd.DataFrame([result_entry])[
                                    ["test_input", "full_response", "generated_tokens"]
                                ]
                            )
                            experiment.monitor_and_maybe_abort(result_entry, task_type)
                        except Exception as e:
                            print(f"Warning: Item failed: {e}")
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
                            simple_reasoning_results.append(result_entry)
                            print(
                                pd.DataFrame([result_entry])[
                                    ["test_input", "full_response", "generated_tokens"]
                                ]
                            )
                            experiment.monitor_and_maybe_abort(result_entry, task_type)
                        now = time.time()
                        if now - last_save_time >= periodic_save_interval:
                            save_results(
                                simple_reasoning_results = simple_reasoning_results,
                                perplexity_results = perplexity_results,
                                reasoning_primitives_results = reasoning_primitives_results,
                                benchmark_results = benchmark_results,
                                output_dir = output_dir,
                                overwrite = True,
                            )
                            last_save_time = now

                # Log and display task summary
                _log_task_summary(
                    task_results, task_type, ut_steps, task_start_time, use_wandb
                )

                # Display sample results (first 5)
                _display_sample_results(task_results, task_type)

            # C. REASONING PRIMITIVES EVALUATION (if enabled)
            if config["DATA"].get("reasoning_primitives"):
                print(f"\n{'=' * 70}")
                print(f"REASONING PRIMITIVES EVALUATION")
                print(f"{'=' * 70}\n")

                try:
                    reasoning_primitives_results = run_reasoning_primitives_evaluation(
                        model, tokenizer, config
                    )
                    for result in reasoning_primitives_results:
                        result["ut_steps"] = ut_steps
                    print(f"Reasoning primitives evaluation completed successfully\n")
                except Exception as e:
                    print(f"Warning: Reasoning primitives evaluation failed: {e}\n")
                now = time.time()
                if now - last_save_time >= periodic_save_interval:
                    save_results(
                        simple_reasoning_results = simple_reasoning_results,
                        perplexity_results = perplexity_results,
                        reasoning_primitives_results = reasoning_primitives_results,
                        benchmark_results = benchmark_results,
                        output_dir = output_dir,
                        overwrite = True,
                    )
                    last_save_time = now
            
            if config.get("ENABLE_HEAVY_BENCHMARKS"):
                print(f"\n{'=' * 70}")
                print(f"STANDARD BENCHMARKS EVALUATION")
                print(f"{'=' * 70}\n")

                try:
                    benchmark_results = run_benchmark_evaluation(
                        model, tokenizer, config
                    )
                    # Optionally, add UT steps info to each result
                    if benchmark_results:
                        for result in benchmark_results:
                            result["ut_steps"] = ut_steps
                        print(f"Standard benchmarks evaluation completed successfully\n")
                except Exception as e:
                    print(f"Warning: Standard benchmarks evaluation failed: {e}\n")
                now = time.time()
                if now - last_save_time >= periodic_save_interval:
                    save_results(
                        simple_reasoning_results = simple_reasoning_results,
                        perplexity_results = perplexity_results,
                        reasoning_primitives_results = reasoning_primitives_results,
                        benchmark_results = benchmark_results,
                        output_dir = output_dir,
                        overwrite = True,
                    )
                    last_save_time = now


            # Cleanup GPU memory
            print(f"{'=' * 70}")
            print(f"Cleaning up GPU memory...")
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            print(f"GPU memory freed successfully")
            print(f"{'=' * 70}\n")

        # 7. Final Summary
        print(f"\n{'=' * 70}")
        print(f"FINAL EXPERIMENT SUMMARY")
        print(f"{'=' * 70}\n")

        if simple_reasoning_results:
            df_all = pd.DataFrame(simple_reasoning_results)

            # Check for garbage outputs
            if "is_degenerate" in df_all.columns:
                num_garbage = df_all["is_degenerate"].sum()
                if num_garbage > 0:
                    print(
                        f"WARNING: {num_garbage} garbage/degenerate outputs detected\n"
                    )

            print("Overall Accuracy by Task Type:")
            print(f"{'─' * 70}")
            accuracy_by_task = df_all.groupby("task_type")["is_correct"].agg(
                ["mean", "count"]
            )
            accuracy_by_task.columns = ["Accuracy", "N"]
            accuracy_by_task["Accuracy"] = (accuracy_by_task["Accuracy"] * 100).round(2)
            accuracy_by_task["Accuracy"] = accuracy_by_task["Accuracy"].apply(
                lambda x: f"{x:.2f}%"
            )
            print(accuracy_by_task)
            print()

            print("Accuracy by UT Steps:")
            print(f"{'─' * 70}")
            accuracy_by_steps = df_all.groupby("ut_steps")["is_correct"].agg(
                ["mean", "count"]
            )
            accuracy_by_steps.columns = ["Accuracy", "N"]
            accuracy_by_steps["Accuracy"] = (accuracy_by_steps["Accuracy"] * 100).round(
                2
            )
            accuracy_by_steps["Accuracy"] = accuracy_by_steps["Accuracy"].apply(
                lambda x: f"{x:.2f}%"
            )
            print(accuracy_by_steps)
            print()

            print("Accuracy by Task Type and UT Steps:")
            print(f"{'─' * 70}")
            accuracy_pivot = (
                df_all.pivot_table(
                    values="is_correct",
                    index="task_type",
                    columns="ut_steps",
                    aggfunc="mean",
                )
                * 100
            )
            print(accuracy_pivot.round(2))
            print()

        if perplexity_results:
            print("Perplexity by UT Steps:")
            print(f"{'─' * 70}")
            df_ppl = pd.DataFrame(perplexity_results)
            print(df_ppl.to_string(index=False))
            print()

        # 9. Close W&B
        if use_wandb and run:
            print(f"{'=' * 70}")
            print("Finalizing W&B...")
            wandb.finish()
            print("W&B session closed")
            print(f"{'=' * 70}\n")

    except ExperimentFailureException as efe:
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT ABORTED GRACEFULLY: {efe}")
        print(f"{'=' * 70}")
        print("Finalizing W&B...")
        wandb.finish()
        print("W&B session closed")
        print(f"{'=' * 70}")
        print(f"Cleaning up GPU memory...")
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU memory freed")
        print(f"{'=' * 70}\n")


    except KeyboardInterrupt:
        print(f"\n{'=' * 70}")
        print("EXPERIMENT INTERRUPTED BY USER")
        print(f"{'=' * 70}\n")
        print("Finalizing W&B...")
        wandb.finish()
        print("W&B session closed")
        print(f"{'=' * 70}")
        print(f"Cleaning up GPU memory...")
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU memory freed")
        print(f"{'=' * 70}\n")


    # Save results to csv files
    save_results(
        simple_reasoning_results = simple_reasoning_results,
        perplexity_results = perplexity_results,
        reasoning_primitives_results = reasoning_primitives_results,
        benchmark_results = benchmark_results,
        output_dir = output_dir,
        overwrite = True,
    )

    # Save config file into yaml file
    save_config(config, output_dir=output_dir, experiment=experiment)

    return [simple_reasoning_results, perplexity_results, reasoning_primitives_results, benchmark_results]


def _create_result_entry(
    result: Dict[str, Any], item: Dict[str, Any], task_type: str, ut_steps: int
) -> Dict[str, Any]:
    """Create a standardized result entry with correctness evaluation.
    
    Args:
        result: Dictionary containing model prediction results
        item: Original test item dictionary
        task_type: Type of task (n_ary, p_hop, igsm)
        ut_steps: Number of UT steps used for inference
        
    Returns:
        Dict: Standardized result entry with all evaluation metrics
    """
    pred = str(result.get("prediction", "ERROR")).strip().lower()
    target = str(item["expected_answer"]).strip().lower()

    # Determine correctness based on task type
    is_correct = False

    if task_type == "p_hop":
        is_correct = pred == target
    elif task_type in ["n_ary", "igsm"]:
        try:
            pred_num = float(pred)
            target_num = float(target)
            is_correct = abs(pred_num - target_num) < 0.001
        except (ValueError, TypeError):
            is_correct = pred == target
    else:
        is_correct = pred == target

    return {
        "task_type": task_type,
        "difficulty": item.get("difficulty", "unknown"),
        "test_input": item["prompt"],
        "expected_answer": item["expected_answer"],
        "prediction": result.get("prediction", "ERROR"),
        "is_correct": is_correct,
        "test_id": generate_test_id(
            task_type, item.get("difficulty", ""), item["prompt"]
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
    use_wandb: bool,
) -> None:
    """Log summary statistics for a task.
    
    Args:
        results: List of result dictionaries for the task
        task_type: Type of task being evaluated
        ut_steps: Number of UT steps used
        start_time: Start time of the task evaluation
        use_wandb: Whether to log to Weights & Biases
    """
    if not results:
        print(f"\n   Warning: No results to summarize for {task_type}\n")
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
    print(f"\n{'─' * 70}")
    print(f"Summary for {task_type.upper()}")
    print(f"{'─' * 70}")
    print(f"Accuracy:            {accuracy * 100:6.2f}% ({num_correct}/{num_samples})")
    if num_degenerate > 0:
        print(
            f"Garbage Outputs:     {num_degenerate:6d} ({num_degenerate / num_samples * 100:.1f}%)"
        )
    print(f"Avg Gen Time:        {avg_gen_time:6.3f}s")
    print(f"Avg Tokens:          {avg_tokens:6.1f}")
    print(f"Total Duration:      {total_duration:6.1f}s")
    print(f"Throughput:          {throughput:6.2f} samples/sec")
    print(f"{'─' * 70}\n")

    # Log to W&B
    if use_wandb:
        try:
            wandb.log(
                {
                    f"{task_type}/accuracy": accuracy,
                    f"{task_type}/num_degenerate": num_degenerate,
                    f"{task_type}/avg_generation_time": avg_gen_time,
                    f"{task_type}/avg_tokens": avg_tokens,
                    f"{task_type}/throughput": throughput,
                    f"{task_type}/num_samples": num_samples,
                    "ut_steps": ut_steps,
                }
            )
        except Exception as e:
            print(f"   Warning: Failed to log to W&B: {e}")


def _display_sample_results(
    results: List[Dict[str, Any]], task_type: str, num_samples: int = 10
) -> None:
    """Display sample results for inspection.
    
    Args:
        results: List of result dictionaries
        task_type: Type of task being displayed
        num_samples: Number of sample results to display (default: 10)
    """
    if not results:
        return

    print(f"Sample Results for {task_type.upper()} (first {num_samples}):")
    print(f"{'─' * 70}")

    df_sample = pd.DataFrame(results).head(num_samples)
    display_cols = ["test_input", "full_response", "generated_tokens", "is_correct"]

    # Add degenerate flag if present
    if "is_degenerate" in df_sample.columns:
        display_cols.append("is_degenerate")

    # Truncate long text for display
    for col in ["test_input", "full_response", "generated_tokens"]:
        if col in df_sample.columns:
            df_sample[col] = df_sample[col].astype(str).str[:60]

    print(df_sample[display_cols].to_string(index=False))
    print()