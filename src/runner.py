"""
Experiment runner for evaluating language models with different inference configurations.

This module orchestrates model evaluation across multiple tasks including:
- Simple reasoning tasks (n-ary, p-hop, igsm)
- Perplexity evaluation
- Reasoning primitives
- Standard NLP benchmarks
"""

import os
import time
import gc
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import torch
import pandas as pd
import wandb
from tqdm.auto import tqdm

from .data_generator import (
    create_reasoning_primitives_data,
    format_5_shot_prompt,
    create_test_datasets,
    create_perplexity_data,
    load_and_preprocess_data,
)
from .utils import (
    save_results,
    save_config,
    generate_test_id,
)
from .evaluation_analysis import (
    EnhancedOuroMetrics,
    analyze_experiment_results,
    PaperComplianceChecker,
)
from .output_monitor import OutputQualityMonitor, ExperimentFailureException
from .new_model import OuroExperiment, SafeOptimizations


# =============================================================================
# Constants
# =============================================================================

RESULT_COLUMNS_TO_DISPLAY = ["test_input", "full_response", "generated_tokens"]
DEFAULT_PERIODIC_SAVE_INTERVAL = 300  # 5 minutes
MAX_TEXT_DISPLAY_LENGTH = 60


# =============================================================================
# Evaluation Functions
# =============================================================================

def run_reasoning_primitives_evaluation(
    model, tokenizer, config: Dict
) -> List[Dict]:
    """
    Evaluate model on custom reasoning primitives (e.g., Depth-k Variable Assignment).
    
    Args:
        model: The language model to evaluate
        tokenizer: Tokenizer for the model
        config: Experiment configuration dictionary
        
    Returns:
        List of result dictionaries for each evaluation sample
    """
    print("\n" + "=" * 60)
    print("Running Reasoning Primitives (5-shot)")
    print("=" * 60)
    
    primitives = create_reasoning_primitives_data(config=config)
    
    if not primitives:
        print("Warning: No reasoning primitives configured. Skipping.")
        return []
    
    template_format = config.get("reasoning_primitives", {}).get(
        "template_format", "chat"
    )
    
    all_results = []
    
    for task_name, samples in primitives.items():
        print(f"\nTask: {task_name} ({len(samples)} samples)")
        
        task_results = _evaluate_primitive_task(
            model, tokenizer, task_name, samples, template_format
        )
        all_results.extend(task_results)
        
        # Display task accuracy
        accuracy = sum(r["is_correct"] for r in task_results) / len(task_results)
        print(f"    Accuracy: {accuracy:.2%} ({sum(r['is_correct'] for r in task_results)}/{len(task_results)})")
    
    return all_results


def _evaluate_primitive_task(
    model, tokenizer, task_name: str, samples: List[Dict], 
    template_format: str
) -> List[Dict]:
    """Evaluate a single reasoning primitive task."""
    results = []
    
    for sample in tqdm(samples, desc=f"  {task_name}", leave=False):
        prompt = format_5_shot_prompt(
            task_samples=samples,
            current_sample=sample,
            template_format=template_format
        )
        
        prediction = _generate_primitive_answer(model, tokenizer, prompt)
        is_correct = prediction == sample["expected_answer"]
        
        results.append({
            "task_category": "Reasoning Primitive",
            "task_name": task_name,
            "prompt": prompt,
            "prediction": prediction,
            "target": sample["expected_answer"],
            "is_correct": is_correct,
        })
    
    return results


def _generate_primitive_answer(model, tokenizer, prompt: str) -> str:
    """Generate answer for a reasoning primitive using greedy decoding."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer using various markers
    for marker in ["Answer:", "Response:", "Assistant:"]:
        if marker in full_text:
            answer = full_text.split(marker)[-1].strip()
            return answer.split()[0] if answer.split() else answer
    
    # Fallback: take text after prompt
    generated = full_text[len(prompt):].strip()
    return generated.split()[0] if generated.split() else generated


def run_benchmark_evaluation(model, tokenizer, config: Dict) -> List[Dict]:
    """
    Evaluate model on standard NLP benchmarks using lm-evaluation-harness.
    
    Args:
        model: The language model to evaluate
        tokenizer: Tokenizer for the model
        config: Experiment configuration dictionary
        
    Returns:
        List of benchmark result dictionaries
    """
    print("\n" + "=" * 60)
    print("Running Standard Benchmarks (lm-evaluation-harness)")
    print("=" * 60)
    
    if not config.get("ENABLE_HEAVY_BENCHMARKS", False):
        print("Skipping heavy benchmarks (set ENABLE_HEAVY_BENCHMARKS = True to run)")
        return []
    
    try:
        from lm_eval import evaluator
    except ImportError:
        print("Warning: 'lm-evaluation-harness' not installed. Skipping.")
        print("Info: Install with: pip install lm-eval")
        return []
    
    standard_tasks = [
        "triviaqa", "nq_open", "webqs",  # Closed Book QA
        "squadv2", "drop",                # Open Book QA
        "gsm8k", "asdiv",                 # Math Word Problems
    ]
    
    print(f"Configured tasks: {', '.join(standard_tasks)}")
    print("Note: This may take significant time and download large datasets.\n")
    
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=(
            f"pretrained={model.name_or_path},"
            "dtype=bfloat16,"
            "trust_remote_code=True,"
        ),
        tasks=standard_tasks,
        num_fewshot=5,
        batch_size=config.get("eval_batch_size", 4),
        verbosity="yes",
    )
    
    return _parse_benchmark_results(results)


def _parse_benchmark_results(results: Dict) -> List[Dict]:
    """Parse raw benchmark results into standardized format."""
    benchmark_results = []
    
    print("\nBenchmark Results:")
    for task, res in results["results"].items():
        accuracy = res.get("acc,none") or res.get("acc") or res.get("exact_match")
        
        try:
            print(f"  • {task}: {accuracy:.2%}")
            benchmark_results.append({
                "task_category": "Standard Benchmark",
                "task_name": task,
                "is_correct": accuracy,
            })
        except Exception as e:
            print(f"Warning: Could not parse {task} results: {e}")
            benchmark_results.append({
                "task_category": "Standard Benchmark",
                "task_name": task,
                "results": res,
            })
    
    return benchmark_results


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment(config: Dict) -> List[List[Dict]]:
    """
    Execute the complete experiment pipeline.
    
    Args:
        config: Experiment configuration dictionary containing:
            - MODEL: Model configuration (path, dtype, quantization)
            - INFERENCE_STEPS: List of UT steps to evaluate
            - DATA: Dataset configuration
            - EVAL_SETTINGS: Evaluation settings
            - OPTIMIZATION: Performance optimization settings
            - WANDB: Weights & Biases logging configuration
            
    Returns:
        List containing [simple_reasoning_results, perplexity_results, 
                        reasoning_primitives_results, benchmark_results]
    """
    # Initialize experiment
    wandb_run = _initialize_wandb(config)
    experiment = _setup_experiment(config)
    output_dir = _create_output_directory(config)
    
    # Prepare data
    test_datasets = _load_test_datasets(config)
    perplexity_data = _prepare_perplexity_data(config)
    
    # Initialize result storage
    results = ResultCollector()
    periodic_saver = PeriodicSaver(
        output_dir=output_dir,
        interval=config.get("PERIODIC_SAVE_INTERVAL", DEFAULT_PERIODIC_SAVE_INTERVAL)
    )
    
    # Save initial configuration
    save_config(config, output_dir=output_dir, experiment=experiment)
    
    try:
        # Main evaluation loop across different UT steps
        for ut_step_idx, ut_steps in enumerate(config["INFERENCE_STEPS"]):
            _print_experiment_header(ut_step_idx, ut_steps, config)
            
            # Load model with specific UT configuration
            model, tokenizer = _load_model_for_ut_steps(experiment, ut_steps)
            
            # Build task templates (only once)
            _build_task_templates_once(experiment, tokenizer, output_dir, config)
            
            # Run evaluations
            _run_perplexity_evaluation(
                experiment, model, tokenizer, perplexity_data, 
                ut_steps, config, results, periodic_saver
            )
            
            _run_accuracy_evaluation(
                experiment, model, tokenizer, test_datasets,
                ut_steps, config, results, periodic_saver
            )
            
            _run_reasoning_primitives(
                model, tokenizer, ut_steps, config, results, periodic_saver
            )
            
            _run_standard_benchmarks(
                model, tokenizer, ut_steps, config, results, periodic_saver
            )
            
            # Cleanup GPU memory
            _cleanup_gpu_memory(model, tokenizer)
        
        # Display final summary
        _display_final_summary(results)
        
    except (ExperimentFailureException, KeyboardInterrupt) as e:
        _handle_experiment_interruption(e, model, tokenizer)
    
    finally:
        # Save final results
        results.save_all(output_dir)
        save_config(config, output_dir=output_dir, experiment=experiment)
        
        if wandb_run:
            wandb.finish()
    
    return results.to_list()


# =============================================================================
# Helper Classes
# =============================================================================

class ResultCollector:
    """Manages collection of all experiment results."""
    
    def __init__(self):
        self.simple_reasoning = []
        self.perplexity = []
        self.reasoning_primitives = []
        self.benchmarks = []
    
    def add_simple_reasoning(self, result: Dict):
        """Add a simple reasoning result."""
        self.simple_reasoning.append(result)
    
    def add_perplexity(self, result: Dict):
        """Add a perplexity result."""
        self.perplexity.append(result)
    
    def add_reasoning_primitive(self, result: Dict):
        """Add a reasoning primitive result."""
        self.reasoning_primitives.append(result)
    
    def add_benchmark(self, result: Dict):
        """Add a benchmark result."""
        self.benchmarks.append(result)
    
    def save_all(self, output_dir: str):
        """Save all results to CSV files."""
        save_results(
            simple_reasoning_results=self.simple_reasoning,
            perplexity_results=self.perplexity,
            reasoning_primitives_results=self.reasoning_primitives,
            benchmark_results=self.benchmarks,
            output_dir=output_dir,
            overwrite=True,
        )
    
    def to_list(self) -> List[List[Dict]]:
        """Return results as list of lists."""
        return [
            self.simple_reasoning,
            self.perplexity,
            self.reasoning_primitives,
            self.benchmarks
        ]


class PeriodicSaver:
    """Handles periodic saving of results during long experiments."""
    
    def __init__(self, output_dir: str, interval: int):
        self.output_dir = output_dir
        self.interval = interval
        self.last_save_time = time.time()
    
    def maybe_save(self, results: ResultCollector):
        """Save results if enough time has elapsed."""
        current_time = time.time()
        if current_time - self.last_save_time >= self.interval:
            results.save_all(self.output_dir)
            self.last_save_time = current_time


# =============================================================================
# Setup and Initialization Helpers
# =============================================================================

def _initialize_wandb(config: Dict) -> Optional[Any]:
    """Initialize Weights & Biases logging if enabled."""
    if not config.get("WANDB", {}).get("enabled", False):
        return None
    
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
        return run
    except Exception as e:
        print(f"Warning: W&B initialization failed: {e}. Continuing offline.")
        return None


def _setup_experiment(config: Dict) -> OuroExperiment:
    """Initialize the experiment handler."""
    _display_configuration(config)
    
    model_config = config["MODEL"]
    optimization_config = config.get("OPTIMIZATION", {})
    
    return OuroExperiment(
        model_config["path"],
        dtype=model_config.get("dtype", torch.bfloat16),
        use_4bit_quant=model_config.get("use_4bit_quant", True),
        use_torch_compile=model_config.get("use_torch_compile", True),
        max_batch_size=optimization_config.get("max_batch_size", 8),
        max_new_tokens=optimization_config.get("max_new_tokens", 256),
    )


def _display_configuration(config: Dict):
    """Display experiment configuration."""
    model_config = config["MODEL"]
    optimization_config = config.get("OPTIMIZATION", {})
    eval_settings = config["EVAL_SETTINGS"]
    
    print(f"\n{'=' * 70}")
    print("EXPERIMENT CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"Model Path/Name: {model_config['path']}")
    print(f"UT Steps to Test: {config['INFERENCE_STEPS']}")
    print(f"Data Type: {model_config.get('dtype', torch.bfloat16)}")
    print()
    print(f"4-bit Quantization: {model_config.get('use_4bit_quant', True)}")
    print(f"Torch Compile: {model_config.get('use_torch_compile', True)}")
    print(f"Batching: {optimization_config.get('enable_batch', True)}")
    print(f"Max Batch Size: {optimization_config.get('max_batch_size', 8)}")
    print(f"Max New Tokens: {optimization_config.get('max_new_tokens', 256)}")
    print()
    print(f"Calculate Perplexity: {eval_settings.get('calculate_perplexity', True)}")
    print(f"Early Exit: {eval_settings.get('early_exit_threshold', 1.0)}")
    print(f"{'=' * 70}\n")


def _create_output_directory(config: Dict) -> str:
    """Create output directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ut_steps_str = '-'.join(map(str, config["INFERENCE_STEPS"]))
    output_dir = f"../results_{timestamp}_UT_{ut_steps_str}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _load_test_datasets(config: Dict) -> Dict[str, List[Dict]]:
    """Load or generate test datasets."""
    print(f"\n{'=' * 70}")
    print("LOADING TEST DATASETS")
    print(f"{'=' * 70}")
    
    data_config = config["DATA"]
    
    if data_config.get("load_existing", False):
        print(f"Loading from: {data_config['data_file_path']}")
        test_datasets = load_and_preprocess_data(data_config["data_file_path"])
        print("Loaded existing data successfully")
    else:
        print("Generating new test datasets...")
        test_datasets = create_test_datasets(data_config)
        print("Generated test datasets successfully")
    
    # Display dataset summary
    print("\nDataset Summary:")
    for task_type, items in test_datasets.items():
        print(f"   {task_type:12s}: {len(items):4d} samples")
    print(f"{'=' * 70}\n")
    
    return test_datasets


def _prepare_perplexity_data(config: Dict) -> List[str]:
    """Prepare data for perplexity evaluation."""
    eval_settings = config["EVAL_SETTINGS"]
    
    if not eval_settings.get("calculate_perplexity", False):
        return []
    
    print("Preparing perplexity evaluation data...")
    raw_ppl_data = create_perplexity_data(eval_settings["ppl_num_samples"])
    print(f"Prepared {eval_settings['ppl_num_samples']} samples for PPL\n")
    
    return ["\n\n".join(raw_ppl_data)]


def _build_task_templates_once(
    experiment: OuroExperiment, tokenizer, output_dir: str, config: Dict
):
    """Build task templates only once and save configuration."""
    if hasattr(experiment, "_templates_precomputed"):
        return
    
    print("Building task templates...")
    experiment._build_task_templates(tokenizer)
    experiment._templates_precomputed = True
    print("Task templates built successfully\n")
    
    save_config(config, output_dir=output_dir, experiment=experiment)
    print("Experiment configuration saved with task templates\n")


# =============================================================================
# Evaluation Runners
# =============================================================================

def _run_perplexity_evaluation(
    experiment: OuroExperiment,
    model, tokenizer,
    perplexity_data: List[str],
    ut_steps: int,
    config: Dict,
    results: ResultCollector,
    saver: PeriodicSaver
):
    """Run perplexity evaluation if configured."""
    if not perplexity_data:
        return
    
    print(f"{'=' * 70}")
    print("PERPLEXITY EVALUATION")
    print(f"{'=' * 70}\n")
    
    eval_settings = config["EVAL_SETTINGS"]
    
    try:
        with torch.inference_mode():
            ppl, avg_loss = experiment.calculate_perplexity(
                model, tokenizer, perplexity_data, ut_steps,
                max_length=eval_settings.get("ppl_max_length", 2048),
                stride=eval_settings.get("ppl_stride", 512),
            )
        
        results.add_perplexity({
            "ut_steps": ut_steps,
            "perplexity": ppl,
            "avg_loss": avg_loss
        })
        
        print(f"\nPerplexity Results:")
        print(f"   Perplexity: {ppl:.4f}")
        print(f"   Avg Loss:   {avg_loss:.4f}\n")
        
        if config.get("WANDB", {}).get("enabled", False):
            wandb.log({
                "perplexity": ppl,
                "val_loss": avg_loss,
                "ut_steps": ut_steps,
            })
        
        saver.maybe_save(results)
        
    except Exception as e:
        print(f"Warning: Perplexity calculation failed: {e}\n")


def _run_accuracy_evaluation(
    experiment: OuroExperiment,
    model, tokenizer,
    test_datasets: Dict[str, List[Dict]],
    ut_steps: int,
    config: Dict,
    results: ResultCollector,
    saver: PeriodicSaver
):
    """Run accuracy evaluation on all test datasets."""
    print(f"{'=' * 70}")
    print("ACCURACY EVALUATION")
    print(f"{'=' * 70}\n")
    
    optimization_config = config.get("OPTIMIZATION", {})
    use_wandb = config.get("WANDB", {}).get("enabled", False)
    
    for task_type, items in test_datasets.items():
        if not items:
            print(f"Warning: Skipping {task_type} - no test items\n")
            continue
        
        task_results = _evaluate_task(
            experiment, model, tokenizer, task_type, items,
            ut_steps, optimization_config, results, saver
        )
        
        _log_task_summary(task_results, task_type, ut_steps, use_wandb)
        _display_sample_results(task_results, task_type)


def _evaluate_task(
    experiment: OuroExperiment,
    model, tokenizer,
    task_type: str,
    items: List[Dict],
    ut_steps: int,
    optimization_config: Dict,
    results: ResultCollector,
    saver: PeriodicSaver
) -> List[Dict]:
    """Evaluate model on a single task type."""
    print(f"\n{'─' * 70}")
    print(f"Task: {task_type.upper()}")
    print(f"{'─' * 70}")
    print(f"Total Samples: {len(items)}")
    
    # Determine batch size
    enable_batch = optimization_config.get("enable_batch", True)
    batch_size = _determine_batch_size(
        task_type, enable_batch, optimization_config, experiment
    )
    
    print(f"Batch Size: {batch_size}")
    print(f"Strategy: {'Batched' if batch_size > 1 else 'Sequential'} Processing\n")
    
    task_start_time = time.time()
    task_results = []
    
    if batch_size > 1 and len(items) >= batch_size:
        task_results = _evaluate_batched(
            experiment, model, tokenizer, task_type, items,
            batch_size, ut_steps, results, saver
        )
    else:
        task_results = _evaluate_sequential(
            experiment, model, tokenizer, task_type, items,
            ut_steps, results, saver
        )
    
    return task_results


def _determine_batch_size(
    task_type: str,
    enable_batch: bool,
    optimization_config: Dict,
    experiment: OuroExperiment
) -> int:
    """Determine optimal batch size for a task."""
    if not enable_batch:
        return 1
    
    task_batch_limits = optimization_config.get(
        "task_batch_size",
        {"n_ary": 8, "p_hop": 4, "igsm": 2}
    )
    
    return min(
        task_batch_limits.get(task_type, 1),
        experiment.max_batch_size
    )


def _evaluate_batched(
    experiment: OuroExperiment,
    model, tokenizer,
    task_type: str,
    items: List[Dict],
    batch_size: int,
    ut_steps: int,
    results: ResultCollector,
    saver: PeriodicSaver
) -> List[Dict]:
    """Evaluate items in batches."""
    num_batches = (len(items) + batch_size - 1) // batch_size
    print(f"Running {num_batches} batches...")
    
    task_results = []
    
    for batch_idx in tqdm(range(0, len(items), batch_size), 
                          desc=f"   {task_type}", leave=False, total=num_batches):
        batch_items = items[batch_idx:batch_idx + batch_size]
        prompts = [item["prompt"] for item in batch_items]
        
        try:
            with torch.inference_mode():
                batch_outputs = experiment.predict(
                    user_inputs=prompts,
                    task_type=task_type,
                    model=model,
                    tokenizer=tokenizer,
                    ut_steps=ut_steps,
                )
            
            for output, item in zip(batch_outputs, batch_items):
                result_entry = _create_result_entry(output, item, task_type, ut_steps)
                task_results.append(result_entry)
                results.add_simple_reasoning(result_entry)
                _display_result_entry(result_entry)
                experiment.monitor_and_maybe_abort(result_entry, task_type)
        
        except Exception as e:
            print(f"\nWarning: Batch {batch_idx // batch_size + 1} failed: {e}")
            # Fallback to sequential processing for this batch
            _process_failed_batch(
                experiment, model, tokenizer, task_type, batch_items,
                ut_steps, task_results, results
            )
        
        saver.maybe_save(results)
    
    return task_results


def _evaluate_sequential(
    experiment: OuroExperiment,
    model, tokenizer,
    task_type: str,
    items: List[Dict],
    ut_steps: int,
    results: ResultCollector,
    saver: PeriodicSaver
) -> List[Dict]:
    """Evaluate items sequentially."""
    print(f"Processing {len(items)} items sequentially...")
    
    task_results = []
    
    for item in tqdm(items, desc=f"   {task_type}", leave=False):
        try:
            with torch.inference_mode():
                output = experiment.predict(
                    user_inputs=item["prompt"],
                    task_type=task_type,
                    model=model,
                    tokenizer=tokenizer,
                    ut_steps=ut_steps,
                )
            
            result_entry = _create_result_entry(output, item, task_type, ut_steps)
            task_results.append(result_entry)
            results.add_simple_reasoning(result_entry)
            _display_result_entry(result_entry)
            experiment.monitor_and_maybe_abort(result_entry, task_type)
        
        except Exception as e:
            print(f"Warning: Item failed: {e}")
            result_entry = _create_error_result_entry(e, item, task_type, ut_steps)
            task_results.append(result_entry)
            results.add_simple_reasoning(result_entry)
            _display_result_entry(result_entry)
            experiment.monitor_and_maybe_abort(result_entry, task_type)
        
        saver.maybe_save(results)
    
    return task_results


def _process_failed_batch(
    experiment: OuroExperiment,
    model, tokenizer,
    task_type: str,
    batch_items: List[Dict],
    ut_steps: int,
    task_results: List[Dict],
    results: ResultCollector
):
    """Process a failed batch sequentially."""
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
            result_entry = _create_result_entry(output, item, task_type, ut_steps)
        except Exception as e2:
            print(f"Warning: Item failed: {e2}")
            result_entry = _create_error_result_entry(e2, item, task_type, ut_steps)
        
        task_results.append(result_entry)
        results.add_simple_reasoning(result_entry)
        _display_result_entry(result_entry)
        experiment.monitor_and_maybe_abort(result_entry, task_type)


def _run_reasoning_primitives(
    model, tokenizer, ut_steps: int, config: Dict,
    results: ResultCollector, saver: PeriodicSaver
):
    """Run reasoning primitives evaluation if configured."""
    if not config["DATA"].get("reasoning_primitives"):
        return
    
    print(f"\n{'=' * 70}")
    print("REASONING PRIMITIVES EVALUATION")
    print(f"{'=' * 70}\n")
    
    try:
        primitive_results = run_reasoning_primitives_evaluation(
            model, tokenizer, config
        )
        
        for result in primitive_results:
            result["ut_steps"] = ut_steps
            results.add_reasoning_primitive(result)
        
        print("Reasoning primitives evaluation completed successfully\n")
        saver.maybe_save(results)
    
    except Exception as e:
        print(f"Warning: Reasoning primitives evaluation failed: {e}\n")


def _run_standard_benchmarks(
    model, tokenizer, ut_steps: int, config: Dict,
    results: ResultCollector, saver: PeriodicSaver
):
    """Run standard benchmarks if configured."""
    if not config.get("ENABLE_HEAVY_BENCHMARKS"):
        return
    
    print(f"\n{'=' * 70}")
    print("STANDARD BENCHMARKS EVALUATION")
    print(f"{'=' * 70}\n")
    
    try:
        benchmark_results = run_benchmark_evaluation(model, tokenizer, config)
        
        for result in benchmark_results:
            result["ut_steps"] = ut_steps
            results.add_benchmark(result)
        
        print("Standard benchmarks evaluation completed successfully\n")
        saver.maybe_save(results)
    
    except Exception as e:
        print(f"Warning: Standard benchmarks evaluation failed: {e}\n")


# =============================================================================
# Result Processing Helpers
# =============================================================================

def _create_result_entry(
    result: Dict[str, Any],
    item: Dict[str, Any],
    task_type: str,
    ut_steps: int
) -> Dict[str, Any]:
    """
    Create a standardized result entry with correctness evaluation.
    
    Args:
        result: Dictionary containing model prediction results
        item: Original test item dictionary
        task_type: Type of task (n_ary, p_hop, igsm)
        ut_steps: Number of UT steps used for inference
        
    Returns:
        Standardized result entry with all evaluation metrics
    """
    pred = str(result.get("prediction", "ERROR")).strip().lower()
    target = str(item["expected_answer"]).strip().lower()
    
    is_correct = _check_correctness(pred, target, task_type)
    
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


def _create_error_result_entry(
    error: Exception,
    item: Dict[str, Any],
    task_type: str,
    ut_steps: int
) -> Dict[str, Any]:
    """
    Create a result entry for a failed prediction.
    
    Args:
        error: The exception that occurred
        item: Original test item dictionary
        task_type: Type of task
        ut_steps: Number of UT steps used
        
    Returns:
        Error result entry
    """
    error_result = {
        "prediction": "ERROR",
        "full_response": str(error),
        "generation_time": 0,
        "generated_tokens": 0,
        "input_tokens": 0,
        "is_degenerate": False,
    }
    return _create_result_entry(error_result, item, task_type, ut_steps)


def _check_correctness(pred: str, target: str, task_type: str) -> bool:
    """
    Determine if prediction is correct based on task type.
    
    Args:
        pred: Predicted answer (lowercase, stripped)
        target: Expected answer (lowercase, stripped)
        task_type: Type of task (determines comparison method)
        
    Returns:
        True if prediction is correct, False otherwise
    """
    if task_type == "p_hop":
        return pred == target
    
    elif task_type in ["n_ary", "igsm"]:
        try:
            pred_num = float(pred)
            target_num = float(target)
            return abs(pred_num - target_num) < 0.001
        except (ValueError, TypeError):
            return pred == target
    
    else:
        return pred == target


def _display_result_entry(result_entry: Dict[str, Any]):
    """
    Display a single result entry in a readable format.
    
    Args:
        result_entry: Result dictionary to display
    """
    df = pd.DataFrame([result_entry])
    print(df[RESULT_COLUMNS_TO_DISPLAY])


def _log_task_summary(
    results: List[Dict[str, Any]],
    task_type: str,
    ut_steps: int,
    use_wandb: bool,
    start_time: Optional[float] = None
):
    """
    Log and display summary statistics for a task.
    
    Args:
        results: List of result dictionaries for the task
        task_type: Type of task being evaluated
        ut_steps: Number of UT steps used
        use_wandb: Whether to log to Weights & Biases
        start_time: Start time of task evaluation (optional)
    """
    if not results:
        print(f"\n   Warning: No results to summarize for {task_type}\n")
        return
    
    # Calculate metrics
    metrics = _calculate_task_metrics(results, start_time)
    
    # Print summary
    _print_task_summary(task_type, metrics)
    
    # Log to W&B
    if use_wandb:
        _log_to_wandb(task_type, metrics, ut_steps)


def _calculate_task_metrics(
    results: List[Dict[str, Any]],
    start_time: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate summary metrics for a task.
    
    Args:
        results: List of result dictionaries
        start_time: Start time of task evaluation (optional)
        
    Returns:
        Dictionary of calculated metrics
    """
    num_samples = len(results)
    num_correct = sum(r["is_correct"] for r in results)
    num_degenerate = sum(r.get("is_degenerate", False) for r in results)
    
    accuracy = num_correct / num_samples if num_samples > 0 else 0.0
    
    total_gen_time = sum(r.get("generation_time", 0) for r in results)
    avg_gen_time = total_gen_time / num_samples if num_samples > 0 else 0.0
    
    total_tokens = sum(r.get("generated_tokens", 0) for r in results)
    avg_tokens = total_tokens / num_samples if num_samples > 0 else 0.0
    
    total_duration = time.time() - start_time if start_time else 0.0
    throughput = num_samples / total_duration if total_duration > 0 else 0.0
    
    return {
        "num_samples": num_samples,
        "num_correct": num_correct,
        "num_degenerate": num_degenerate,
        "accuracy": accuracy,
        "avg_gen_time": avg_gen_time,
        "avg_tokens": avg_tokens,
        "total_duration": total_duration,
        "throughput": throughput,
    }


def _print_task_summary(task_type: str, metrics: Dict[str, float]):
    """
    Print formatted task summary.
    
    Args:
        task_type: Type of task
        metrics: Dictionary of calculated metrics
    """
    print(f"\n{'─' * 70}")
    print(f"Summary for {task_type.upper()}")
    print(f"{'─' * 70}")
    print(f"Accuracy:            {metrics['accuracy'] * 100:6.2f}% "
          f"({metrics['num_correct']:.0f}/{metrics['num_samples']:.0f})")
    
    if metrics['num_degenerate'] > 0:
        degenerate_pct = metrics['num_degenerate'] / metrics['num_samples'] * 100
        print(f"Garbage Outputs:     {metrics['num_degenerate']:6.0f} ({degenerate_pct:.1f}%)")
    
    print(f"Avg Gen Time:        {metrics['avg_gen_time']:6.3f}s")
    print(f"Avg Tokens:          {metrics['avg_tokens']:6.1f}")
    
    if metrics['total_duration'] > 0:
        print(f"Total Duration:      {metrics['total_duration']:6.1f}s")
        print(f"Throughput:          {metrics['throughput']:6.2f} samples/sec")
    
    print(f"{'─' * 70}\n")


def _log_to_wandb(task_type: str, metrics: Dict[str, float], ut_steps: int):
    """
    Log metrics to Weights & Biases.
    
    Args:
        task_type: Type of task
        metrics: Dictionary of metrics to log
        ut_steps: Number of UT steps used
    """
    try:
        wandb.log({
            f"{task_type}/accuracy": metrics['accuracy'],
            f"{task_type}/num_degenerate": metrics['num_degenerate'],
            f"{task_type}/avg_generation_time": metrics['avg_gen_time'],
            f"{task_type}/avg_tokens": metrics['avg_tokens'],
            f"{task_type}/throughput": metrics['throughput'],
            f"{task_type}/num_samples": metrics['num_samples'],
            "ut_steps": ut_steps,
        })
    except Exception as e:
        print(f"   Warning: Failed to log to W&B: {e}")


def _display_sample_results(
    results: List[Dict[str, Any]],
    task_type: str,
    num_samples: int = 10
):
    """
    Display sample results for inspection.
    
    Args:
        results: List of result dictionaries
        task_type: Type of task being displayed
        num_samples: Number of sample results to display
    """
    if not results:
        return
    
    print(f"Sample Results for {task_type.upper()} (first {num_samples}):")
    print(f"{'─' * 70}")
    
    df_sample = pd.DataFrame(results).head(num_samples)
    
    display_cols = ["test_input", "full_response", "generated_tokens", "is_correct"]
    if "is_degenerate" in df_sample.columns:
        display_cols.append("is_degenerate")
    
    # Truncate long text for display
    for col in ["test_input", "full_response", "generated_tokens"]:
        if col in df_sample.columns:
            df_sample[col] = df_sample[col].astype(str).str[:MAX_TEXT_DISPLAY_LENGTH]
    
    print(df_sample[display_cols].to_string(index=False))
    print()


# =============================================================================
# Utility Helpers
# =============================================================================

def _print_experiment_header(ut_step_idx: int, ut_steps: int, config: Dict):
    """Print header for current experiment iteration."""
    total_steps = len(config["INFERENCE_STEPS"])
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT {ut_step_idx + 1}/{total_steps}: UT Steps = {ut_steps}")
    print(f"{'=' * 70}\n")


def _load_model_for_ut_steps(
    experiment: OuroExperiment, ut_steps: int
) -> Tuple[Any, Any]:
    """
    Load model with specific UT steps configuration.
    
    Args:
        experiment: OuroExperiment instance
        ut_steps: Number of UT steps to configure
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        model, tokenizer, model_config, config_dict = (
            experiment.load_model_with_ut_steps(ut_steps)
        )
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model with UT steps={ut_steps}: {e}")


def _cleanup_gpu_memory(model, tokenizer):
    """Clean up GPU memory after model evaluation."""
    print(f"{'=' * 70}")
    print("Cleaning up GPU memory...")
    
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print("GPU memory freed successfully")
    print(f"{'=' * 70}\n")


def _handle_experiment_interruption(error: Exception, model, tokenizer):
    """Handle graceful shutdown of experiment."""
    print(f"\n{'=' * 70}")
    
    if isinstance(error, ExperimentFailureException):
        print(f"EXPERIMENT ABORTED GRACEFULLY: {error}")
    else:
        print("EXPERIMENT INTERRUPTED BY USER")
    
    print(f"{'=' * 70}")
    
    # Cleanup
    print("Finalizing W&B...")
    wandb.finish()
    print("W&B session closed")
    
    print(f"{'=' * 70}")
    print("Cleaning up GPU memory...")
    
    try:
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory freed")
    except:
        pass
    
    print(f"{'=' * 70}\n")


def _display_final_summary(results: ResultCollector):
    """Display final experiment summary with aggregated results."""
    print(f"\n{'=' * 70}")
    print("FINAL EXPERIMENT SUMMARY")
    print(f"{'=' * 70}\n")
    
    if results.simple_reasoning:
        _display_accuracy_summary(results.simple_reasoning)
    
    if results.perplexity:
        _display_perplexity_summary(results.perplexity)


def _display_accuracy_summary(simple_reasoning_results: List[Dict[str, Any]]):
    """Display accuracy summary for simple reasoning tasks."""
    df_all = pd.DataFrame(simple_reasoning_results)
    
    # Check for garbage outputs
    if "is_degenerate" in df_all.columns:
        num_garbage = df_all["is_degenerate"].sum()
        if num_garbage > 0:
            print(f"WARNING: {num_garbage} garbage/degenerate outputs detected\n")
    
    # Accuracy by task type
    print("Overall Accuracy by Task Type:")
    print(f"{'─' * 70}")
    accuracy_by_task = df_all.groupby("task_type")["is_correct"].agg(["mean", "count"])
    accuracy_by_task.columns = ["Accuracy", "N"]
    accuracy_by_task["Accuracy"] = (accuracy_by_task["Accuracy"] * 100).round(2)
    accuracy_by_task["Accuracy"] = accuracy_by_task["Accuracy"].apply(
        lambda x: f"{x:.2f}%"
    )
    print(accuracy_by_task)
    print()
    
    # Accuracy by UT steps
    print("Accuracy by UT Steps:")
    print(f"{'─' * 70}")
    accuracy_by_steps = df_all.groupby("ut_steps")["is_correct"].agg(["mean", "count"])
    accuracy_by_steps.columns = ["Accuracy", "N"]
    accuracy_by_steps["Accuracy"] = (accuracy_by_steps["Accuracy"] * 100).round(2)
    accuracy_by_steps["Accuracy"] = accuracy_by_steps["Accuracy"].apply(
        lambda x: f"{x:.2f}%"
    )
    print(accuracy_by_steps)
    print()
    
    # Accuracy pivot table
    print("Accuracy by Task Type and UT Steps:")
    print(f"{'─' * 70}")
    accuracy_pivot = (
        df_all.pivot_table(
            values="is_correct",
            index="task_type",
            columns="ut_steps",
            aggfunc="mean",
        ) * 100
    )
    print(accuracy_pivot.round(2))
    print()


def _display_perplexity_summary(perplexity_results: List[Dict[str, Any]]):
    """Display perplexity summary."""
    print("Perplexity by UT Steps:")
    print(f"{'─' * 70}")
    df_ppl = pd.DataFrame(perplexity_results)
    print(df_ppl.to_string(index=False))
    print()