import torch
import time
import re
from typing import List, Optional, Dict, Any, Union
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from tqdm.auto import tqdm
from .output_monitor import OutputQualityMonitor, ExperimentFailureException


class SafeOptimizations:
    """Safe optimization methods that don't contaminate model state"""
    
    @staticmethod
    def enable_static_cache(model, max_seq_length: int = 2048):
        """Pre-allocate static KV cache"""
        if hasattr(model, 'generation_config'):
            model.generation_config.cache_implementation = "static"
            model.generation_config.max_cache_length = max_seq_length
            print("   ✓ Static KV cache enabled")
    
    @staticmethod
    def optimize_attention_backend(model):
        """Enable Flash Attention / Memory-Efficient SDPA"""
        if torch.cuda.is_available() and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("   ✓ Flash Attention / SDPA enabled")
        return model
    
    @staticmethod
    def apply_inference_optimizations(model):
        """Apply safe inference-only optimizations"""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        if hasattr(model, 'generation_config'):
            model.generation_config.use_cache = True
        
        if torch.cuda.is_available():
            # TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("   ✓ TF32 enabled for matmul")
            
            # cuDNN auto-tuning
            torch.backends.cudnn.benchmark = True
            print("   ✓ cuDNN auto-tuning enabled")
        
        return model
    
    @staticmethod
    def optimize_memory():
        """Optimize CUDA memory allocation"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   ✓ Memory pool optimized")
    
    @staticmethod
    def warmup_model(model, tokenizer, num_passes: int = 3):
        """Warmup CUDA kernels"""
        device = model.device
        dummy_input = tokenizer("warmup test", return_tensors="pt")
        input_ids = dummy_input.input_ids.to(device)
        
        print(f"   → Running {num_passes} warmup passes...")
        with torch.inference_mode():
            for i in range(num_passes):
                _ = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=32,
                    use_cache=True,
                    do_sample=False,
                )
        
        torch.cuda.empty_cache()
        print("   ✓ Warmup complete")


class SafeOuroThinkingExperiment:
    """Core experiment class for Ouro model testing with unified prediction"""

    def __init__(
        self,
        model_path: str,
        dtype=torch.bfloat16,
        use_4bit_quant: bool = False,
        use_torch_compile: bool = False,
        k_repeat_abort: int = 5,
        max_batch_size: int = 4,
        max_new_tokens: int = 256,
    ):
        torch.cuda.empty_cache()
        self.model_path = model_path
        self.dtype = dtype
        self.use_4bit_quant = use_4bit_quant
        self.use_torch_compile = use_torch_compile
        self.tokenizer = None
        self.task_templates = {}
        self.last_k_outputs = []
        self.k_repeat_abort = k_repeat_abort
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens

        # Initialize quality monitor to prevent kaggle GPU loss meaninglessly
        self.initialize_quality_monitor()

    def check_repeated_outputs_and_abort(self, output: str):
        if not hasattr(self, "last_k_outputs"):
            self.last_k_outputs = []
        if not hasattr(self, "k_repeat_abort"):
            self.k_repeat_abort = 5  # or set via config

        self.last_k_outputs.append(output)
        if len(self.last_k_outputs) > self.k_repeat_abort:
            self.last_k_outputs.pop(0)
        if (
            len(self.last_k_outputs) == self.k_repeat_abort
            and all(o == self.last_k_outputs[0] for o in self.last_k_outputs)
        ):
            print(f"❌ Aborting due to repeated outputs...")
            raise ExperimentFailureException(f"Experiment failed: {self.k_repeat_abort} repeated outputs")
    
    def initialize_quality_monitor(
        self,
        garbage_threshold: float = 0.3,
        example_similarity_threshold: float = 0.85,
        min_samples: int = 10,
        window_size: int = 20
    ):
        """Initialize output quality monitoring"""
        self.quality_monitor = OutputQualityMonitor(
            garbage_threshold=garbage_threshold,
            example_similarity_threshold=example_similarity_threshold,
            min_samples_before_check=min_samples,
            window_size=window_size
        )
        print(f"[+] Quality monitor initialized:")
        print(f"    → Garbage threshold: {garbage_threshold*100:.0f}%")
        print(f"    → Example similarity threshold: {example_similarity_threshold*100:.0f}%")
        print(f"    → Min samples before check: {min_samples}")

    def load_model_with_ut_steps(self, total_ut_steps: int):
        """Load model with specific UT steps configuration and apply safe optimizations"""
        quantization_config = None
        if self.use_4bit_quant:
            print("→ Applying 4-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        # Auto-enable torch.compile only for ut_steps=1
        auto_compile = self.use_torch_compile
        
        print(f"\n{'='*60}")
        print(f"⚙️  LOADING MODEL CONFIGURATION")
        print(f"{'='*60}")
        print(f"Model Path: {self.model_path}")
        print(f"Requested UT Steps: {total_ut_steps}")
        print(f"Data Type: {self.dtype}")
        print(f"4-bit Quantization: {self.use_4bit_quant}")
        print(f"Torch Compile: {auto_compile}")

        # Load base config
        base_config = AutoConfig.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        print(f"\n→ Base config loaded")
        print(f"   Original UT steps: {getattr(base_config, 'total_ut_steps', 'N/A')}")
        print(f"   Original early exit: {getattr(base_config, 'early_exit_threshold', 'N/A')}")
        
        # Apply UT step configuration
        base_config.total_ut_steps = total_ut_steps
        print(f"\n→ Modified config:")
        print(f"   New UT steps: {base_config.total_ut_steps}")
        print(f"   Early exit threshold: {base_config.early_exit_threshold} (from default)")
                
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"\n→ Tokenizer loaded")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        print(f"   PAD token: {tokenizer.pad_token}")
        print(f"   EOS token: {tokenizer.eos_token}")

        # Load model
        print(f"\n→ Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=base_config,
            device_map="auto",
            attn_implementation="sdpa_paged",
            torch_dtype=self.dtype if not self.use_4bit_quant else None,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        # Apply torch.compile only for UT=1
        if auto_compile:
            print("→ Applying torch.compile()")
            model = torch.compile(model)

        model.eval()
        
        print(f"\n{'─'*60}")
        print(f"🚀 APPLYING SAFE OPTIMIZATIONS")
        print(f"{'─'*60}")
        
        try:
            model = SafeOptimizations.optimize_attention_backend(model)
        except Exception as e:
            print(f"   ⚠️ Attention optimization failed: {e}")
        
        try:
            model = SafeOptimizations.apply_inference_optimizations(model)
        except Exception as e:
            print(f"   ⚠️ Inference optimization failed: {e}")
        
        try:
            SafeOptimizations.optimize_memory()
        except Exception as e:
            print(f"   ⚠️ Memory optimization failed: {e}")
        
        try:
            SafeOptimizations.warmup_model(model, tokenizer, num_passes=3)
        except Exception as e:
            print(f"   ⚠️ Warmup failed: {e}")
        
        print(f"{'─'*60}")
        
        # Final verification
        print(f"\n{'='*60}")
        print(f"✅ MODEL LOADED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Device: {model.device}")
        print(f"Model dtype: {model.dtype}")
        print(f"VERIFIED UT steps: {model.config.total_ut_steps}")
        print(f"VERIFIED early exit: {model.config.early_exit_threshold}")
        
        if model.config.total_ut_steps != total_ut_steps:
            print(f"\n⚠️  WARNING: UT STEPS MISMATCH!")
            print(f"   Requested: {total_ut_steps}")
            print(f"   Actual: {model.config.total_ut_steps}")
        
        print(f"{'='*60}\n")
        
        return model, tokenizer, base_config, {
            "total_ut_steps": total_ut_steps,
            "early_exit_threshold": base_config.early_exit_threshold,
        }
    
    def check_chat_format(self, response: str) -> bool:
        """
        Check if the response string contains the correct chat format:
        <|im_start|>system ... <|im_end|>
        <|im_start|>user ... <|im_end|>
        <|im_start|>assistant
        Returns True if all are present in order, False otherwise.
        """
        system_idx = response.find("<|im_start|>system")
        user_idx = response.find("<|im_start|>user")
        assistant_idx = response.find("<|im_start|>assistant")

        # All must be present and in correct order
        if system_idx == -1 or user_idx == -1 or assistant_idx == -1:
            return False
        return system_idx < user_idx < assistant_idx
        

    def _build_task_templates(self, tokenizer):
        """
        Pre-compute prompt templates using strict format and anti-pattern warnings.
        Stores only the system prompt and force_start text for each task.
        Adds explicit instruction to stop calculation at the correct symbol for each task.
        """
        self.tokenizer = tokenizer

        task_configs = {
        "n_ary": {
            "system": (
                "You are a calculator performing sequential addition.\n\n"
                "INSTRUCTIONS:\n"
                "1. Count the numbers in the input (call this N)\n"
                "2. Perform exactly N addition steps\n"
                "3. Stop after N steps and output [FINAL]\n\n"
                "FORMAT:\n"
                "Step {1}: 0 + {first_number} = {sum_1}\n"
                "Step {2}: {sum_1} + {second_number} = {sum_2}\n"
                "Step {3}: {sum_2} + {third_number} = {sum_3}\n"
                "...\n"
                "Step {N-1}: {sum_N-2} + {penultimate_number} = {sum_N-1}\n"
                "Step {N}: {sum_N-1} + {last_number} = {final_sum}\n"
                "[FINAL] {final_sum} [END]\n\n"
                "CRITICAL RULES:\n"
                "• Each input number appears in EXACTLY ONE step\n"
                "• Number of steps = number of input numbers\n"
                "• After step N, immediately output [FINAL] {final_sum} [END]\n"
                "• NO additional steps after the line contains [END]\n\n"
                "PATTERN EXPLANATION:\n"
                "Input '{A} + {B} =' has 2 numbers → Output 2 steps + [FINAL]\n"
                "Input '{A} + {B} + {C} =' has 3 numbers → Output 3 steps + [FINAL]\n"
                "Input '{A} + {B} + {C} + {D} =' has 4 numbers → Output 4 steps + [FINAL]\n\n"
                "FORBIDDEN:\n"
                "❌ NO repeating the same number in multiple steps\n"
                "❌ NO continuing after all input numbers are used\n"
                "❌ NO generating steps beyond the input count\n"
                "❌ NO explanations or commentary"
            ),
            "force_start": "",
        },
        "p_hop": {
            "system": (
                "You are a sequence position tracker.\n\n"
                "INSTRUCTIONS:\n"
                "1. Count the number of hops requested (call this N)\n"
                "2. Perform exactly N hop steps\n"
                "3. Stop after N hops and output [FINAL]\n\n"
                "FORMAT:\n"
                "Hop {1}: At {token_1} → Next is {token_2}\n"
                "Hop {2}: At {token_2} → Next is {token_3}\n"
                "Hop {3}: At {token_3} → Next is {token_4}\n"
                "...\n"
                "Hop {N}: At {token_N} → Next is {final_token}\n"
                "[FINAL] {final_token}\n\n"
                "CRITICAL RULES:\n"
                "• Perform exactly the requested number of hops\n"
                "• Follow the sequence order (if sequence repeats, wrap around)\n"
                "• After hop N, immediately output [FINAL]\n"
                "• Use only tokens from the input sequence\n\n"
                "PATTERN EXPLANATION:\n"
                "If input says 'Hop 3 times' → Output 3 hop lines + [FINAL]\n"
                "If input says 'Hop 5 times' → Output 5 hop lines + [FINAL]\n\n"
                "FORBIDDEN:\n"
                "❌ NO extra hops beyond the requested count\n"
                "❌ NO inventing tokens not in the sequence\n"
                "❌ NO explanations or commentary"
            ),
            "force_start": "Hop 1:",
        },
        "igsm": {
            "system": (
                "You are a symbolic expression evaluator (modulo 7).\n\n"
                "INSTRUCTIONS:\n"
                "1. Count the number of assignments (call this N)\n"
                "2. Evaluate exactly N assignments\n"
                "3. Stop after N steps and output [FINAL]\n\n"
                "FORMAT:\n"
                "Step {1}: {var_1} = {value_1} (mod 7) = {result_1}\n"
                "Step {2}: {var_2} = {substituted_expr} = {computed} (mod 7) = {result_2}\n"
                "Step {3}: {var_3} = {substituted_expr} = {computed} (mod 7) = {result_3}\n"
                "...\n"
                "Step {N}: {query_var} = {value} (mod 7) = {answer}\n"
                "[FINAL] {answer}\n\n"
                "CRITICAL RULES:\n"
                "• Process each assignment exactly once\n"
                "• Substitute variable values immediately\n"
                "• Show computation before applying mod 7\n"
                "• Final result must be in range [0, 6]\n"
                "• After evaluating the query, immediately output [FINAL]\n\n"
                "OPERATION PATTERNS:\n"
                "Assignment: {A} := {5} means {A} = 5 (mod 7) = 5\n"
                "Copy: {B} := {A} where A=5 means {B} = 5 (mod 7) = 5\n"
                "Addition: {C} := {A} + {B} where A=5, B=4 means {C} = 5 + 4 = 9 (mod 7) = 2\n\n"
                "FORBIDDEN:\n"
                "❌ NO skipping assignments\n"
                "❌ NO continuing after the query is answered\n"
                "❌ NO results outside [0, 6]\n"
                "❌ NO explanations or commentary"
            ),
            "force_start": "Step 1:",
        }
    }

        self.task_templates = {}
        for task_type, config in task_configs.items():
            self.task_templates[task_type] = {
                "system": config["system"],
                "force_start_text": config["force_start"],
            }
        print("[+] Task templates (strict hybrid) pre-computed.")

    @torch.inference_mode()
    def predict(
        self,
        user_inputs: Union[str, List[str]],
        task_type: str,
        model,
        tokenizer,
        ut_steps: int,
        generation_config: dict = None,
        enable_batch: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Unified prediction function that handles both single and batch inputs.
        Uses chat template correctly to avoid misplaced assistant tokens.
        """
        is_single = isinstance(user_inputs, str)
        if is_single:
            user_inputs = [user_inputs]

        if not hasattr(model.config, 'total_ut_steps'):
            print("❌ ERROR: Model missing total_ut_steps config!")
            error_results = [self._create_error_result(inp, ut_steps) for inp in user_inputs]
            return error_results[0] if is_single else error_results

        if not hasattr(self, "task_templates") or task_type not in self.task_templates:
            self._build_task_templates(tokenizer)

        template = self.task_templates[task_type]
        device = model.device

        # Build chat messages for each input
        prompts = []
        for user_input in user_inputs:
            # construct one message
            messages = [
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": user_input},
            ]

            # Apply chat template (adding <|im_start|>assistant)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Add_generate_prompt is true so it will append <|im_start|>assistant
            # Now add the force_start_text to guide generation
            # The artiface <|im_start|>assistant is added with \n at the end so we dont need \n at the beginning of force start ahihi
            prompt += template["force_start_text"]

            print(f"DEBUG: Full prompt for '{task_type}':\n{prompt}\n")
            if self.check_chat_format(prompt):
                print(f"   ✓ Chat format verified for input.")
            else:
                print(f"   ⚠️ Chat format NOT verified for input.")

            # add single prompt to batch for batch inference
            prompts.append(prompt)

        # Tokenize all prompts at once (let tokenizer handle padding)
        # adlready set padding left on tokenizer init
        encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        default_config = self._get_optimal_generation_config(task_type)
        if generation_config:
            default_config.update(generation_config)


        start_time = time.perf_counter()
        try:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
                generation_config=GenerationConfig(**default_config),
            )
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            error_results = [self._create_error_result(inp, ut_steps, str(e)) for inp in user_inputs]
            return error_results[0] if is_single else error_results

        generation_time = time.perf_counter() - start_time

        results = []
        for i in range(len(user_inputs)):
            prompt_length = attention_mask[i].sum().item()

            generated_ids = outputs.sequences[i, input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            full_response = template["force_start_text"] + " " + generated_text

            
            self.check_repeated_outputs_and_abort(full_response)
            is_degenerate = self._detect_degenerate_output(full_response)

            if is_degenerate:
                print(f"⚠️ GARBAGE OUTPUT detected for {task_type} (batch item {i})")
                print(f"   Response preview: {full_response[:200]}...")
                pred = "DEGENERATE"
            else:
                pred = self._extract_final_answer(full_response, task_type)

            result = {
                "full_response": full_response,
                "prediction": pred,
                "generation_time": generation_time / len(user_inputs),
                "generated_tokens": generated_ids.shape[0],
                "input_tokens": prompt_length,
                "ut_steps": ut_steps,
                "is_degenerate": is_degenerate,
                "test_input": user_inputs[i],
            }
            results.append(result)

        return results[0] if is_single else results
        
    def _extract_final_answer(self, full_response: str, task_type: str) -> str:
        """Extract final answer with improved parsing"""
        pred = "0"
        
        try:
            full_response = full_response.strip()
            
            if task_type == "p_hop":
                patterns = [
                    r'\[FINAL\]\s*([A-D])\b',
                    r'Final:\s*([A-D])\b',
                    r'\b([A-D])\s*$',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, full_response, re.IGNORECASE)
                    if matches:
                        pred = matches[-1].upper()
                        break
                else:
                    pred = "ERROR"
            
            else:
                patterns = [
                    r'\[FINAL\]\s*(-?\d+)',
                    r'Final:\s*(-?\d+)',
                    r'=\s*(-?\d+)\s*$',
                    r'\b(-?\d+)\s*$',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, full_response)
                    if matches:
                        pred = matches[-1]
                        break
                else:
                    lines = [l.strip() for l in full_response.split('\n') if l.strip()]
                    if lines:
                        last_line = lines[-1]
                        numbers = re.findall(r'-?\d+', last_line)
                        if numbers:
                            pred = numbers[-1]
                        else:
                            pred = "ERROR"
                    else:
                        pred = "ERROR"
        
        except Exception as e:
            print(f"[!] Parsing error: {e}")
            pred = "ParseError"
        
        return pred

    def _detect_degenerate_output(self, text: str) -> bool:
        """Detect if output is degenerate/garbage"""
        if not text or len(text.strip()) < 5:
            return True
        
        if text.count('\n\n\n') > 3:
            return True
        
        bracket_ratio = (text.count('[') + text.count(']')) / max(len(text), 1)
        if bracket_ratio > 0.3:
            return True
        
        if len(text) > 100:
            unique_chars = len(set(text))
            if unique_chars < 10:
                return True
        
        whitespace_ratio = (text.count(' ') + text.count('\n')) / max(len(text), 1)
        if whitespace_ratio > 0.7:
            return True
        
        if len(text) > 50:
            for char in ['[', ']', '\n', ' ', '.']:
                if text.count(char) > len(text) * 0.4:
                    return True
        
        return False

    def _get_optimal_generation_config(self, task_type: str) -> Dict:
        """Get optimized generation parameters for task type"""
        task_token_limits = {
            "n_ary": 256,
            "p_hop": 256,
            "igsm": 512,
        }
        
        return {
            "max_new_tokens": task_token_limits.get(task_type, 256),
            "min_new_tokens": 10,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "temperature": 0.0,
        }
    
    def _create_error_result(self, user_input: str, ut_steps: int, error_msg: str = "Model config error") -> Dict[str, Any]:
        """Create an error result dictionary"""
        return {
            "error": error_msg,
            "prediction": "ERROR",
            "full_response": "",
            "generation_time": 0,
            "generated_tokens": 0,
            "input_tokens": 0,
            "ut_steps": ut_steps,
            "is_degenerate": False,
            "test_input": user_input,
        }

    @torch.inference_mode()
    def calculate_perplexity(
        self,
        model,
        tokenizer,
        text_data: List[str],
        ut_steps: int,
        max_length: int = 2048,
        stride: int = 512,
    ):
        """Calculate perplexity using sliding window"""
        device = model.device
        model.eval()

        if not text_data or not text_data[0]:
            return float("nan"), float("nan")

        text_concat = text_data[0]
        encodings = tokenizer(
            text_concat, 
            return_tensors="pt", 
            max_length=max_length * 2, 
            truncation=True
        )
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        if input_ids.size(1) < 2:
            return float("nan"), float("nan")

        total_loss = 0.0
        total_tokens = 0

        for i in tqdm(
            range(0, input_ids.size(1), stride), 
            desc=f"Calculating PPL (UT={ut_steps})"
        ):
            end_loc = min(i + max_length, input_ids.size(1))
            input_slice = input_ids[:, i:end_loc]
            target_slice = input_slice.clone()

            if input_slice.size(1) < 2:
                continue

            with torch.inference_mode():
                outputs = model(
                    input_ids=input_slice,
                    attention_mask=attention_mask[:, i:end_loc],
                    labels=target_slice,
                )
                loss = outputs.loss
                num_tokens = input_slice.size(1) - 1
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        if total_tokens == 0:
            return float("nan"), float("nan")

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return avg_loss, perplexity

    def monitor_and_maybe_abort(self, result: Dict[str, Any], task_type: str):
        """
        Add result to quality monitor and check for experiment failure.
        If failure is detected, log details and raise ExperimentFailureException.
        """
        if self.quality_monitor is None:
            return

        self.quality_monitor.add_result(result, task_type)
        failure = self.quality_monitor.check_failure_conditions()
        if failure:
            print("\n" + "="*60)
            print("❌ EXPERIMENT TERMINATED DUE TO OUTPUT QUALITY FAILURE")
            print(f"Reason: {failure.reason}")
            print("Details:")
            for k, v in failure.failure_stats.items():
                if isinstance(v, list):
                    print(f"  {k}:")
                    for idx, item in enumerate(v, 1):
                        print(f"    [{idx}] {item}")
                else:
                    print(f"  {k}: {v}")
            print("="*60 + "\n")
            raise ExperimentFailureException(f"Experiment failed: {failure.reason}")