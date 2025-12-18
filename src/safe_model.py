import torch
import time
import re
from typing import List, Optional, Dict, Any
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from tqdm.auto import tqdm
from .output_monitor import OutputQualityMonitor

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
        with torch.no_grad():
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
    """Core experiment class for Ouro model testing"""

    def __init__(
        self,
        model_path: str,
        dtype=torch.bfloat16,
        use_4bit_quant: bool = False,
        use_torch_compile: bool = False,
        k_repeat_abort: int = 5,
    ):
        torch.cuda.empty_cache()
        self.model_path = model_path
        self.dtype = dtype
        self.use_4bit_quant = use_4bit_quant
        self.use_torch_compile = use_torch_compile
        self.tokenizer = None
        self.task_templates = {}
        self.quality_monitor = None
        self.last_k_outputs = []
        self.k_repeat_abort = k_repeat_abort

    def check_repeated_outputs_and_abort(self, output: str):
        """Abort if the same output is generated for k consecutive inputs."""
        self.last_k_outputs.append(output)
        if len(self.last_k_outputs) > self.k_repeat_abort:
            self.last_k_outputs.pop(0)
        if (
            len(self.last_k_outputs) == self.k_repeat_abort
            and all(o == self.last_k_outputs[0] for o in self.last_k_outputs)
        ):
            print("\n" + "="*60)
            print(f"❌ EXPERIMENT TERMINATED: Model generated the same output {self.k_repeat_abort} times in a row.")
            print(f"Repeated output:\n{self.last_k_outputs[0]}")
            print("="*60 + "\n")
            raise SystemExit(f"Experiment failed: {self.k_repeat_abort} repeated outputs")
    
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
        auto_compile = (total_ut_steps == 1) and self.use_torch_compile
        
        print(f"\n{'='*60}")
        print(f"⚙️  LOADING MODEL CONFIGURATION")
        print(f"{'='*60}")
        print(f"Model Path: {self.model_path}")
        print(f"Requested UT Steps: {total_ut_steps}")
        print(f"Data Type: {self.dtype}")
        print(f"4-bit Quantization: {self.use_4bit_quant}")
        print(f"Torch Compile: {auto_compile} (auto={'ON' if total_ut_steps==1 else 'OFF'})")

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
            device_map="cuda",
            attn_implementation="sdpa_paged",
            torch_dtype=self.dtype if not self.use_4bit_quant else None,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        # Apply torch.compile only for UT=1
        if auto_compile:
            print("→ Applying torch.compile()")
            model = torch.compile(
                model, 
                mode="reduce-overhead",
                fullgraph=False
            )

        model.eval()
        
        # APPLY SAFE OPTIMIZATIONS (especially important for UT > 1)
        if total_ut_steps > 1:
            print(f"\n{'─'*60}")
            print(f"🚀 APPLYING SAFE OPTIMIZATIONS (UT > 1)")
            print(f"{'─'*60}")
            
            # try:
            #     SafeOptimizations.enable_static_cache(model)
            # except Exception as e:
            #     print(f"   ⚠️ Static cache failed: {e}")
            
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
        else:
            # Still apply basic optimizations for UT=1
            print(f"\n→ Applying basic optimizations (UT=1)...")
            try:
                model = SafeOptimizations.apply_inference_optimizations(model)
                SafeOptimizations.optimize_memory()
            except Exception as e:
                print(f"   ⚠️ Optimization failed: {e}")
        
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

    def _build_task_templates(self, tokenizer):
        """
        Pre-compute prompt templates with system prompts that match the data format
        generated by create_test_datasets in data_generator.py.
        """
        self.tokenizer = tokenizer

        task_configs = {
            "n_ary": {
                # Data format: "408 + 819 + 667 + 413 ="
                "system": (
                    "You are a calculator. Given an addition problem with several numbers (e.g., '{number_1} + {number_2} + {number_3} + ... ='), "
                    "show your work step by step. For each number, add it to the running total and show the calculation. "
                    "After all steps, output only the final sum on a new line as [FINAL] [sum].\n"
                    "Example:\n"
                    "Input: {number_1} + {number_2} + {number_3} + ... =\n"
                    "Step 1: \n"
                    "Step 2: {number_1} + {number_2} = {sum_2}\n"
                    "Step 3: {sum_2} + {number_3} = {sum_3}\n"
                    "[FINAL] {final_sum}"
                ),
                "force_start": "\nStep 1:",
            },
            "p_hop": {
                # Data format: "Sequence: A B C D A B. Start: A. Hop 1 times."
                "system": (
                    "You are a sequence tracer. Given a sequence of tokens, a start token, and a number of hops, "
                    "trace the sequence step by step. At each hop, move to the next occurrence in the sequence. "
                    "Show each hop as 'Hop {X}: At {token} → Next is {token}'. After all hops, output the result as [FINAL] {token}.\n"
                    "Example:\n"
                    "Input: Sequence: {token_1} {token_2} {token_3} .... Start: {token_1}. Hop 2 times.\n"
                    "Hop 1: At {token_1} → Next is {token_2}\n"
                    "Hop 2: At {token_2} → Next is {token_3}\n"
                    "[FINAL] {token_3}"
                ),
                "force_start": "\nHop 1:",
            },
            "igsm": {
                # Data format: "Question. E#I := 4. E#J := E#I. F#K := E#J. H#J := E#J + F#K. H#J?"
                "system": (
                    "You are a symbolic math solver working modulo 7. Given a list of assignments and a query, "
                    "evaluate each variable step by step. For each assignment, substitute known values and show the calculation. "
                    "For each step, show: '{var} = {expression} = {value} (mod 7)'. For the query, output the answer as [FINAL] {value}.\n"
                    "Example:\n"
                    "Input: Question. {token_1} := {value_1}. {token_2} := {token_1}. {token_3} := {token_2} + {token_1}. {token_3}?\n"
                    "Step 1: {token_1} = {value_1} (mod 7) = {value_after_mod}\n"
                    "Step 2: {token_2} = {token_1} = {value_1} (mod 7) = {value_after_mod}\n"
                    "Step 3: {token_3} = {token_2} + {token_1} = {value_2} + {value_1} = {value_3} (mod 7) = {value_after_mod}\n"
                    "[FINAL] {final_value}"
                ),
                "force_start": "\nStep 1:",
            }
        }

        self.task_templates = {}

        for task_type, config in task_configs.items():
            static_messages = [
                {"role": "system", "content": config["system"]}
            ]

            static_prompt_text = tokenizer.apply_chat_template(
                static_messages,
                tokenize=False,
                add_generation_prompt=True
            ).rstrip()
            static_inputs = tokenizer(static_prompt_text, return_tensors="pt")

            force_text = config["force_start"].strip()
            force_start_tokens = tokenizer(
                force_text,
                return_tensors="pt",
                add_special_tokens=False
            )

            self.task_templates[task_type] = {
                "static_input_ids": static_inputs.input_ids,
                "static_attention_mask": static_inputs.attention_mask,
                "force_start_ids": force_start_tokens.input_ids,
                "force_start_text": config.get("force_start", ""),
                "example_response": config.get("example_asst", None)
            }

        print("[+] Task templates pre-computed (this templates version runs properly on kaggle 10:35 18/12/2025 NLP thuyết trình).")
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
            "n_ary": 256,   # Need steps
            "p_hop": 256,   # Need steps
            "igsm": 512,    # Need reasoning steps
        }
        
        return {
            "max_new_tokens": task_token_limits.get(task_type, 256),
            "min_new_tokens": 10,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.0,
        }

    @torch.no_grad()
    def predict_with_metrics_optimized(
        self,
        user_input: str,
        task_type: str,
        model,
        tokenizer,
        ut_steps: int,
        generation_config: dict = None,
    ) -> Dict[str, Any]:
        """Optimized prediction with improved generation control"""
        
        if not hasattr(model.config, 'total_ut_steps'):
            print("❌ ERROR: Model missing total_ut_steps config!")
            return {
                "error": "Bad model config", 
                "prediction": "ERROR",
                "full_response": "",
                "generation_time": 0,
                "generated_tokens": 0,
                "input_tokens": 0,
                "ut_steps": ut_steps,
                "is_degenerate": False,
                "test_input": user_input,
            }

        if not hasattr(self, "task_templates") or task_type not in self.task_templates:
            self._build_task_templates(tokenizer)

        template = self.task_templates[task_type]
        device = model.device

        static_ids = template["static_input_ids"].to(device)
        
        user_tokens = tokenizer(
            user_input, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids.to(device)
        
        force_start_ids = template["force_start_ids"].to(device)

        input_ids = torch.cat([static_ids, user_tokens, force_start_ids], dim=1)
        attention_mask = torch.ones_like(input_ids, device=device)

        start_time = time.perf_counter()

        # Get optimal config for task type
        default_config = self._get_optimal_generation_config(task_type)
        
        if generation_config:
            default_config.update(generation_config)

        try:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
                **default_config,
            )
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return {
                "error": str(e),
                "prediction": "ERROR",
                "full_response": "",
                "generation_time": 0,
                "generated_tokens": 0,
                "input_tokens": input_ids.shape[1],
                "ut_steps": ut_steps,
                "is_degenerate": False,
                "test_input": user_input,
            }

        generation_time = time.perf_counter() - start_time

        prompt_length = input_ids.shape[1]
        generated_ids = outputs.sequences[0, prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        full_response = template["force_start_text"] + " " + generated_text
        self.check_repeated_outputs_and_abort(full_response)
        is_degenerate = self._detect_degenerate_output(full_response)
        
        if is_degenerate:
            print(f"⚠️ GARBAGE OUTPUT detected for {task_type}")
            print(f"   Response preview: {full_response[:200]}...")
            pred = "DEGENERATE"
        else:
            pred = self._extract_final_answer(full_response, task_type)

        return {
            "full_response": full_response,
            "prediction": pred,
            "generation_time": generation_time,
            "generated_tokens": generated_ids.shape[0],
            "input_tokens": input_ids.shape[1],
            "ut_steps": ut_steps,
            "is_degenerate": is_degenerate,
            "test_input": user_input,
        }

    @torch.no_grad()
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

            with torch.no_grad():
                outputs = model(
                    input_ids=input_slice,
                    attention_mask=attention_mask[:, i:end_loc],
                    labels=target_slice,
                )
                loss = outputs.loss
                num_tokens = input_slice.size(1) - 1  # exclude BOS
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
        If failure is detected, log details and raise SystemExit.
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
            raise SystemExit(f"Experiment failed: {failure.reason}")

class SafeOuroBatchExperiment(SafeOuroThinkingExperiment):
    """Extended experiment class with batch processing"""

    def __init__(
        self,
        model_path: str,
        dtype=torch.bfloat16,
        use_4bit_quant: bool = False,
        use_torch_compile: bool = False,
        max_batch_size: int = 4,
        max_new_tokens: int = 256,
    ):
        super().__init__(model_path, dtype, use_4bit_quant, use_torch_compile)
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens

    def prepare_batch_inputs(
        self, prompts: List[str], task_type: str
    ) -> List[List[int]]:
        """Prepare inputs for batch generation"""
        if task_type not in self.task_templates:
            raise ValueError("Templates not built. Call _build_task_templates first.")
        
        template = self.task_templates[task_type]
        
        user_encodings = self.tokenizer(
            prompts, 
            add_special_tokens=False,
            padding=False
        )
        
        static_ids = template["static_input_ids"].squeeze(0).tolist()
        force_ids = template["force_start_ids"].squeeze(0).tolist()
        
        input_id_lists = []
        for user_ids in user_encodings['input_ids']:
            full_seq = static_ids + user_ids + force_ids
            input_id_lists.append(full_seq)
        
        return input_id_lists

    @torch.no_grad()
    def batch_predict_with_metrics(
        self, 
        prompts: List[str], 
        task_type: str,
        model, 
        tokenizer, 
        ut_steps: int,
        generation_config: Optional[GenerationConfig] = None
    ):
        """Batch prediction with metrics"""
        if not prompts:
            return []
        
        if not hasattr(model, 'generate_batch'):
            print("⚠️ Model doesn't support generate_batch(). Using sequential.")
            return self._sequential_fallback(
                prompts, task_type, model, tokenizer, ut_steps, generation_config
            )
        
        simple_batch_inputs = self.prepare_batch_inputs(prompts, task_type)
        input_lengths = [len(ids) for ids in simple_batch_inputs]
        
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                use_cuda_graph=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                repetition_penalty=1.0,
                max_batch_tokens=self.max_batch_size * self.max_new_tokens,
            )
        
        start_time = time.perf_counter()
        
        try:
            batch_outputs = model.generate_batch(
                inputs=simple_batch_inputs,
                generation_config=generation_config,
            )
        except Exception as e:
            print(f"⚠️ generate_batch failed: {e}. Falling back to sequential.")
            return self._sequential_fallback(
                prompts, task_type, model, tokenizer, ut_steps, generation_config
            )
        
        batch_time = time.perf_counter() - start_time
        
        template = self.task_templates[task_type]
        results = [None] * len(prompts)
        request_ids = list(batch_outputs.keys())
        
        if all(isinstance(rid, str) for rid in request_ids):
            input_to_index = {
                " ".join(map(str, inp)): idx 
                for idx, inp in enumerate(simple_batch_inputs)
            }
            
            for request_id in request_ids:
                output = batch_outputs[request_id]
                
                if hasattr(output, 'prompt_ids'):
                    input_key = " ".join(map(str, output.prompt_ids))
                    if input_key in input_to_index:
                        idx = input_to_index[input_key]
                        results[idx] = self._process_single_output(
                            output, idx, len(output.prompt_ids),
                            template, tokenizer, task_type, ut_steps,
                            batch_time / len(prompts)
                        )
        
        for i in range(len(prompts)):
            if results[i] is None:
                results[i] = {
                    'full_response': 'ERROR: No output',
                    'prediction': 'ERROR',
                    'generation_time': batch_time / len(prompts),
                    'generated_tokens': 0,
                    'input_tokens': input_lengths[i],
                    'ut_steps': ut_steps,
                    'is_degenerate': False,
                }
        
        return results

    def _process_single_output(
        self,
        output,
        prompt_idx: int,
        input_length: int,
        template: dict,
        tokenizer,
        task_type: str,
        ut_steps: int,
        sample_time: float,
    ):
        """Process single batch output with garbage detection"""
        if hasattr(output, "generated_tokens"):
            generated_ids = output.generated_tokens
        elif hasattr(output, "sequences") and len(output.sequences) > 0:
            generated_ids = output.sequences[0]
        else:
            generated_ids = []

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_response = template["force_start_text"] + " " + generated_text
        self.check_repeated_outputs_and_abort(full_response)
        is_degenerate = self._detect_degenerate_output(full_response)
        
        if is_degenerate:
            print(f"⚠️ GARBAGE OUTPUT detected in batch for {task_type} (idx={prompt_idx})")
            pred = "DEGENERATE"
        else:
            pred = self._extract_final_answer(full_response, task_type)

        return {
            "full_response": full_response,
            "prediction": pred,
            "generation_time": sample_time,
            "generated_tokens": len(generated_ids),
            "input_tokens": input_length,
            "ut_steps": ut_steps,
            "prompt_idx": prompt_idx,
            "is_degenerate": is_degenerate,
        }

    def _sequential_fallback(
        self, 
        prompts, 
        task_type, 
        model, 
        tokenizer, 
        ut_steps, 
        generation_config
    ):
        """Fallback to sequential processing"""
        results = []
        for prompt in tqdm(prompts, desc=f"Sequential ({task_type})"):
            result = self.predict_with_metrics_optimized(
                user_input=prompt,
                task_type=task_type,
                model=model,
                tokenizer=tokenizer,
                ut_steps=ut_steps,
                generation_config=generation_config,
            )
            results.append(result)
        return results