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
            device_map="cuda",
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
        
        # APPLY SAFE OPTIMIZATIONS (especially important for UT > 1)
        if total_ut_steps > 1:
            print(f"\n{'─'*60}")
            print(f"🚀 APPLYING SAFE OPTIMIZATIONS (UT > 1)")
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
        Store raw prompt strings instead of tokenized tensors.
        """
        self.tokenizer = tokenizer

        task_configs = {
            "n_ary": {
                "system": (
                    "Add numbers step-by-step in this format:\n"
                    "Step <i>: <sum_so_far> + <next_num> = <new_sum>\n"
                    "Start with Step 1: 0 + <first> = <result>\n"
                    "End with [FINAL] <total>\n\n"
                    "STRICT RULES:\n"
                    "• Output ONLY the format above\n"
                    "• NO explanations or words\n"
                    "• NO repeating the input\n"
                    "• NO extra symbols\n"
                    "Begin Step 1 immediately."
                ),
                "force_start": "\nStep 1:",
            },
            "p_hop": {
                "system": (
                    "Trace hops in this format:\n"
                    "Hop <i>: At <current_token> → Next is <next_token>\n"
                    "End with [FINAL] <final_token>\n\n"
                    "STRICT RULES:\n"
                    "• Output ONLY the format above\n"
                    "• NO explanations or words\n"
                    "• NO repeating the sequence\n"
                    "• NO extra symbols\n"
                    "Begin Hop 1 immediately."
                ),
                "force_start": "\nHop 1:",
            },
            "igsm": {
                "system": (
                    "Evaluate (mod 7) in this format:\n"
                    "Step <i>: <var> = <expr> = <calc> (mod 7) = <result>\n"
                    "End with [FINAL] <answer>\n\n"
                    "STRICT RULES:\n"
                    "• Output ONLY the format above\n"
                    "• NO explanations or words\n"
                    "• NO repeating the problem\n"
                    "• NO extra parentheses\n"
                    "Begin Step 1 immediately."
                ),
                "force_start": "\nStep 1:",
            }
        }

        self.task_templates = {}
        for task_type, config in task_configs.items():
            static_messages = [{"role": "system", "content": config["system"]}]
            static_prompt_text = tokenizer.apply_chat_template(
                static_messages, tokenize=False, add_generation_prompt=True
            ).rstrip()
            force_text = config["force_start"].strip()
            self.task_templates[task_type] = {
                "static_prompt_text": static_prompt_text,
                "force_start_text": config.get("force_start", ""),
                "example_response": None
            }
        print("[+] Task templates (string only) pre-computed.")
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
        }

    @torch.inference_mode()
    def predict(
        self,
        user_inputs: Union[str, List[str]],
        task_type: str,
        model,
        tokenizer,
        ut_steps: int,
        generation_config: dict = None,
    ):
        # VERIFY model has the right config
        if not hasattr(model.config, 'total_ut_steps'):
            print("❌ ERROR: Model missing total_ut_steps config!")
            if isinstance(user_inputs, list):
                return [self._create_error_result(inp, ut_steps) for inp in user_inputs]
            else:
                return self._create_error_result(user_inputs, ut_steps)

        # Ensure task templates are built
        if not hasattr(self, "task_templates") or task_type not in self.task_templates:
            self._build_task_templates(tokenizer)

        template = self.task_templates[task_type]
        device = model.device

        # Handle single or batch input
        is_single = isinstance(user_inputs, str)
        if is_single:
            user_inputs = [user_inputs]

        # Concatenate prompt parts as strings for each input
        prompts = [
            f"{template['static_prompt_text']}{user_input}{template['force_start_text']}"
            for user_input in user_inputs
        ]

        # Tokenize all prompts at once (let tokenizer handle padding)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start_time = time.perf_counter()

        gen_config = generation_config or {
            "max_new_tokens": 512,
            "do_sample": False,
            "num_beams": 1,
            "min_length": 5,
            "repetition_penalty": 1
        }

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
            disable_compile=False,
            return_dict_in_generate=True,
            output_scores=False,
            **gen_config,
        )

        generation_time = time.perf_counter() - start_time

        results = []
        for i in range(len(user_inputs)):
            prompt_length = inputs["input_ids"][i].shape[0]
            # Find where the prompt ends in the generated sequence
            gen_seq = outputs.sequences[i]
            generated_ids = gen_seq[inputs["input_ids"].shape[1]:] if gen_seq.shape[0] > inputs["input_ids"].shape[1] else gen_seq[prompt_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            full_response = template["force_start_text"] + generated_text
            pred = self._extract_final_answer(full_response, task_type)
            results.append({
                "full_response": full_response,
                "prediction": pred,
                "generation_time": generation_time / len(user_inputs),
                "generated_tokens": generated_ids.shape[0],
                "input_tokens": prompt_length,
                "ut_steps": ut_steps,
            })

        return results[0] if is_single else results
            
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