import random
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
from .data_generator import create_test_datasets

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
                    tokenizer=tokenizer,
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
        # elif self.use_8bit_quant:
        #     print("→ Applying 8-bit quantization")
        #     quantization_config = BitsAndBytesConfig(load_in_8bit=True)

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

        # try:
        #     from optimum.bettertransformer import BetterTransformer
        #     model = BetterTransformer.transform(model, keep_original_model=True) 
        #     print("✓ BetterTransformer enabled")
        # except Exception as e:
        #     print(f"✗ BetterTransformer not available: {e}")

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
        

    def _build_task_templates(self, tokenizer, num_shots: int = 4):
        """
        Build task templates using REAL few-shot examples from data_generator.py.
        This ensures the system prompt matches actual test distribution.
        
        Args:
            tokenizer: HuggingFace tokenizer
            num_shots: Number of few-shot examples per task (default: 4)
        """
        self.tokenizer = tokenizer

        # --- STEP 1: Generate real samples from data_generator ---
        print(f"[+] Generating {num_shots} real few-shot examples for each task...")
        config = {
            "n_ary": {"ops_levels": [2, 4], "num_samples_per_level": max(10, num_shots + 2)},
            "p_hop": {"hop_levels": [4, 8], "num_samples_per_level": max(10, num_shots + 2)},
            "igsm": {"num_samples_total": max(20, num_shots + 5)},
        }
        task_data = create_test_datasets(config)

        # --- STEP 2: Sample few-shot examples ---
        def sample_shots(samples, n=num_shots):
            """Sample n diverse examples"""
            if len(samples) < n:
                return samples
            return random.sample(samples, n)

        # --- STEP 3: Build system prompts with real examples ---
        task_configs = {}

        # N-ARY ADDITION
        if "n_ary" in task_data and task_data["n_ary"]:
            shots = ""
            shots = sample_shots(task_data["n_ary"])
            examples_text = ""
            for ex in shots:
                # Format: "Input: 234 + 567 = \n[FINAL] 801 [END]\n\n"
                examples_text += f"Input: {ex['prompt']}\n[FINAL] {ex['expected_answer']} [END]\n\n"
            
            task_configs["n_ary"] = {
                "system": (
                    "You solve ADDITION by adding numbers one at a time.\n\n"
                    f"EXAMPLES:\n{examples_text}"
                    "RULES:\n"
                    "✓ Show ONLY the final answer, no intermediate steps.\n"
                    "✓ Start with 0, add one number per step (but steps are hidden).\n"
                    "✓ Use ONLY numbers from input (no inventing).\n"
                    "✓ After final step: [FINAL] {answer} [END]\n"
                    "✓ STOP at [END]\n\n"
                    "✗ NO extra steps after all input numbers used.\n"
                    "✗ NO inventing numbers or patterns.\n"
                    "✗ NO code/explanations after [END].\n"
                ),
                "force_start": "[FINAL]",
                "few_shot_examples": shots,  # Store for debugging
            }

        # P-HOP INDUCTION
        if "p_hop" in task_data and task_data["p_hop"]:
            shots = ""
            shots = sample_shots(task_data["p_hop"])
            examples_text = ""
            for ex in shots:
                # Parse prompt to extract components
                prompt = ex['prompt']
                # "Sequence: A B C D E. Start: A. Hop 3 times."
                if "Sequence:" in prompt and "Start:" in prompt and "Hop" in prompt:
                    seq_part = prompt.split("Sequence: ")[1].split(". Start:")[0]
                    start_part = prompt.split("Start: ")[1].split(". Hop")[0]
                    hops_part = prompt.split("Hop ")[1].split(" times")[0]
                    examples_text += (
                        f"Input: Sequence: {seq_part} | Start: {start_part} | Hops: {hops_part}\n"
                        f"[FINAL] {ex['expected_answer']} [END]\n\n"
                    )
                else:
                    # Fallback: use raw prompt
                    examples_text += f"Input: {ex['prompt']}\n[FINAL] {ex['expected_answer']} [END]\n\n"
            
            task_configs["p_hop"] = {
                "system": (
                    "You solve SEQUENCE HOPPING: follow a sequence forward N steps.\n\n"
                    f"EXAMPLES:\n{examples_text}"
                    "RULES:\n"
                    "✓ Show ONLY the final token, no hop details.\n"
                    "✓ Each hop moves one position forward (but hops are hidden).\n"
                    "✓ Use ONLY tokens from input sequence.\n"
                    "✓ After final hop: [FINAL] {token} [END]\n"
                    "✓ STOP at [END]\n\n"
                    "✗ NO skipping hops.\n"
                    "✗ NO inventing tokens.\n"
                    "✗ NO jumping to [FINAL] without showing hops.\n"
                    "✗ NO code/explanations after [END].\n"
                ),
                "force_start": "[FINAL]",
                "few_shot_examples": shots,
            }

        # IGSM (Symbolic Modular Arithmetic)
        if "igsm" in task_data and task_data["igsm"]:
            shots = ""
            shots = sample_shots(task_data["igsm"])
            examples_text = ""
            for ex in shots:
                # Parse prompt: "Question. A := 5. B := A + 3. B?"
                prompt = ex['prompt']
                if "Question." in prompt:
                    eqs = prompt.replace("Question. ", "")
                    # Split on last period to separate query
                    if "?" in eqs:
                        parts = eqs.rsplit(".", 1)
                        if len(parts) == 2:
                            assignments, query = parts
                            examples_text += (
                                f"Input: {assignments.strip()} | Query: {query.strip()}\n"
                                f"[FINAL] {ex['expected_answer']} [END]\n\n"
                            )
                            continue
                # Fallback
                examples_text += f"Input: {ex['prompt']}\n[FINAL] {ex['expected_answer']} [END]\n\n"
            
            task_configs["igsm"] = {
                "system": (
                    "You solve MODULAR ARITHMETIC (mod 7): evaluate assignments, apply mod 7.\n"
                    "Valid results: 0‑6 only.\n\n"
                    "MOD 7 QUICK REFERENCE:\n"
                    "5 mod 7 = 5 | 8 mod 7 = 1 | 14 mod 7 = 0 | 20 mod 7 = 6\n\n"
                    f"EXAMPLES:\n{examples_text}"
                    "RULES:\n"
                    "✓ Process every assignment in order (but calculations are hidden).\n"
                    "✓ Show ONLY the final answer, no intermediate calculations.\n"
                    "✓ Final answer must be 0‑6.\n"
                    "✓ After query variable: [FINAL] {answer} [END]\n"
                    "✓ STOP at [END]\n\n"
                    "✗ NO skipping assignments.\n"
                    "✗ NO results outside 0‑6.\n"
                    "✗ NO continuing after query found.\n"
                    "✗ NO code/explanations after [END].\n"
                ),
                "force_start": "[FINAL]",
                "few_shot_examples": shots,
            }

        # --- STEP 4: Pre-compute generation configs (unchanged) ---
        task_generation_configs = {
            "n_ary": GenerationConfig(
                max_new_tokens=16,
                min_new_tokens=2,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                temperature=0.7,
            ),
            "p_hop": GenerationConfig(
                max_new_tokens=16,
                min_new_tokens=2,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                temperature=0.7,
            ),
            "igsm": GenerationConfig(
                max_new_tokens=16,
                min_new_tokens=2,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                temperature=0.7,
            ),
        }

        # --- STEP 5: Pre-tokenize everything (unchanged logic) ---
        self.task_templates = {}
        
        for task_type, config in task_configs.items():
            # Pre-tokenize system prompt
            system_messages = [{"role": "system", "content": config["system"]}]
            system_text = tokenizer.apply_chat_template(
                system_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            system_tokens = tokenizer.encode(system_text, add_special_tokens=False)
            
            # Pre-tokenize force_start
            force_start_tokens = tokenizer.encode(
                config["force_start"],
                add_special_tokens=False,
            )
            
            # Pre-compute user message wrapper tokens
            dummy_user_msg = [{"role": "user", "content": "PLACEHOLDER"}]
            user_template_text = tokenizer.apply_chat_template(
                dummy_user_msg,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            if "PLACEHOLDER" in user_template_text:
                prefix, suffix = user_template_text.split("PLACEHOLDER", 1)
                user_prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                user_suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
            else:
                user_prefix_tokens = []
                user_suffix_tokens = tokenizer.encode(user_template_text, add_special_tokens=False)
            
            self.task_templates[task_type] = {
                # Original strings (for debugging)
                "system": config["system"],
                "force_start_text": config["force_start"],
                
                # Pre-tokenized components
                "system_tokens": system_tokens,
                "user_prefix_tokens": user_prefix_tokens,
                "user_suffix_tokens": user_suffix_tokens,
                "force_start_tokens": force_start_tokens,
                
                # Pre-computed generation config
                "generation_config": task_generation_configs[task_type],
                
                # Stop sequences
                "stop_sequences": [
                    "[END]", "[FINAL]", "SAMPLE", "\n```", "\n\n",
                    "```python", "def ", "#", "**Final", "Example usage",
                ],
                
                # Store few-shot examples for reference
                "few_shot_examples": config.get("few_shot_examples", []),
            }
        
        print("[+] Task templates built with REAL few-shot examples from data generator.")
        print(f"    Few-shot examples per task: {num_shots}")
        for task_type in self.task_templates:
            n_examples = len(self.task_templates[task_type]["few_shot_examples"])
            n_tokens = len(self.task_templates[task_type]["system_tokens"])
            print(f"    • {task_type}: {n_examples} examples, {n_tokens} system tokens")
        print(f"    User prefix tokens: {len(self.task_templates['n_ary']['user_prefix_tokens'])} tokens")
        print(f"    User suffix tokens: {len(self.task_templates['n_ary']['user_suffix_tokens'])} tokens")
        print(f"    Force start tokens: {len(self.task_templates['n_ary']['force_start_tokens'])} tokens")


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
        Ultra-optimized prediction with minimal per-sample overhead.
        All templates and configs are pre-computed.
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

        # Pre-computed constant components (no computation here!)
        system_tokens = torch.tensor(template["system_tokens"], dtype=torch.long)
        user_prefix_tokens = torch.tensor(template["user_prefix_tokens"], dtype=torch.long)
        user_suffix_tokens = torch.tensor(template["user_suffix_tokens"], dtype=torch.long)
        force_start_tokens = torch.tensor(template["force_start_tokens"], dtype=torch.long)

        # OPTIMIZATION: Batch tokenize all user inputs at once
        # This is faster than tokenizing one by one
        user_contents_only = tokenizer(
            user_inputs,
            add_special_tokens=False,
            padding=False,  # We'll pad manually later
            truncation=False,
        )["input_ids"]

        # Build full sequences: [system] + [user_prefix] + [content] + [user_suffix] + [force_start]
        concatenated_input_ids = []
        for content_tokens in user_contents_only:
            content_tensor = torch.tensor(content_tokens, dtype=torch.long)
            full_sequence = torch.cat([
                system_tokens,
                user_prefix_tokens,
                content_tensor,        # ONLY this varies per input
                user_suffix_tokens,
                force_start_tokens,
            ])
            concatenated_input_ids.append(full_sequence)

        # Efficient padding with pre-allocation
        max_len = max(seq.size(0) for seq in concatenated_input_ids)
        batch_size = len(concatenated_input_ids)
        
        # Pre-allocate tensors (faster than appending to list)
        input_ids_padded = torch.full(
            (batch_size, max_len), 
            tokenizer.pad_token_id, 
            dtype=torch.long
        )
        attention_masks = torch.zeros((batch_size, max_len), dtype=torch.long)
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        for i, seq in enumerate(concatenated_input_ids):
            seq_len = seq.size(0)
            input_lengths[i] = seq_len
            
            # Left padding for decoder-only models
            pad_len = max_len - seq_len
            input_ids_padded[i, pad_len:] = seq
            attention_masks[i, pad_len:] = 1

        # Move to device once
        input_ids = input_ids_padded.to(device)
        attention_mask = attention_masks.to(device)

        # Use pre-computed generation config
        gen_config = template["generation_config"]
        if generation_config:
            # Create new config to avoid modifying cached one
            gen_config = GenerationConfig(**{**gen_config.to_dict(), **generation_config})
        
        # Add stop sequences
        gen_config.stop_strings = template["stop_sequences"]

        # Generate
        start_time = time.perf_counter()
        
        try:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                tokenizer=tokenizer,
                return_dict_in_generate=True,
                output_scores=False,
                generation_config=gen_config,
            )
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            error_results = [self._create_error_result(inp, ut_steps, str(e)) for inp in user_inputs]
            return error_results[0] if is_single else error_results

        total_generation_time = time.perf_counter() - start_time
        
        # Process results (this part is still per-sample)
        results = []
        for i in range(len(user_inputs)):
            generated_ids = outputs.sequences[i, input_ids.shape[1]:]
            
            # Remove padding
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is not None:
                non_pad_mask = generated_ids != pad_token_id
                if non_pad_mask.any():
                    last_non_pad = non_pad_mask.nonzero()[-1].item() + 1
                    generated_ids = generated_ids[:last_non_pad]
            
            # Decode
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            full_response = template["force_start_text"] + " " + generated_text
            
            actual_generated_tokens = len(generated_ids)
            
            # Quality checks
            self.check_repeated_outputs_and_abort(full_response)
            is_degenerate = self._detect_degenerate_output(full_response)

            if is_degenerate:
                pred = "DEGENERATE"
            else:
                pred = self._extract_final_answer(full_response, task_type)

            result = {
                "full_response": full_response,
                "prediction": pred,
                "generation_time": total_generation_time,
                "generated_tokens": actual_generated_tokens,
                "input_tokens": input_lengths[i].item(),
                "ut_steps": ut_steps,
                "is_degenerate": is_degenerate,
                "test_input": user_inputs[i],
            }
            results.append(result)

        return results[0] if is_single else results
    def _extract_final_answer(self, full_response: str, task_type: str) -> str:
        """Extract final answer with improved parsing, aligned with prompt templates"""
        pred = "0"
        
        try:
            full_response = full_response.strip()
            
            # First, try to extract everything before <END> if it exists
            if "<END>" in full_response:
                full_response = full_response.split("<END>")[0].strip()
            
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
            
            else:  # n_ary and igsm
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
                    # Fallback: get last number from last non-empty line
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
            "n_ary": 16,
            "p_hop": 16,
            "igsm": 16,
        }
        
        return {
            "max_new_tokens": task_token_limits.get(task_type, 256),
            "min_new_tokens": 10,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "temperature": 0.7,
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