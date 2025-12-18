import pandas as pd
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
from.safe_optimization import apply_all_safe_optimizations

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

            torch.cuda.empty_cache()
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
        
        # Final verification
        print(f"\n{'='*60}")
        print(f"✅ MODEL LOADED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Device: {model.device}")
        print(f"Model dtype: {model.dtype}")
        print(f"VERIFIED UT steps: {model.config.total_ut_steps}")
        print(f"VERIFIED early exit: {model.config.early_exit_threshold}")
        
        if model.config.total_ut_steps != total_ut_steps:
            print(f"\n WARNING: UT STEPS MISMATCH!")
            print(f"   Requested: {total_ut_steps}")
            print(f"   Actual: {model.config.total_ut_steps}")
        
        print(f"{'='*60}\n")
        
        # APPLY SAFE OPTIMIZATIONS (especially important for UT > 1)
        opt_result = apply_all_safe_optimizations(
            model, tokenizer, total_ut_steps
        )
        model = opt_result["model"]

        return model, tokenizer, base_config, {
            "total_ut_steps": total_ut_steps,
            "early_exit_threshold": base_config.early_exit_threshold,
        }

    def _build_task_templates(self, tokenizer):
        """
        Pre-compute prompt templates with enhanced system prompts for p_hop and igsm.
        """
        self.tokenizer = tokenizer

        task_configs = {
            "n_ary": {
                "system": (
                    "You are a calculator. Given an addition problem with several numbers (e.g., '{number_1} + {number_2} + {number_3} + ... ='), "
                    "show your work step by step. For each number, add it to the running total and show the calculation. "
                    "After all steps, output only the final sum on a new line as [FINAL] [sum].\n"
                    "Example:\n"
                    "Input: {number_i} + {number_i+1} + {number_i+2} + ... =\n"
                    "Output:\n"
                    "Step {i}: 0 + {number_i} = {sum_i}\n"
                    "Step {i+1}: {sum_i} + {number_i+1} = {sum_i+1}\n"
                    "Step {i+2}: {sum_i+1} + {number_i+2} = {sum_i+2}\n"
                    "..."
                    "[FINAL] {final_sum}"
                ),
                "force_start": "\nStep 1:",
            },
            "p_hop": {
                # Data format: "Sequence: A D C B... Start: A. Hop 16 times."
                "system": (
                    "You are a sequence tracer. You will be given a sequence of tokens, a starting node, and a number of hops.\n"
                    "Rule: Find the current node in the sequence and move to the very next token. If you are at the end, wrap around to the start.\n"
                    "STRICT: Do not repeat the input sequence. Do not list the alphabet.\n\n"
                    "Format each hop: 'Hop {X}: At {current} -> Next is {target}'\n"
                    "End with: '[FINAL] {token}'\n\n"
                    "Example:\n"
                    "Input: Sequence: A B D C. Start: A. Hop 2 times.\n"
                    "Hop 1: At A -> Next is B\n"
                    "Hop 2: At B -> Next is D\n"
                    "[FINAL] D"
                ),
                "force_start": "\nHop 1:",
            },
            "igsm": {
                # Data format: "Question. E#I := 4. E#J := E#I. F#K := E#J + F#K. H#J?"
                "system": (
                    "You are a symbolic math solver working strictly in Modulo 7 (results 0-6).\n"
                    "You will receive a list of variable assignments using ':=' and a final query.\n"
                    "Rule: Solve for variables in order of dependency. Reduce every addition, subtraction, or multiplication by Modulo 7.\n\n"
                    "Format: 'Step {X}: {var} = {substituted expression} = {value} (mod 7)'\n"
                    "End with: '[FINAL] {value}'\n\n"
                    "Example:\n"
                    "Input: Question. A#B := 3. C#D := A#B + 5. C#D?\n"
                    "Step 1: A#B = 3 = 3 (mod 7)\n"
                    "Step 2: C#D = 3 + 5 = 1 (mod 7)\n"
                    "[FINAL] 1"
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

        print("[+] Task templates pre-computed with enhanced p_hop and igsm logic.")

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

    @torch.no_grad()
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
        
        Args:
            user_inputs: Single string or list of strings
            task_type: Type of task (n_ary, p_hop, igsm)
            model: The model to use for generation
            tokenizer: The tokenizer
            ut_steps: Number of UT steps
            generation_config: Optional generation config overrides
            enable_batch: Whether to enable batch processing (only for UT=1)
            
        Returns:
            Single dict if user_inputs is str, list of dicts if user_inputs is list
        """
        # Handle single input case
        is_single = isinstance(user_inputs, str)
        if is_single:
            user_inputs = [user_inputs]
        
        # Validate model config
        if not hasattr(model.config, 'total_ut_steps'):
            print("❌ ERROR: Model missing total_ut_steps config!")
            error_results = [self._create_error_result(inp, ut_steps) for inp in user_inputs]
            return error_results[0] if is_single else error_results
        
        # Build templates if needed
        if not hasattr(self, "task_templates") or task_type not in self.task_templates:
            self._build_task_templates(tokenizer)
        
        template = self.task_templates[task_type]
        device = model.device
        
        # Prepare batch inputs following HuggingFace docs
        batch_size = len(user_inputs)
        
        # Get static template components
        static_ids = template["static_input_ids"].squeeze(0)  # Remove batch dim
        force_start_ids = template["force_start_ids"].squeeze(0)  # Remove batch dim
        
        # Tokenize all user inputs (without special tokens)
        user_tokens_list = tokenizer(
            user_inputs,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,  # Don't pad yet, we'll do it after concatenation
        ).input_ids
        
        # Build full sequences: [static + user_input + force_start] for each sample
        input_ids_list = []
        for user_tokens in user_tokens_list:
            full_seq = torch.cat([static_ids, user_tokens, force_start_ids], dim=0)
            input_ids_list.append(full_seq)
        
        # Pad sequences to same length (following HF docs: left padding for generation)
        # Since tokenizer has padding_side="left", this will pad on the left
        max_length = max(seq.shape[0] for seq in input_ids_list)
        
        padded_input_ids = []
        attention_masks = []
        
        for seq in input_ids_list:
            pad_length = max_length - seq.shape[0]
            if pad_length > 0:
                # Left padding
                padded_seq = torch.cat([
                    torch.full((pad_length,), tokenizer.pad_token_id, dtype=seq.dtype, device=device),
                    seq.to(device)
                ], dim=0)
                attention_mask = torch.cat([
                    torch.zeros(pad_length, dtype=torch.long, device=device),
                    torch.ones(seq.shape[0], dtype=torch.long, device=device)
                ], dim=0)
            else:
                padded_seq = seq.to(device)
                attention_mask = torch.ones(seq.shape[0], dtype=torch.long, device=device)
            
            padded_input_ids.append(padded_seq)
            attention_masks.append(attention_mask)
        
        # Stack into batch tensors
        input_ids = torch.stack(padded_input_ids, dim=0)  # [batch_size, max_length]
        attention_mask = torch.stack(attention_masks, dim=0)  # [batch_size, max_length]
        
        # Get optimal config for task type
        default_config = self._get_optimal_generation_config(task_type)
        if generation_config:
            default_config.update(generation_config)
        
        # Start timing
        start_time = time.perf_counter()
        
        # Generate
        try:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
                **default_config,
            )
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            error_results = [self._create_error_result(inp, ut_steps, str(e)) for inp in user_inputs]
            return error_results[0] if is_single else error_results
        
        generation_time = time.perf_counter() - start_time
        
        # Process outputs for each sample in batch
        results = []
        for i in range(batch_size):
            # Get the prompt length for this sample (excluding padding)
            prompt_length = attention_mask[i].sum().item()
            
            # Extract generated tokens (everything after the prompt)
            generated_ids = outputs.sequences[i, prompt_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Construct full response
            full_response = template["force_start_text"] + " " + generated_text
            
            # Check for repeated outputs
            self.check_repeated_outputs_and_abort(full_response)
            
            # Detect garbage
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
                "generation_time": generation_time / batch_size,  # Distribute time across batch
                "generated_tokens": generated_ids.shape[0],
                "input_tokens": prompt_length,
                "ut_steps": ut_steps,
                "is_degenerate": is_degenerate,
                "test_input": user_inputs[i],
            }
            results.append(result)
        
        # Return single result or list based on input
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
            
            torch.cuda.empty_cache()
            raise SystemExit(f"Experiment failed: {failure.reason}")