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

class OuroThinkingExperiment:
    """Core experiment class for Ouro model testing"""

    def __init__(
        self,
        model_path: str,
        dtype=torch.float16,
        use_4bit_quant: bool = False,
        use_torch_compile: bool = False,
    ):
        # Clear CUDA cache before loading new model
        torch.cuda.empty_cache()
        self.model_path = model_path
        self.dtype = dtype
        self.use_4bit_quant = use_4bit_quant
        self.use_torch_compile = use_torch_compile
        self.tokenizer = None
        self.task_templates = {}

    def load_model_with_ut_steps(
        self, total_ut_steps: int, early_exit_threshold: float = 1.0
    ):
        """Load model with specific UT steps configuration"""
        quantization_config = None
        if self.use_4bit_quant:
            print("→ Applying 4-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        print(f"\n{'='*60}")
        print(f"⚙️  LOADING MODEL CONFIGURATION")
        print(f"{'='*60}")
        print(f"Model Path: {self.model_path}")
        print(f"Requested UT Steps: {total_ut_steps}")
        print(f"Early Exit Threshold: {early_exit_threshold}")
        print(f"Data Type: {self.dtype}")
        print(f"4-bit Quantization: {self.use_4bit_quant}")
        print(f"Torch Compile: {self.use_torch_compile}")

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
        base_config.early_exit_threshold = early_exit_threshold
        print(f"\n→ Modified config:")
        print(f"   New UT steps: {base_config.total_ut_steps}")
        print(f"   New early exit: {base_config.early_exit_threshold}")
                
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

        # Load model with modified config
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

        if self.use_torch_compile:
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
            print(f"\n⚠️  WARNING: UT STEPS MISMATCH!")
            print(f"   Requested: {total_ut_steps}")
            print(f"   Actual: {model.config.total_ut_steps}")
        
        print(f"{'='*60}\n")
        
        return model, tokenizer, base_config, {
            "total_ut_steps": total_ut_steps,
            "early_exit_threshold": early_exit_threshold,
        }

    def _build_task_templates(self, tokenizer):
        """Pre-compute prompt templates for faster inference"""
        self.tokenizer = tokenizer
        
        task_configs = {
            "n_ary": {
                "system": "You are a precise arithmetic calculator. Follow the exact step-by-step format shown in the example. Output only the calculation steps, nothing else.",
                "example_user": "10 + 20 + 30 =",
                "example_asst": "[STEP 1] Current: 0\n[STEP 2] Add 10: 0 + 10 = 10\n[STEP 3] Current: 10\n[STEP 4] Add 20: 10 + 20 = 30\n[STEP 5] Current: 30\n[STEP 6] Add 30: 30 + 30 = 60\n[FINAL] 60",
                "force_start": "[STEP 1]",
            },
            
            "p_hop": {
                "system": "You are a sequence tracer. Follow the exact format shown in the example. Trace through the sequence step-by-step and output only the final answer.",
                "example_user": "Sequence: A B C D A B. Start: A. Hop 1 times.",
                "example_asst": "[TRACE] Start at A. Found 'A' in sequence. Next token is B.\n[FINAL] B",
                "force_start": "[TRACE]",
            },
            
            "igsm": {
                "system": "You are a symbolic equation solver. Solve equations modulo 7. Follow the exact format shown in the example with numbered equations.",
                "example_user": "Question. E#I := 4. E#J := E#I. F#K := E#J. H#J := E#J + F#K. H#J?",
                "example_asst": "[EQ 1] E#I = 4. [EQ 2] E#J = E#I. ==> E#J = 4. [EQ 3] F#K = E#J. ==> F#K = 4. [EQ 4] H#J = E#J + F#K. ==> H#J = 1.\n[FINAL] 1",
                "force_start": "[EQ 1]",
            }
        }
        
        self.task_templates = {}
        
        for task_type, config in task_configs.items():
            # Build static context with chat template
            static_messages = [
                {"role": "system", "content": config["system"]},
                {"role": "user", "content": config["example_user"]},
                {"role": "assistant", "content": config["example_asst"]}
            ]
            
            static_prompt_text = tokenizer.apply_chat_template(
                static_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Remove any trailing whitespace
            static_prompt_text = static_prompt_text.rstrip()
            
            static_inputs = tokenizer(static_prompt_text, return_tensors="pt")
            
            # Tokenize force start separately
            force_start_tokens = tokenizer(
                config["force_start"], 
                return_tensors="pt", 
                add_special_tokens=False
            )
            
            self.task_templates[task_type] = {
                "static_input_ids": static_inputs.input_ids,
                "static_attention_mask": static_inputs.attention_mask,
                "force_start_ids": force_start_tokens.input_ids,
                "force_start_text": config["force_start"],
                "system_content": config["system"],
            }
        
        print("[+] Task templates pre-computed successfully")

    def _extract_final_answer(self, full_response: str, task_type: str) -> str:
        """Extract final answer using robust regex patterns"""
        pred = "0"
        
        try:
            full_response = full_response.strip()
            
            if task_type == "p_hop":
                # For p_hop, look for letter answers
                patterns = [
                    r'\[FINAL\]\s*([A-D])\b',
                    r'Final:\s*([A-D])\b',
                    r'Next token is\s*([A-D])\b',
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
                # For math tasks, look for numbers
                patterns = [
                    r'\[FINAL\]\s*(\d+)',
                    r'Final:\s*(\d+)',
                    r'=\s*(\d+)\s*$',
                    r'\b(\d+)\s*$',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, full_response)
                    if matches:
                        pred = matches[-1]
                        break
                else:
                    # Last resort: find any number in last line
                    lines = [l.strip() for l in full_response.split('\n') if l.strip()]
                    if lines:
                        last_line = lines[-1]
                        numbers = re.findall(r'\b(\d+)\b', last_line)
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
        """Detect if output is degenerate/chaotic/garbage
           Super important because this model is highly unstable and has just released on 2025 October"""
        if not text or len(text.strip()) < 5:
            return True
        
        # Check for excessive newlines
        if text.count('\n\n\n') > 3:
            return True
        
        # Check for excessive brackets
        bracket_ratio = (text.count('[') + text.count(']')) / max(len(text), 1)
        if bracket_ratio > 0.3:
            return True
        
        # Check for very repetitive content
        if len(text) > 100:
            unique_chars = len(set(text))
            if unique_chars < 10:
                return True
        
        # Check for excessive whitespace
        whitespace_ratio = (text.count(' ') + text.count('\n')) / max(len(text), 1)
        if whitespace_ratio > 0.7:
            return True
        
        # Check for single character repetition
        if len(text) > 50:
            for char in ['[', ']', '\n', ' ', '.']:
                if text.count(char) > len(text) * 0.4:
                    return True
        
        return False

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
        
        # Verify model config
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
            }

        # Build templates if needed
        if not hasattr(self, "task_templates") or task_type not in self.task_templates:
            self._build_task_templates(tokenizer)

        template = self.task_templates[task_type]
        device = model.device

        # Construct input sequence
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

        # Generation config
        default_config = {
            "max_new_tokens": 512,
            "min_new_tokens": 5,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "early_stopping": True,
        }
        
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
            }

        generation_time = time.perf_counter() - start_time

        # Extract generated text
        prompt_length = input_ids.shape[1]
        generated_ids = outputs.sequences[0, prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Construct full response
        full_response = template["force_start_text"] + " " + generated_text
        
        # Check for degenerate output
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

            if i > 0:
                context_len = input_slice.size(1) - stride
                if context_len > 0:
                    target_slice[:, :context_len] = -100

            if (target_slice != -100).sum() == 0:
                continue

            outputs = model(
                input_ids=input_slice,
                attention_mask=attention_mask[:, i:end_loc],
                labels=target_slice,
            )

            if torch.isnan(outputs.loss):
                continue

            num_valid = (target_slice != -100).sum().item()
            total_loss += (outputs.loss * num_valid).item()
            total_tokens += num_valid

        if total_tokens == 0:
            return float("nan"), float("nan")

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity, avg_loss


class OuroBatchExperiment(OuroThinkingExperiment):
    """Extended experiment class with batch processing"""

    def __init__(
        self,
        model_path: str,
        dtype=torch.float16,
        use_4bit_quant: bool = False,
        use_torch_compile: bool = False,
        max_batch_size: int = 4,
        max_new_tokens: int = 512,
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
        
        # Check if model supports batch generation
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
        
        # Map outputs to prompts
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
        
        # Fill missing results
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

        # Check for garbage output
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