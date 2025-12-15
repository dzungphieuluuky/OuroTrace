import torch
import time
import re
from typing import List, Optional
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
        """Load model with specific UT steps configuration - COMPLETE FIX"""
        quantization_config = None
        if self.use_4bit_quant:
            print("→ Applying 4-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        print(f"Loading model: UT steps={total_ut_steps}, Early exit={early_exit_threshold}")

        # CRITICAL: Load the base config first
        base_config = AutoConfig.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        # Apply the UT step configuration BEFORE loading the model
        base_config.total_ut_steps = total_ut_steps
        base_config.early_exit_threshold = early_exit_threshold
                
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with the COMPLETE modified config
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=base_config,
            device_map="cuda",
            attn_implementation="sdpa_paged",
            torch_dtype=self.dtype if not self.use_4bit_quant else None,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        if self.use_torch_compile:
            print("→ Applying torch.compile()")
            model = torch.compile(model)

        model.eval()
        print(f"✅ Model loaded on {model.device}")
        
        # VERIFICATION: Check the config was applied
        print(f"✅ VERIFIED: Model configured with total_ut_steps = {model.config.total_ut_steps}")
        
        return model, tokenizer, base_config, {
            "total_ut_steps": total_ut_steps,
            "early_exit_threshold": early_exit_threshold,
        }

    def _build_task_templates(self, tokenizer):
        """
        FIXED: Improved few-shot examples to prevent babbling and ensure proper reasoning.
        Key fixes:
        1. Use 3-shot examples that match exact test format
        2. Ensure examples show COMPLETE reasoning paths
        3. Make force_start more explicit
        4. Add stronger guardrails against repetition
        """
        self.tokenizer = tokenizer
        
        task_configs = {
            # 1. N-ARY ADDITION - FIXED: Show all numbers, handle leading zeros
            "n_ary": {
                "system": "You are a mechanical calculation engine. Your output MUST be strictly sequential. DO NOT repeat steps, start over, or output the input question. Only output the calculation steps once.",
                "few_shots": [
                    {
                        "role": "user",
                        "content": "100 + 200 + 300 =",
                        "role_response": "[STEP 1] Current: 0.\n[STEP 2] Add 100: 0 + 100 = 100.\n[STEP 3] Current: 100.\n[STEP 4] Add 200: 100 + 200 = 300.\n[STEP 5] Current: 300.\n[STEP 6] Add 300: 300 + 300 = 600.\n[FINAL] 600.\n"
                    },
                    # {
                    #     "role": "user",
                    #     "content": "050 + 025 + 100 =",
                    #     "role_response": "[STEP 1] Current: 0.\n[STEP 2] Add 050: 0 + 50 = 50.\n[STEP 3] Current: 50.\n[STEP 4] Add 025: 50 + 25 = 75.\n[STEP 5] Current: 75.\n[STEP 6] Add 100: 75 + 100 = 175.\n[FINAL] 175.\n"
                    # },
                ],
                "force_start": "[STEP 1] Current: 0.\n",
                "input_prefix": ""
            },
            
            # 2. P-HOP INDUCTION - FIXED: Show complete tracing for multiple hops
            "p_hop": {
                "system": "You are an induction head mechanism. Trace EXACTLY the requested number of hops. Find each token's position in the sequence. Stop after the required hops. DO NOT output anything after the final answer.",
                "few_shots": [
                    {
                        "role": "user",
                        "content": "Sequence: DCBADC. Start: D. Hop 2 times.",
                        "role_response": "[TRACE] Start at D.\n[TRACE] Found 'D' at position 0. Next token is C.\n[TRACE] Found 'C' at position 1. Next token is B.\n[FINAL] B.\n"
                    },
                    {
                        "role": "user",
                        "content": "Sequence: AABBCC. Start: A. Hop 3 times.",
                        "role_response": "[TRACE] Start at A.\n[TRACE] Found 'A' at position 0. Next token is A.\n[TRACE] Found 'A' at position 1. Next token is B.\n[TRACE] Found 'B' at position 2. Next token is B.\n[FINAL] B.\n"
                    }
                ],
                "force_start": "[TRACE] Start at",
                "input_prefix": ""
            },
            
            # 3. SYMBOLIC i-GSM - FIXED: Show actual modulo 7 calculations
            "igsm": {
                "system": "You are a symbolic math solver. Solve equations modulo 7 (results 0-6). For each equation, show the substitution and modulo 7 calculation. STOP after solving for the target variable.",
                "few_shots": [
                    {
                        "role": "user",
                        "content": "Question. X#Y := 3. Z#Z := X#Y * 2. Z#Z?",
                        "role_response": "[EQ 1] X#Y = 3.\n[EQ 2] Z#Z = X#Y * 2 = 3 * 2 = 6.\n[FINAL] 6.\n"
                    },
                    {
                        "role": "user",
                        "content": "Question. B#K := 1. L#L := B#K - 5. L#L?",
                        "role_response": "[EQ 1] B#K = 1.\n[EQ 2] L#L = B#K - 5 = 1 - 5 = -4 mod 7 = 3.\n[FINAL] 3.\n"
                    },
                    {
                        "role": "user",
                        "content": "Question. C#D := 5. E#F := C#D + 4. G#H := E#F * 2. G#H?",
                        "role_response": "[EQ 1] C#D = 5.\n[EQ 2] E#F = C#D + 4 = 5 + 4 = 9 mod 7 = 2.\n[EQ 3] G#H = E#F * 2 = 2 * 2 = 4.\n[FINAL] 4.\n"
                    }
                ],
                "force_start": "[EQ 1]",
                "input_prefix": ""
            }
        }
        
        self.task_templates = {}
        
        for task_type, config in task_configs.items():
            # Build messages list with System + Few-Shots
            messages = [{"role": "system", "content": config["system"]}]
            
            for shot in config["few_shots"]:
                messages.append({"role": "user", "content": shot["content"]})
                messages.append({"role": "assistant", "content": shot["role_response"]})
            
            static_prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            static_inputs = tokenizer(static_prompt_text, return_tensors="pt")
            
            # Tokenize Force Start
            force_start_tokens = tokenizer(
                config["force_start"],
                return_tensors="pt",
                add_special_tokens=False
            )
            
            self.task_templates[task_type] = {
                "static_input_ids": static_inputs.input_ids,
                "static_attention_mask": static_inputs.attention_mask,
                "force_start_ids": force_start_tokens.input_ids,
                "input_prefix": config["input_prefix"],
                "force_start_text": config["force_start"]
            }
        
        print("[+] FIXED Task templates with improved few-shot examples")

    def _extract_final_answer(self, full_response: str, task_type: str) -> str:
        """Extract answer from model response"""
        pred = "0"
        
        try:
            # First, clean the response
            clean_response = full_response.strip()
            
            if task_type == "p_hop":
                # Look for [FINAL] marker
                if "[FINAL]" in clean_response:
                    final_part = clean_response.split("[FINAL]")[-1].strip()
                    # Extract just the letter
                    match = re.search(r'(\w+)', final_part)
                    if match:
                        pred = match.group(1).strip()
                    else:
                        pred = "Error"
                else:
                    # Fallback to old patterns
                    patterns = [
                        r"Final\s*:\s*(\w+)",
                        r"Next token is\s*(\w+)",
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, clean_response, re.IGNORECASE)
                        if match:
                            pred = match.group(1).strip()
                            break
                    else:
                        pred = "Error"
                        
            else:  # n_ary or igsm
                # Look for [FINAL] marker
                if "[FINAL]" in clean_response:
                    final_part = clean_response.split("[FINAL]")[-1].strip()
                    # Extract number
                    match = re.search(r'([-+]?\d*\.?\d+)', final_part)
                    if match:
                        pred = match.group(1).strip()
                    else:
                        # Try to find last number in the response
                        numbers = re.findall(r'=\s*([-+]?\d+)', clean_response)
                        if numbers:
                            pred = numbers[-1]
                        else:
                            pred = "Error"
                else:
                    # Fallback patterns
                    patterns = [
                        r"=\s*([-+]?\d+)$",
                        r"=\s*([-+]?\d+)\s*$",
                        r"Answer\s*:\s*([-+]?\d+)",
                    ]
                    all_matches = []
                    for pattern in patterns:
                        matches = re.findall(pattern, clean_response, re.IGNORECASE)
                        all_matches.extend(matches)
                    
                    if all_matches:
                        pred = all_matches[-1]
                        
        except Exception as e:
            print(f"[!] Parsing error: {e}")
            pred = "ParseError"
        
        return pred

    @torch.no_grad()
    def predict_with_metrics_optimized(
        self,
        user_input: str,
        task_type: str,
        model,
        tokenizer,
        ut_steps: int,
        generation_config: dict = None,
    ):
            # VERIFY model has the right config
        if not hasattr(model.config, 'total_ut_steps'):
            print("❌ ERROR: Model missing total_ut_steps config!")
            return {"error": "Bad model config", "prediction": "0"}

        """Optimized prediction with repetition penalty to prevent loops"""
        if not hasattr(self, "task_templates") or task_type not in self.task_templates:
            self._build_task_templates(tokenizer)

        template = self.task_templates[task_type]
        device = model.device

        input_ids = template["static_input_ids"].to(device)
        user_query = template["input_prefix"] + user_input
        user_tokens = tokenizer(
            user_query, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        force_start_ids = template["force_start_ids"].to(device)

        input_ids = torch.cat([input_ids, user_tokens, force_start_ids], dim=1)
        attention_mask = torch.ones_like(input_ids, device=device)

        start_time = time.perf_counter()

        gen_config = generation_config or {
            "max_new_tokens": 1024,
            "do_sample": False,
            "num_beams": 1,
            "min_length": 5,
            "repetition_penalty": 1.2
        }

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
            **gen_config,
        )

        generation_time = time.perf_counter() - start_time

        prompt_length = input_ids.shape[1]
        generated_ids = outputs.sequences[0, prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        full_response = template["force_start_text"] + generated_text
        pred = self._extract_final_answer(full_response, task_type)

        return {
            "full_response": full_response,
            "prediction": pred,
            "generation_time": generation_time,
            "generated_tokens": generated_ids.shape[0],
            "input_tokens": input_ids.shape[1],
            "ut_steps": ut_steps,
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
            text_concat, return_tensors="pt", max_length=max_length * 2, truncation=True
        )
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        if input_ids.size(1) < 2:
            return float("nan"), float("nan")

        total_loss = 0.0
        total_tokens = 0

        for i in tqdm(
            range(0, input_ids.size(1), stride), desc=f"Calculating PPL (UT={ut_steps})"
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
        max_new_tokens: int = 1024,
    ):
        super().__init__(model_path, dtype, use_4bit_quant, use_torch_compile)
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens

    def prepare_batch_inputs(self, prompts: List[str], task_type: str) -> List[List[int]]:
        """Prepare inputs for batch generation"""
        if task_type not in self.task_templates:
            raise ValueError("Templates not built. Call _build_task_templates first.")
        
        template = self.task_templates[task_type]
        
        batch_texts = [template["input_prefix"] + p for p in prompts]
        user_encodings = self.tokenizer(batch_texts, add_special_tokens=False)
        
        static_ids = template["static_input_ids"].squeeze(0).tolist()
        force_ids = template["force_start_ids"].squeeze(0).tolist()
        
        input_id_lists = []
        for user_ids in user_encodings['input_ids']:
            full_seq = static_ids + user_ids + force_ids
            input_id_lists.append(full_seq)
        
        return input_id_lists

    @torch.no_grad()
    def batch_predict_with_metrics(self, prompts: List[str], task_type: str,
                                   model, tokenizer, ut_steps: int,
                                   generation_config: Optional[GenerationConfig] = None):
        """Batch prediction with metrics"""
        if not prompts:
            return []
        
        simple_batch_inputs = self.prepare_batch_inputs(prompts, task_type)
        input_lengths = [len(ids) for ids in simple_batch_inputs]
        
        if not hasattr(model, 'generate_batch'):
            print("⚠️ Model doesn't support generate_batch(). Using sequential.")
            return self._sequential_fallback(prompts, task_type, model, 
                                            tokenizer, ut_steps, generation_config)
        
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                use_cuda_graph=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                max_batch_tokens=self.max_batch_size * self.max_new_tokens,
            )
        
        start_time = time.perf_counter()
        
        try:
            batch_outputs = model.generate_batch(
                inputs=simple_batch_inputs,
                generation_config=generation_config,
            )
        except Exception as e:
            print(f"⚠️ generate_batch failed: {e}. Falling back.")
            return self._sequential_fallback(prompts, task_type, model, 
                                            tokenizer, ut_steps, generation_config)
        
        batch_time = time.perf_counter() - start_time
        
        template = self.task_templates[task_type]
        results = [None] * len(prompts)
        request_ids = list(batch_outputs.keys())
        
        # Map outputs to prompts
        # if all(isinstance(rid, int) for rid in request_ids):
        #     for request_id in request_ids:
        #         if 0 <= request_id < len(prompts):
        #             output = batch_outputs[request_id]
        #             results[request_id] = self._process_single_output(
        #                 output, request_id, input_lengths[request_id],
        #                 template, tokenizer, task_type, ut_steps,
        #                 batch_time / len(prompts)
        #             )
        if all(isinstance(rid, str) for rid in request_ids):
            # String/UUID request IDs
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
                    'ut_steps': ut_steps
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
        """Process single batch output"""
        if hasattr(output, "generated_tokens"):
            generated_ids = output.generated_tokens
        elif hasattr(output, "sequences") and len(output.sequences) > 0:
            generated_ids = output.sequences[0]
        else:
            generated_ids = []

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_response = template["force_start_text"] + generated_text

        pred = self._extract_final_answer(full_response, task_type)

        return {
            "full_response": full_response,
            "prediction": pred,
            "generation_time": sample_time,
            "generated_tokens": len(generated_ids),
            "input_tokens": input_length,
            "ut_steps": ut_steps,
            "prompt_idx": prompt_idx,
        }

    def _sequential_fallback(
        self, prompts, task_type, model, tokenizer, ut_steps, generation_config
    ):
        """Fallback to sequential processing"""
        results = []
        for prompt in tqdm(prompts, desc=f"Sequential fallback ({task_type})"):
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