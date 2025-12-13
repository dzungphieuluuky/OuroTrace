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
        self, total_ut_steps: int, early_exit_threshold: float
    ):
        """Load model with specific UT steps configuration"""
        quantization_config = None
        if self.use_4bit_quant:
            print("→ Applying 4-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        print(
            f"Loading model: UT steps={total_ut_steps}, Early exit={early_exit_threshold}"
        )

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        config.total_ut_steps = total_ut_steps
        config.early_exit_threshold = early_exit_threshold

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            device_map="cuda",
            attn_implementation="sdpa_paged",
            dtype=self.dtype if not self.use_4bit_quant else None,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )

        if self.use_torch_compile:
            print("→ Applying torch.compile()")
            model = torch.compile(model)

        model.eval()
        print(f"✅ Model loaded on {model.device}")

        return (
            model,
            tokenizer,
            None,
            {
                "total_ut_steps": total_ut_steps,
                "early_exit_threshold": early_exit_threshold,
            },
        )

    def _build_task_templates(self, tokenizer):
        """
        Pre-compute prompt templates for faster inference.
        UPDATED: Added spaces/newlines to force_start to prevent token gluing/contamination.
        """
        self.tokenizer = tokenizer

        task_configs = {
            "n_ary": {
                "system": "You are a mechanical calculation engine. Output the accumulation steps strictly.",
                "example_user": "10 + 20 + 30 =",
                "example_asst": "Current: 0\nAdd 10: 0 + 10 = 10\nCurrent: 10\nAdd 20: 10 + 20 = 30\nCurrent: 30\nAdd 30: 30 + 30 = 60\nFinal: 60",
                "force_start": " Current: 0",
                "input_prefix": "",
            },
            "p_hop": {
                "system": "You are an induction head mechanism. Trace the sequence occurrences step-by-step.",
                "example_user": "Sequence: A B C D A B. Start: A. Hop 1 times.",
                "example_asst": "\nStart at A. Found 'A' in sequence. Next token is B. Final: B",
                "force_start": "\nStart at",
                "input_prefix": "",
            },
            "igsm": {
                "system": "You are a symbolic math solver. Solve the DAG modulo 7.",
                "example_user": "Question. E#I := 4. E#J := E#I. F#K := E#J. H#J := E#J + F#K. H#J?",
                "example_asst": "\nAnswer with CoT. E#I = 4. ==> E#I = 4. E#J = E#I. ==> E#J = 4. F#K = E#J. ==> F#K = 4. H#J = E#J + F#K. ==> H#J = 1. Final: 1",
                "force_start": "\nAnswer with CoT.",
                "input_prefix": "",
            },
        }

        for task_type, config in task_configs.items():
            static_messages = [
                {"role": "system", "content": config["system"]},
                {"role": "user", "content": config["example_user"]},
                {"role": "assistant", "content": config["example_asst"]},
            ]

            static_prompt_text = tokenizer.apply_chat_template(
                static_messages, tokenize=False, add_generation_prompt=True
            )
            static_inputs = tokenizer(static_prompt_text, return_tensors="pt")

            force_start_tokens = tokenizer(
                config["force_start"], return_tensors="pt", add_special_tokens=False
            )

            self.task_templates[task_type] = {
                "static_input_ids": static_inputs.input_ids,
                "static_attention_mask": static_inputs.attention_mask,
                "force_start_ids": force_start_tokens.input_ids,
                "input_prefix": config["input_prefix"],
                "force_start_text": config["force_start"],
            }

        print("[+] Task templates pre-computed (Corrected with spacers)")

    def _extract_final_answer(self, full_response: str, task_type: str) -> str:
        """Extract answer from model response"""
        pred = "0"

        try:
            if task_type == "p_hop":
                patterns = [
                    r"Final\s*:\s*(\w+)",
                    r"Next token is\s*(\w+)",
                    r"Answer\s*:\s*(\w+)",
                ]
                for pattern in patterns:
                    match = re.search(pattern, full_response, re.IGNORECASE)
                    if match:
                        pred = match.group(1).strip()
                        break
                else:
                    pred = "Error"
            else:
                patterns = [
                    r"Final\s*:\s*([-+]?\d*\.?\d+)",
                    r"Answer\s*:\s*([-+]?\d*\.?\d+)",
                    r"=\s*([-+]?\d*\.?\d+)$",
                ]
                all_matches = []
                for pattern in patterns:
                    matches = re.findall(pattern, full_response, re.IGNORECASE)
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

    def prepare_batch_inputs(
        self, prompts: List[str], task_type: str
    ) -> List[List[int]]:
        """Prepare inputs for batch generation"""
        if task_type not in self.task_templates:
            raise ValueError("Templates not built. Call _build_task_templates first.")

        template = self.task_templates[task_type]

        batch_texts = [template["input_prefix"] + p for p in prompts]
        user_encodings = self.tokenizer(batch_texts, add_special_tokens=False)

        static_ids = template["static_input_ids"].squeeze(0).tolist()
        force_ids = template["force_start_ids"].squeeze(0).tolist()

        input_id_lists = []
        for user_ids in user_encodings["input_ids"]:
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
        generation_config: Optional[GenerationConfig] = None,
    ):
        """Batch prediction with metrics"""
        if not prompts:
            return []

        simple_batch_inputs = self.prepare_batch_inputs(prompts, task_type)
        input_lengths = [len(ids) for ids in simple_batch_inputs]

        if not hasattr(model, "generate_batch"):
            print("⚠️ Model doesn't support generate_batch(). Using sequential.")
            return self._sequential_fallback(
                prompts, task_type, model, tokenizer, ut_steps, generation_config
            )

        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                use_cuda_graph=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                max_batch_tokens=self.max_batch_size * self.max_new_tokens,
                repetition_penalty=1.2
            )

        start_time = time.perf_counter()

        try:
            batch_outputs = model.generate_batch(
                inputs=simple_batch_inputs,
                generation_config=generation_config,
            )
        except Exception as e:
            print(f"⚠️ generate_batch failed: {e}. Falling back.")
            return self._sequential_fallback(
                prompts, task_type, model, tokenizer, ut_steps, generation_config
            )

        batch_time = time.perf_counter() - start_time

        template = self.task_templates[task_type]
        results = [None] * len(prompts)
        request_ids = list(batch_outputs.keys())

        # Map outputs to prompts
        if all(isinstance(rid, int) for rid in request_ids):
            for request_id in request_ids:
                if 0 <= request_id < len(prompts):
                    output = batch_outputs[request_id]
                    results[request_id] = self._process_single_output(
                        output,
                        request_id,
                        input_lengths[request_id],
                        template,
                        tokenizer,
                        task_type,
                        ut_steps,
                        batch_time / len(prompts),
                    )
        else:
            # String/UUID request IDs
            input_to_index = {
                " ".join(map(str, inp)): idx
                for idx, inp in enumerate(simple_batch_inputs)
            }

            for request_id in request_ids:
                output = batch_outputs[request_id]

                if hasattr(output, "prompt_ids"):
                    input_key = " ".join(map(str, output.prompt_ids))
                    if input_key in input_to_index:
                        idx = input_to_index[input_key]
                        results[idx] = self._process_single_output(
                            output,
                            idx,
                            len(output.prompt_ids),
                            template,
                            tokenizer,
                            task_type,
                            ut_steps,
                            batch_time / len(prompts),
                        )

        # Fill missing results
        for i in range(len(prompts)):
            if results[i] is None:
                results[i] = {
                    "full_response": "ERROR: No output",
                    "prediction": "ERROR",
                    "generation_time": batch_time / len(prompts),
                    "generated_tokens": 0,
                    "input_tokens": input_lengths[i],
                    "ut_steps": ut_steps,
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