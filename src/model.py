import torch
import time
import re
from typing import List, Optional
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

class OuroThinkingExperiment:
    def __init__(self, model_path: str, dtype=torch.float16, use_4bit_quant: bool = False, use_torch_compile: bool = False):
        torch.cuda.empty_cache()
        self.model_path = model_path
        self.dtype = dtype
        self.use_4bit_quant = use_4bit_quant
        self.use_torch_compile = use_torch_compile
        self.tokenizer = None
        self.task_templates = {}
    
    def load_model_with_ut_steps(self, total_ut_steps: int, early_exit_threshold: float):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if self.use_4bit_quant else None
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        config.total_ut_steps = total_ut_steps
        config.early_exit_threshold = early_exit_threshold
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, padding_side="left")
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, config=config, device_map="cuda", attn_implementation="sdpa_paged",
            dtype=self.dtype if not self.use_4bit_quant else None, trust_remote_code=True, quantization_config=quantization_config
        )
        if self.use_torch_compile: model = torch.compile(model)
        model.eval()
        return model, tokenizer, None, {"total_ut_steps": total_ut_steps, "early_exit_threshold": early_exit_threshold}
    
    def _build_task_templates(self, tokenizer):
        self.tokenizer = tokenizer
        task_configs = {
            "n_ary": {
                "system": "You are a mechanical calculation engine. Output ONLY the accumulation steps. Do not use lists or english explanations.",
                "few_shots": [
                    {"role": "user", "content": "100 + 200 + 300 =", "role_response": "Current: 0\nAdd 100: 0 + 100 = 100\nCurrent: 100\nAdd 200: 100 + 200 = 300\nCurrent: 300\nAdd 300: 300 + 300 = 600\nFinal: 600"},
                    {"role": "user", "content": "050 + 025 =", "role_response": "Current: 0\nAdd 050: 0 + 50 = 50\nCurrent: 50\nAdd 025: 50 + 25 = 75\nFinal: 75"},
                    {"role": "user", "content": "010 + 010 + 010 =", "role_response": "Current: 0\nAdd 010: 0 + 10 = 10\nCurrent: 10\nAdd 010: 10 + 10 = 20\nCurrent: 20\nAdd 010: 20 + 10 = 30\nFinal: 30"}
                ],
                "input_prefix": "", "force_start": "Current: 0\n"
            },
            "p_hop": {
                "system": "You are an induction head mechanism. Trace the sequence jumps step-by-step. Return the Final token.",
                "few_shots": [
                    {"role": "user", "content": "Sequence: A B C D A B. Start: A. Hop 1 times.", "role_response": "Start at A. Found 'A' at index 0. Next token is B. Final: B"},
                    {"role": "user", "content": "Sequence: D C B A D C. Start: D. Hop 2 times.", "role_response": "Start at D. Found 'D' at index 0. Next token is C. Found 'C' at index 1. Next token is B. Final: B"},
                    {"role": "user", "content": "Sequence: A A B B. Start: A. Hop 1 times.", "role_response": "Start at A. Found 'A' at index 0. Next token is A. Final: A"}
                ],
                "input_prefix": "", "force_start": "Start at"
            },
            "igsm": {
                "system": "You are a symbolic math engine. Solve the DAG equations modulo 7. Output strictly in the format: 'Eq. ==> Result.'",
                "few_shots": [
                    {"role": "user", "content": "Question. A#A := 4. A#B := A#A + 2. A#B?", "role_response": "Answer with CoT. A#A = 4. ==> A#A = 4. A#B = A#A + 2. ==> A#B = 6. Final: 6"},
                    {"role": "user", "content": "Question. X#Y := 3. Z#Z := X#Y * 2. Z#Z?", "role_response": "Answer with CoT. X#Y = 3. ==> X#Y = 3. Z#Z = X#Y * 2. ==> Z#Z = 6. Final: 6"},
                    {"role": "user", "content": "Question. B#K := 1. L#L := B#K - 5. L#L?", "role_response": "Answer with CoT. B#K = 1. ==> B#K = 1. L#L = B#K - 5. ==> L#L = 3. Final: 3"}
                ],
                "input_prefix": "", "force_start": "Answer with CoT."
            }
        }
        
        for task_type, config in task_configs.items():
            messages = [{"role": "system", "content": config["system"]}]
            for shot in config["few_shots"]:
                messages.append({"role": "user", "content": shot["content"]})
                messages.append({"role": "assistant", "content": shot["role_response"]})
            
            static_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            static_inputs = tokenizer(static_prompt, return_tensors="pt")
            force_start_tokens = tokenizer(config["force_start"], return_tensors="pt", add_special_tokens=False)
            
            self.task_templates[task_type] = {
                "static_input_ids": static_inputs.input_ids,
                "static_attention_mask": static_inputs.attention_mask,
                "force_start_ids": force_start_tokens.input_ids,
                "input_prefix": config["input_prefix"],
                "force_start_text": config["force_start"]
            }
    
    def _extract_final_answer(self, full_response: str, task_type: str) -> str:
        pred = "0"
        try:
            if task_type == "p_hop":
                patterns = [r"Final\s*:\s*(\w+)", r"Next token is\s*(\w+)", r"Answer\s*:\s*(\w+)"]
                for p in patterns:
                    if m := re.search(p, full_response, re.IGNORECASE): pred = m.group(1).strip(); break
                else: pred = "Error"
            else:
                patterns = [r"Final\s*:\s*([-+]?\d*\.?\d+)", r"Answer\s*:\s*([-+]?\d*\.?\d+)", r"=\s*([-+]?\d*\.?\d+)$"]
                all_matches = []
                for p in patterns: all_matches.extend(re.findall(p, full_response, re.IGNORECASE))
                if all_matches: pred = all_matches[-1]
        except: pred = "ParseError"
        return pred
    
    @torch.no_grad()
    def predict_with_metrics_optimized(self, user_input, task_type, model, tokenizer, ut_steps, generation_config=None):
        if not hasattr(self, 'task_templates') or task_type not in self.task_templates: self._build_task_templates(tokenizer)
        template = self.task_templates[task_type]
        
        user_query = template["input_prefix"] + user_input
        user_tokens = tokenizer(user_query, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        input_ids = torch.cat([template["static_input_ids"].to(model.device), user_tokens, template["force_start_ids"].to(model.device)], dim=1)
        
        start_time = time.perf_counter()
        gen_config = generation_config or {'max_new_tokens': 1024, 'do_sample': False, 'repetition_penalty': 1.2, 'min_length': 5}
        outputs = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), pad_token_id=tokenizer.eos_token_id, use_cache=True, return_dict_in_generate=True, output_scores=False, **gen_config)
        
        gen_tokens = outputs.sequences[0, input_ids.shape[1]:]
        full_response = template["force_start_text"] + tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return {
            'full_response': full_response, 'prediction': self._extract_final_answer(full_response, task_type),
            'generation_time': time.perf_counter() - start_time, 'generated_tokens': len(gen_tokens), 'input_tokens': input_ids.shape[1], 'ut_steps': ut_steps
        }

    @torch.no_grad()
    def calculate_perplexity(self, model, tokenizer, text_data, ut_steps, max_length=2048, stride=512):
        if not text_data: return float('nan'), float('nan')
        encodings = tokenizer(text_data[0], return_tensors='pt', max_length=max_length * 2, truncation=True)
        input_ids = encodings.input_ids.to(model.device)
        if input_ids.size(1) < 2: return float('nan'), float('nan')
        
        nlls = []
        for i in range(0, input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i 
            input_slice = input_ids[:, begin_loc:end_loc]
            target_slice = input_slice.clone()
            target_slice[:, :-trg_len] = -100
            
            if (target_slice != -100).sum() == 0: continue
            outputs = model(input_ids=input_slice, labels=target_slice)
            nlls.append(outputs.loss * trg_len)
            
        return torch.exp(torch.stack(nlls).sum() / end_loc).item(), torch.stack(nlls).sum().item() / end_loc

class OuroBatchExperiment(OuroThinkingExperiment):
    def __init__(self, model_path, dtype=torch.float16, use_4bit_quant=False, use_torch_compile=False, max_batch_size=4, max_new_tokens=1024):
        super().__init__(model_path, dtype, use_4bit_quant, use_torch_compile)
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens

    def batch_predict_with_metrics(self, prompts, task_type, model, tokenizer, ut_steps, generation_config=None):
        if not hasattr(self, 'task_templates') or task_type not in self.task_templates: self._build_task_templates(tokenizer)
        template = self.task_templates[task_type]
        
        batch_inputs = []
        static_ids = template["static_input_ids"].squeeze(0).tolist()
        force_ids = template["force_start_ids"].squeeze(0).tolist()
        
        for p in prompts:
            user_ids = tokenizer(template["input_prefix"] + p, add_special_tokens=False).input_ids
            batch_inputs.append(static_ids + user_ids + force_ids)
            
        if not hasattr(model, 'generate_batch'): return [self.predict_with_metrics_optimized(p, task_type, model, tokenizer, ut_steps, generation_config) for p in prompts]
        
        gen_config = generation_config or GenerationConfig(max_new_tokens=self.max_new_tokens, do_sample=False, repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        
        start = time.perf_counter()
        try:
            outputs = model.generate_batch(inputs=batch_inputs, generation_config=gen_config)
        except:
            return [self.predict_with_metrics_optimized(p, task_type, model, tokenizer, ut_steps, generation_config) for p in prompts]
        
        duration = (time.perf_counter() - start) / len(prompts)
        results = []
        
        # Handle string request_ids if necessary (simplified for brevity)
        req_ids = sorted(list(outputs.keys())) if isinstance(list(outputs.keys())[0], int) else list(range(len(prompts)))
        
        for i, req_id in enumerate(req_ids):
            # Fallback logic for keys needed here if using real batching with UUIDs
            output_obj = outputs[req_id] if req_id in outputs else outputs[list(outputs.keys())[i]]
            
            gen_ids = output_obj.sequences[0] if hasattr(output_obj, 'sequences') else []
            full_resp = template["force_start_text"] + tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append({
                'full_response': full_resp,
                'prediction': self._extract_final_answer(full_resp, task_type),
                'generation_time': duration,
                'generated_tokens': len(gen_ids),
                'input_tokens': len(batch_inputs[i]),
                'ut_steps': ut_steps
            })
        return results