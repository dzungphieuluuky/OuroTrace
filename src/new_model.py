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
        if hasattr(model, "generation_config"):
            model.generation_config.cache_implementation = "static"
            model.generation_config.max_cache_length = max_seq_length
            print("   ✓ Static KV cache enabled")

    @staticmethod
    def optimize_attention_backend(model):
        """Enable Flash Attention / Memory-Efficient SDPA"""
        if torch.cuda.is_available() and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ):
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

        if hasattr(model, "generation_config"):
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
        if len(self.last_k_outputs) == self.k_repeat_abort and all(
            o == self.last_k_outputs[0] for o in self.last_k_outputs
        ):
            print(f"❌ Aborting due to repeated outputs...")
            raise ExperimentFailureException(
                f"Experiment failed: {self.k_repeat_abort} repeated outputs"
            )

    def initialize_quality_monitor(
        self,
        garbage_threshold: float = 0.3,
        example_similarity_threshold: float = 0.85,
        min_samples: int = 10,
        window_size: int = 20,
    ):
        """Initialize output quality monitoring"""
        self.quality_monitor = OutputQualityMonitor(
            garbage_threshold=garbage_threshold,
            example_similarity_threshold=example_similarity_threshold,
            min_samples_before_check=min_samples,
            window_size=window_size,
        )
        print(f"[+] Quality monitor initialized:")
        print(f"    → Garbage threshold: {garbage_threshold * 100:.0f}%")
        print(
            f"    → Example similarity threshold: {example_similarity_threshold * 100:.0f}%"
        )
        print(f"    → Min samples before check: {min_samples}")

    def load_model_with_ut_steps(self, total_ut_steps: int):
        """Load model with specific UT steps configuration and apply safe optimizations"""
        quantization_config = None
        if self.use_4bit_quant:
            print("→ Applying 4-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        # Auto-enable torch.compile only for ut_steps=1
        auto_compile = self.use_torch_compile

        print(f"\n{'=' * 60}")
        print(f"⚙️  LOADING MODEL CONFIGURATION")
        print(f"{'=' * 60}")
        print(f"Model Path: {self.model_path}")
        print(f"Requested UT Steps: {total_ut_steps}")
        print(f"Data Type: {self.dtype}")
        print(f"4-bit Quantization: {self.use_4bit_quant}")
        print(f"Torch Compile: {auto_compile}")

        # Load base config
        base_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        print(f"\n→ Base config loaded")
        print(f"   Original UT steps: {getattr(base_config, 'total_ut_steps', 'N/A')}")
        print(
            f"   Original early exit: {getattr(base_config, 'early_exit_threshold', 'N/A')}"
        )

        # Apply UT step configuration
        base_config.total_ut_steps = total_ut_steps
        print(f"\n→ Modified config:")
        print(f"   New UT steps: {base_config.total_ut_steps}")
        print(
            f"   Early exit threshold: {base_config.early_exit_threshold} (from default)"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, padding_side="left"
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
            attn_implementation="sdpa",
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

        print(f"\n{'─' * 60}")
        print(f"🚀 APPLYING SAFE OPTIMIZATIONS")
        print(f"{'─' * 60}")

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

        print(f"{'─' * 60}")

        # Final verification
        print(f"\n{'=' * 60}")
        print(f"✅ MODEL LOADED SUCCESSFULLY")
        print(f"{'=' * 60}")
        print(f"Device: {model.device}")
        print(f"Model dtype: {model.dtype}")
        print(f"VERIFIED UT steps: {model.config.total_ut_steps}")
        print(f"VERIFIED early exit: {model.config.early_exit_threshold}")

        if model.config.total_ut_steps != total_ut_steps:
            print(f"\n⚠️  WARNING: UT STEPS MISMATCH!")
            print(f"   Requested: {total_ut_steps}")
            print(f"   Actual: {model.config.total_ut_steps}")

        print(f"{'=' * 60}\n")

        return (
            model,
            tokenizer,
            base_config,
            {
                "total_ut_steps": total_ut_steps,
                "early_exit_threshold": base_config.early_exit_threshold,
            },
        )

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
        Pre-compute ALL tokenization components including user message wrapper.
        This eliminates ALL redundant tokenization during inference.
        """
        self.tokenizer = tokenizer

        task_configs = {
            "n_ary": {
                "system": (
                    "You are solving an ADDITION task.\n\n"
                    "TASK EXPLANATION:\n"
                    "You will receive several numbers to add together.\n"
                    "Your job: Add them one at a time, but do NOT show each step.\n\n"
                    "PROCESS:\n"
                    "1. Look at the input and COUNT the numbers (separated by +).\n"
                    "2. Write that count down (this is N).\n"
                    "3. Start with 0.\n"
                    "4. Add the FIRST number to 0.\n"
                    "5. Add the SECOND number to that result.\n"
                    "6. Add the THIRD number to that result.\n"
                    "7. Continue ONLY until you've used all N numbers.\n"
                    "8. As soon as all numbers are used, output [FINAL] with the sum.\n"
                    "9. STOP IMMEDIATELY – do not add any more numbers.\n\n"
                    "⚠️ CRITICAL: The input contains ONLY the numbers shown.\n"
                    "Do NOT invent additional numbers.\n"
                    "Do NOT continue patterns.\n"
                    "Do NOT add numbers that aren't explicitly in the input.\n\n"
                    "CONCRETE EXAMPLE 1 (DO NOT COPY):\n"
                    "User: 141 + 592 + 653 =\n\n"
                    "Assistant: [FINAL] 1386 [END]\n\n"
                    "CONCRETE EXAMPLE 2 (DO NOT COPY):\n"
                    "User: 589 + 793 =\n\n"
                    "Assistant: [FINAL] 1382 [END]\n\n"
                    "CONCRETE EXAMPLE 3 (DO NOT COPY):\n"
                    "User: 238 + 462 + 643 + 383 =\n\n"
                    "Assistant: [FINAL] 1726 [END]\n\n"
                    "OUTPUT FORMAT (YOU MUST FOLLOW THIS EXACTLY):\n"
                    "ASSISTANT: [FINAL] {final_sum} [END]\n\n"
                    "CRITICAL RULES:\n"
                    "✓ READ the input to find HOW MANY numbers to add.\n"
                    "✓ Do NOT show any intermediate steps; only the final answer.\n"
                    "✓ Each number from the input appears exactly once in the calculation.\n"
                    "✓ Start with 0 in Step 1.\n"
                    "✓ After all numbers are processed, write [FINAL] {answer} [END].\n"
                    "✓ STOP immediately after [END].\n\n"
                    "✗ DO NOT add numbers that aren't in the input.\n"
                    "✗ DO NOT invent additional numbers or patterns.\n"
                    "✗ DO NOT skip any numbers from the input.\n"
                    "✗ DO NOT add the same number twice.\n"
                    "✗ DO NOT create extra steps after all numbers are used.\n"
                    "✗ DO NOT split numbers into digits (treat 807 as one number, not 8+0+7).\n"
                    "✗ DO NOT continue generating after [END].\n"
                    "✗ DO NOT add code, explanations, or commentary.\n\n"
                    "PATTERN RECOGNITION:\n"
                    "Input has 2 numbers → only final answer shown.\n"
                    "Input has 3 numbers → only final answer shown.\n"
                    "Input has 4 numbers → only final answer shown.\n"
                    "Input has 5 numbers → only final answer shown.\n\n"
                    "⚠️ CRITICAL: After [FINAL] {answer} [END], STOP.\n"
                    "Do NOT generate: code, examples, explanations, or ANYTHING.\n"
                    "Your response ends at [END].\n\n"
                    "CORRECT OUTPUT:\n"
                    "Input: 179 + 366 =\n"
                    "[FINAL] 545 [END]\n\n"
                    "INCORRECT OUTPUT (DO NOT DO THIS):\n"
                    "[FINAL] 545 [END] ```python\n"
                    "This is WRONG. Stop at [END].\n\n"
                    "ALSO INCORRECT (DO NOT DO THIS):\n"
                    "[FINAL] 545 **Final Answer**\n"
                    "You must end with [END], not **Final Answer**."
                ),
                "force_start": "[FINAL]",
            },
            "p_hop": {
                "system": (
                    "You are solving a SEQUENCE FOLLOWING task.\n\n"
                    "TASK EXPLANATION:\n"
                    "You will receive a sequence of tokens, a starting token, and a number of hops.\n"
                    "Your job: Follow the sequence by hopping forward, but do NOT show each hop step.\n\n"
                    "PROCESS:\n"
                    "1. Find the START token in the sequence.\n"
                    "2. Look at the NEXT token (one position to the right).\n"
                    "3. Move to that token (this is hop 1).\n"
                    "4. Repeat until you've done ALL the required hops.\n"
                    "5. Output [FINAL] with the token you landed on.\n"
                    "6. STOP IMMEDIATELY.\n\n"
                    "CONCRETE EXAMPLE 1 (DO NOT COPY):\n"
                    "User: Sequence: A B C D E\n"
                    "       Start at: A\n"
                    "       Hops: 3\n\n"
                    "Assistant: [FINAL] D [END]\n\n"
                    "CONCRETE EXAMPLE 2 (DO NOT COPY):\n"
                    "User: Sequence: B C D A\n"
                    "       Start at: B\n"
                    "       Hops: 2\n\n"
                    "Assistant: [FINAL] D [END]\n\n"
                    "CONCRETE EXAMPLE 3 (DO NOT COPY):\n"
                    "User: Sequence: D C B A\n"
                    "       Start at: D\n"
                    "       Hops: 1\n\n"
                    "Assistant: [FINAL] C [END]\n\n"
                    "OUTPUT FORMAT (YOU MUST FOLLOW THIS EXACTLY):\n"
                    "[FINAL] {final_token} [END]\n\n"
                    "CRITICAL RULES:\n"
                    "✓ READ the input to find the starting token and hop count.\n"
                    "✓ Do NOT show any intermediate hop steps; only the final token.\n"
                    "✓ Each hop moves exactly ONE position forward in the sequence.\n"
                    "✓ Use ONLY tokens that appear in the input sequence.\n"
                    "✓ After the final hop, write [FINAL] {answer} [END].\n"
                    "✓ STOP immediately after [END].\n\n"
                    "✗ DO NOT skip hops.\n"
                    "✗ DO NOT invent new tokens (like E, F, G if they're not in the sequence).\n"
                    "✗ DO NOT continue generating after [END].\n"
                    "✗ DO NOT add explanations, code, or commentary.\n"
                    "✗ DO NOT output just [FINAL] without showing hops.\n\n"
                    "PATTERN RECOGNITION:\n"
                    "Asked for 2 hops → only final answer shown.\n"
                    "Asked for 3 hops → only final answer shown.\n"
                    "Asked for 5 hops → only final answer shown.\n\n"
                    "⚠️ CRITICAL: After [FINAL] {token} [END], STOP.\n"
                    "Do NOT generate: code, examples, explanations, or ANYTHING.\n"
                    "Your response ends at [END].\n\n"
                    "CORRECT OUTPUT:\n"
                    "Hop 1: At C → Next is A\n"
                    "Hop 2: At A → Next is D\n"
                    "[FINAL] D [END]\n\n"
                    "INCORRECT OUTPUT (DO NOT DO THIS):\n"
                    "[FINAL] D [END] ```python\n"
                    "This is WRONG. Stop at [END].\n\n"
                    "ALSO INCORRECT (DO NOT DO THIS):\n"
                    "[FINAL] D\n\n**Final\n"
                    "You must show the hop steps BEFORE [FINAL]."
                ),
                "force_start": "[FINAL]",
            },
            "igsm": {
                "system": (
                    "You are solving a MODULAR ARITHMETIC task (mod 7).\n\n"
                    "TASK EXPLANATION:\n"
                    "You will receive a series of variable assignments and a query for the value of one variable.\n"
                    "Your job: Evaluate each assignment step by step, but do NOT show each intermediate calculation.\n\n"
                    "PROCESS:\n"
                    "1. Read the first assignment.\n"
                    "2. Calculate the value.\n"
                    "3. Apply mod 7 to get a result between 0‑6.\n"
                    "4. Move to the next assignment.\n"
                    "5. Substitute any variables with their known values.\n"
                    "6. Calculate and apply mod 7.\n"
                    "7. Continue until the queried variable is found.\n"
                    "8. Output [FINAL] with that variable's value.\n"
                    "9. STOP IMMEDIATELY.\n\n"
                    "CONCRETE EXAMPLE 1 (DO NOT COPY):\n"
                    "User: E#E := 4. O#L := K#C. C#M := E#O. K#L := J#M * G#M. I#J := C#E + C#E. E#O := 3. P#L := G#M. N#F := G#M. N#E := E#O * K#C. N#M := N#E - C#E. K#C := 1. G#M := I#J. F#J := N#E. J#M := I#J. C#E := E#E + K#C. L#M := 4. G#K := J#M. K#L? \n\n"
                    "Assistant: [FINAL] 2 [END]\n\n"
                    "CONCRETE EXAMPLE 2 (DO NOT COPY):\n"
                    "User: M#J := F#J. A#K := J#P - J#P. B#P := I#O. J#P := F#B. F#J := 5. C#P := 6. A#P := 0. F#B := M#J. E#M := A#P. I#O := M#J. M#E := M#J. K#I := I#O. M#A := M#J + M#J. L#I := I#O. A#I := 0. O#G := K#I * K#I. A#O := L#I * K#I. J#A := B#P. A#K? \n\n"
                    "Assistant: [FINAL] 0 [END]\n\n"
                    "CONCRETE EXAMPLE 3 (DO NOT COPY):\n"
                    "User: M#M := K#A. L#D := C#O. K#P := 4. P#D := B#K. D#H := G#B. J#A := 0. B#K := K#A. G#B := 1. P#E := 2. E#P := P#D. C#O := M#M + L#B. L#G := D#H. L#B := D#H - D#H. G#A := P#D. L#A := C#O. K#A := P#E. L#A? \n\n"
                    "Assistant: [FINAL] 2 [END]\n\n"
                    "OUTPUT FORMAT (YOU MUST FOLLOW THIS EXACTLY):\n"
                    "[FINAL] {answer} [END]\n\n"
                    "CRITICAL RULES:\n"
                    "✓ Process EVERY assignment in order.\n"
                    "✓ Do NOT show any intermediate arithmetic calculations; only the final answer.\n"
                    "✓ Substitute variable values immediately when they appear.\n"
                    "✓ The final answer must be between 0 and 6 (inclusive).\n"
                    "✓ After finding the query variable, write [FINAL] {answer} [END].\n"
                    "✓ STOP immediately after [END].\n\n"
                    "✗ DO NOT skip any assignments.\n"
                    "✗ DO NOT forget to apply mod 7.\n"
                    "✗ DO NOT output results outside the range 0‑6.\n"
                    "✗ DO NOT continue after finding the queried variable.\n"
                    "✗ DO NOT add code, explanations, or commentary.\n\n"
                    "OPERATION TYPES:\n"
                    "Direct assignment: A := 5\n"
                    "  → A = 5 (mod 7) = 5\n\n"
                    "Variable copy: B := A (where A = 5)\n"
                    "  → B = 5 (mod 7) = 5\n\n"
                    "Addition: C := A + B (where A = 5, B = 4)\n"
                    "  → C = 5 + 4 = 9 (mod 7) = 2\n\n"
                    "⚠️ CRITICAL: After [FINAL] {answer} [END], STOP.\n"
                    "Do NOT generate: code, examples, explanations, or ANYTHING.\n"
                    "Your response ends at [END].\n\n"
                    "CORRECT OUTPUT:\n"
                    "[FINAL] 1 [END]\n\n"
                    "INCORRECT OUTPUT (DO NOT DO THIS):\n"
                    "[FINAL] 1 [END] ```python\n"
                    "This is WRONG. Stop at [END].\n\n"
                    "ALSO INCORRECT (DO NOT DO THIS):\n"
                    "[FINAL] 8 [END]\n"
                    "8 is NOT valid. Must apply mod 7 to get a result between 0‑6."
                ),
                "force_start": "[FINAL]",
            },
        }
        # Pre-compute generation configs (move outside predict loop)
        task_generation_configs = {
            "n_ary": GenerationConfig(
                max_new_tokens=16,
                min_new_tokens=4,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                temperature=0.1,
            ),
            "p_hop": GenerationConfig(
                max_new_tokens=16,
                min_new_tokens=4,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                temperature=0.1,
            ),
            "igsm": GenerationConfig(
                max_new_tokens=16,
                min_new_tokens=4,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                temperature=0.1,
            ),
        }

        # Build templates with pre-tokenized components
        self.task_templates = {}

        for task_type, config in task_configs.items():
            # Step 1: Pre-tokenize system prompt
            system_messages = [{"role": "system", "content": config["system"]}]
            system_text = tokenizer.apply_chat_template(
                system_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            system_tokens = tokenizer.encode(system_text, add_special_tokens=False)

            # Step 2: Pre-tokenize force_start
            force_start_tokens = tokenizer.encode(
                config["force_start"],
                add_special_tokens=False,
            )

            # Step 3: Pre-compute user message wrapper tokens
            # Get the template structure WITHOUT actual content
            dummy_user_msg = [{"role": "user", "content": "PLACEHOLDER"}]
            user_template_text = tokenizer.apply_chat_template(
                dummy_user_msg,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Split into prefix and suffix around PLACEHOLDER
            # This allows us to only tokenize the actual user content
            if "PLACEHOLDER" in user_template_text:
                prefix, suffix = user_template_text.split("PLACEHOLDER", 1)
                user_prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                user_suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
            else:
                # Fallback: tokenize whole template
                user_prefix_tokens = []
                user_suffix_tokens = tokenizer.encode(
                    user_template_text, add_special_tokens=False
                )

            self.task_templates[task_type] = {
                # Original strings (for debugging)
                "system": config["system"],
                "force_start_text": config["force_start"],
                # Pre-tokenized components
                "system_tokens": system_tokens,
                "user_prefix_tokens": user_prefix_tokens,  # NEW: "<|im_start|>user\n"
                "user_suffix_tokens": user_suffix_tokens,  # NEW: "\n<|im_end|>\n<|im_start|>assistant\n"
                "force_start_tokens": force_start_tokens,
                # Pre-computed generation config
                "generation_config": task_generation_configs[task_type],
                # Stop sequences
                "stop_sequences": [
                    "[END]",
                    "[FINAL]",
                    "SAMPLE",
                    "\n```",
                    "\n\n",
                    "```python",
                    "def ",
                    "#",
                    "**Final",
                    "Example usage",
                ],
            }

        print("[+] Task templates with pre-tokenized components computed.")
        print(
            f"    System prompt N_ary tokens: {len(self.task_templates['n_ary']['system_tokens'])} tokens"
        )
        print(
            f"    System prompt P_hop tokens: {len(self.task_templates['p_hop']['system_tokens'])} tokens"
        )
        print(
            f"    System prompt IGSM tokens: {len(self.task_templates['igsm']['system_tokens'])} tokens"
        )

        print(
            f"    User prefix tokens: {len(self.task_templates['n_ary']['user_prefix_tokens'])} tokens"
        )
        print(
            f"    User suffix tokens: {len(self.task_templates['n_ary']['user_suffix_tokens'])} tokens"
        )
        print(
            f"    Force start tokens: {len(self.task_templates['n_ary']['force_start_tokens'])} tokens"
        )

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

        if not hasattr(model.config, "total_ut_steps"):
            print("❌ ERROR: Model missing total_ut_steps config!")
            error_results = [
                self._create_error_result(inp, ut_steps) for inp in user_inputs
            ]
            return error_results[0] if is_single else error_results

        if not hasattr(self, "task_templates") or task_type not in self.task_templates:
            self._build_task_templates(tokenizer)

        template = self.task_templates[task_type]
        device = model.device

        # Pre-computed constant components (no computation here!)
        system_tokens = torch.tensor(template["system_tokens"], dtype=torch.long)
        user_prefix_tokens = torch.tensor(
            template["user_prefix_tokens"], dtype=torch.long
        )
        user_suffix_tokens = torch.tensor(
            template["user_suffix_tokens"], dtype=torch.long
        )
        force_start_tokens = torch.tensor(
            template["force_start_tokens"], dtype=torch.long
        )

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
            full_sequence = torch.cat(
                [
                    system_tokens,
                    user_prefix_tokens,
                    content_tensor,  # ONLY this varies per input
                    user_suffix_tokens,
                    force_start_tokens,
                ]
            )
            concatenated_input_ids.append(full_sequence)

        # Efficient padding with pre-allocation
        max_len = max(seq.size(0) for seq in concatenated_input_ids)
        batch_size = len(concatenated_input_ids)

        # Pre-allocate tensors (faster than appending to list)
        input_ids_padded = torch.full(
            (batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long
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
            gen_config = GenerationConfig(
                **{**gen_config.to_dict(), **generation_config}
            )

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
            error_results = [
                self._create_error_result(inp, ut_steps, str(e)) for inp in user_inputs
            ]
            return error_results[0] if is_single else error_results

        total_generation_time = time.perf_counter() - start_time

        # Process results (this part is still per-sample)
        results = []
        for i in range(len(user_inputs)):
            generated_ids = outputs.sequences[i, input_ids.shape[1] :]

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
                    r"\[FINAL\]\s*([A-D])\b",
                    r"Final:\s*([A-D])\b",
                    r"\b([A-D])\s*$",
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
                    r"\[FINAL\]\s*(-?\d+)",
                    r"Final:\s*(-?\d+)",
                    r"=\s*(-?\d+)\s*$",
                    r"\b(-?\d+)\s*$",
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, full_response)
                    if matches:
                        pred = matches[-1]
                        break
                else:
                    # Fallback: get last number from last non-empty line
                    lines = [l.strip() for l in full_response.split("\n") if l.strip()]
                    if lines:
                        last_line = lines[-1]
                        numbers = re.findall(r"-?\d+", last_line)
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

        if text.count("\n\n\n") > 3:
            return True

        bracket_ratio = (text.count("[") + text.count("]")) / max(len(text), 1)
        if bracket_ratio > 0.3:
            return True

        if len(text) > 100:
            unique_chars = len(set(text))
            if unique_chars < 10:
                return True

        whitespace_ratio = (text.count(" ") + text.count("\n")) / max(len(text), 1)
        if whitespace_ratio > 0.7:
            return True

        if len(text) > 50:
            for char in ["[", "]", "\n", " ", "."]:
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

    def _create_error_result(
        self, user_input: str, ut_steps: int, error_msg: str = "Model config error"
    ) -> Dict[str, Any]:
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
            print("\n" + "=" * 60)
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
            print("=" * 60 + "\n")
            raise ExperimentFailureException(f"Experiment failed: {failure.reason}")
