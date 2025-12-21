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
        Pre-compute prompt templates that enforce exact output format.
        Forces model to show ALL steps before [FINAL] and stop immediately.
        """
        self.tokenizer = tokenizer

        task_configs = {
            "n_ary": {
                "system": (
                    "You are solving an ADDITION task.\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "TASK EXPLANATION:\n"
                    "═══════════════════════════════════════════\n"
                    "You will receive several numbers to add together.\n"
                    "Your job: Add them one at a time, showing each step.\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "STEP-BY-STEP PROCESS:\n"
                    "═══════════════════════════════════════════\n"
                    "1. Look at the input and COUNT the numbers (separated by +)\n"
                    "2. Write that count down (this is N)\n"
                    "3. Start with 0\n"
                    "4. Add the FIRST number to 0\n"
                    "5. Add the SECOND number to that result\n"
                    "6. Add the THIRD number to that result\n"
                    "7. Continue ONLY until you've used all N numbers\n"
                    "8. As soon as all numbers are used, output [FINAL]\n"
                    "9. STOP IMMEDIATELY - do not add any more numbers\n\n"
                    
                    "⚠️ CRITICAL: The input contains ONLY the numbers shown.\n"
                    "Do NOT invent additional numbers.\n"
                    "Do NOT continue patterns.\n"
                    "Do NOT add numbers that aren't explicitly in the input.\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CONCRETE EXAMPLE 1:\n"
                    "═══════════════════════════════════════════\n"
                    "Input: Add these 3 numbers: 100 + 200 + 300\n\n"
                    
                    "Solution:\n"
                    "Step 1: 0 + 100 = 100\n"
                    "Step 2: 100 + 200 = 300\n"
                    "Step 3: 300 + 300 = 600\n"
                    "[FINAL] 600 [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CONCRETE EXAMPLE 2:\n"
                    "═══════════════════════════════════════════\n"
                    "Input: Add these 2 numbers: 807 + 696\n\n"
                    
                    "Solution:\n"
                    "Step 1: 0 + 807 = 807\n"
                    "Step 2: 807 + 696 = 1503\n"
                    "[FINAL] 1503 [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CONCRETE EXAMPLE 3:\n"
                    "═══════════════════════════════════════════\n"
                    "Input: Add these 4 numbers: 50 + 25 + 10 + 5\n\n"
                    
                    "Solution:\n"
                    "Step 1: 0 + 50 = 50\n"
                    "Step 2: 50 + 25 = 75\n"
                    "Step 3: 75 + 10 = 85\n"
                    "Step 4: 85 + 5 = 90\n"
                    "[FINAL] 90 [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "OUTPUT FORMAT (YOU MUST FOLLOW THIS EXACTLY):\n"
                    "═══════════════════════════════════════════\n"
                    "Step 1: 0 + {first_number} = {result_1}\n"
                    "Step 2: {result_1} + {second_number} = {result_2}\n"
                    "Step 3: {result_2} + {third_number} = {result_3}\n"
                    "... (continue for ALL numbers)\n"
                    "Step N: {result_N-1} + {last_number} = {final_sum}\n"
                    "[FINAL] {final_sum} [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CRITICAL RULES:\n"
                    "═══════════════════════════════════════════\n"
                    "✓ READ the input to find HOW MANY numbers to add\n"
                    "✓ The input will say \"Add these N numbers:\" - that's your count\n"
                    "✓ Show EXACTLY N steps (no more, no less)\n"
                    "✓ Each number from the input appears in EXACTLY ONE step\n"
                    "✓ Always start with 0 in Step 1\n"
                    "✓ Each step adds ONE new number to the running total\n"
                    "✓ After all steps, write [FINAL] {answer} [END]\n"
                    "✓ STOP immediately after [END]\n\n"
                    
                    "✗ DO NOT add numbers that aren't in the input\n"
                    "✗ DO NOT invent additional numbers or patterns\n"
                    "✗ DO NOT skip any numbers from the input\n"
                    "✗ DO NOT add the same number twice\n"
                    "✗ DO NOT create extra steps after all numbers are used\n"
                    "✗ DO NOT split numbers into digits (treat 807 as one number, not 8+0+7)\n"
                    "✗ DO NOT continue generating after [END]\n"
                    "✗ DO NOT add code, explanations, or commentary\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "PATTERN RECOGNITION:\n"
                    "═══════════════════════════════════════════\n"
                    "Input has 2 numbers → Show 2 steps + [FINAL]\n"
                    "Input has 3 numbers → Show 3 steps + [FINAL]\n"
                    "Input has 4 numbers → Show 4 steps + [FINAL]\n"
                    "Input has 5 numbers → Show 5 steps + [FINAL]\n\n"
                    
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "⚠️ CRITICAL: After [FINAL] {answer} [END], STOP.\n"
                    "Do NOT generate: code, examples, explanations, or ANYTHING.\n"
                    "Your response ends at [END].\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    
                    "CORRECT OUTPUT:\n"
                    "Input: Add these 2 numbers: 179 + 366\n"
                    "Step 1: 0 + 179 = 179\n"
                    "Step 2: 179 + 366 = 545\n"
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
                    
                    "═══════════════════════════════════════════\n"
                    "TASK EXPLANATION:\n"
                    "═══════════════════════════════════════════\n"
                    "You will receive:\n"
                    "  • A sequence of tokens (letters like A, B, C, D)\n"
                    "  • A starting position (which token to begin at)\n"
                    "  • A number of hops to perform\n\n"
                    
                    "Your job: Follow the sequence by hopping forward, "
                    "one position at a time.\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "STEP-BY-STEP PROCESS:\n"
                    "═══════════════════════════════════════════\n"
                    "1. Find the START token in the sequence\n"
                    "2. Look at the NEXT token (one position to the right)\n"
                    "3. Move to that token (this is hop 1)\n"
                    "4. Repeat: look at the next token, move to it (hop 2)\n"
                    "5. Continue until you've done ALL the required hops\n"
                    "6. Output [FINAL] with the token you landed on\n"
                    "7. STOP IMMEDIATELY\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CONCRETE EXAMPLE 1:\n"
                    "═══════════════════════════════════════════\n"
                    "Input: Sequence: A B C D E\n"
                    "       Start at: A\n"
                    "       Hops: 3\n\n"
                    
                    "Solution:\n"
                    "Hop 1: At A → Next is B\n"
                    "Hop 2: At B → Next is C\n"
                    "Hop 3: At C → Next is D\n"
                    "[FINAL] D [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CONCRETE EXAMPLE 2:\n"
                    "═══════════════════════════════════════════\n"
                    "Input: Sequence: C A D B A C\n"
                    "       Start at: C (position 0)\n"
                    "       Hops: 4\n\n"
                    
                    "Solution:\n"
                    "Hop 1: At C → Next is A\n"
                    "Hop 2: At A → Next is D\n"
                    "Hop 3: At D → Next is B\n"
                    "Hop 4: At B → Next is A\n"
                    "[FINAL] A [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CONCRETE EXAMPLE 3:\n"
                    "═══════════════════════════════════════════\n"
                    "Input: Sequence: D D A B C\n"
                    "       Start at: D (position 0)\n"
                    "       Hops: 2\n\n"
                    
                    "Solution:\n"
                    "Hop 1: At D → Next is D\n"
                    "Hop 2: At D → Next is A\n"
                    "[FINAL] A [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "OUTPUT FORMAT (YOU MUST FOLLOW THIS EXACTLY):\n"
                    "═══════════════════════════════════════════\n"
                    "Hop 1: At {current_token} → Next is {next_token}\n"
                    "Hop 2: At {current_token} → Next is {next_token}\n"
                    "Hop 3: At {current_token} → Next is {next_token}\n"
                    "... (continue for ALL hops)\n"
                    "Hop N: At {current_token} → Next is {final_token}\n"
                    "[FINAL] {final_token} [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CRITICAL RULES:\n"
                    "═══════════════════════════════════════════\n"
                    "✓ Show EVERY SINGLE hop (if asked for 5 hops, show 5 lines)\n"
                    "✓ Each hop moves exactly ONE position forward in the sequence\n"
                    "✓ Use ONLY tokens that appear in the input sequence\n"
                    "✓ After the final hop, write [FINAL] {answer} [END]\n"
                    "✓ STOP immediately after [END]\n\n"
                    
                    "✗ DO NOT skip hops\n"
                    "✗ DO NOT invent new tokens (like E, F, G if they're not in the sequence)\n"
                    "✗ DO NOT continue generating after [END]\n"
                    "✗ DO NOT add explanations, code, or commentary\n"
                    "✗ DO NOT output just [FINAL] without showing the hops\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "PATTERN RECOGNITION:\n"
                    "═══════════════════════════════════════════\n"
                    "Asked for 2 hops → Show 2 hop lines + [FINAL]\n"
                    "Asked for 3 hops → Show 3 hop lines + [FINAL]\n"
                    "Asked for 5 hops → Show 5 hop lines + [FINAL]\n\n"
                    
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "⚠️ CRITICAL: After [FINAL] {token} [END], STOP.\n"
                    "Do NOT generate: code, examples, explanations, or ANYTHING.\n"
                    "Your response ends at [END].\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    
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
                    
                    "═══════════════════════════════════════════\n"
                    "TASK EXPLANATION:\n"
                    "═══════════════════════════════════════════\n"
                    "You will receive:\n"
                    "  • A series of variable assignments (like A := 5, B := A + 3)\n"
                    "  • A query asking for the value of one variable\n\n"
                    
                    "Your job: Evaluate each assignment step by step, "
                    "applying modulo 7 to each result.\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "MODULO 7 QUICK REFERENCE:\n"
                    "═══════════════════════════════════════════\n"
                    "Modulo 7 means: divide by 7 and take the REMAINDER.\n"
                    "Valid results: 0, 1, 2, 3, 4, 5, 6 (never 7 or higher)\n\n"
                    
                    "Examples:\n"
                    "  5 mod 7 = 5 (5 ÷ 7 = 0 remainder 5)\n"
                    "  8 mod 7 = 1 (8 ÷ 7 = 1 remainder 1)\n"
                    "  14 mod 7 = 0 (14 ÷ 7 = 2 remainder 0)\n"
                    "  20 mod 7 = 6 (20 ÷ 7 = 2 remainder 6)\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "STEP-BY-STEP PROCESS:\n"
                    "═══════════════════════════════════════════\n"
                    "1. Read the first assignment\n"
                    "2. Calculate the value\n"
                    "3. Apply mod 7 to get a result between 0-6\n"
                    "4. Move to the next assignment\n"
                    "5. Substitute any variables with their known values\n"
                    "6. Calculate and apply mod 7\n"
                    "7. Continue until the queried variable is found\n"
                    "8. Output [FINAL] with that variable's value\n"
                    "9. STOP IMMEDIATELY\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CONCRETE EXAMPLE 1:\n"
                    "═══════════════════════════════════════════\n"
                    "Input: A := 5\n"
                    "       B := A + 3\n"
                    "       Query: B = ?\n\n"
                    
                    "Solution:\n"
                    "Step 1: A = 5 (mod 7) = 5\n"
                    "Step 2: B = A + 3 = 5 + 3 = 8 (mod 7) = 1\n"
                    "[FINAL] 1 [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CONCRETE EXAMPLE 2:\n"
                    "═══════════════════════════════════════════\n"
                    "Input: X := 10\n"
                    "       Y := 4\n"
                    "       Z := X + Y\n"
                    "       Query: Z = ?\n\n"
                    
                    "Solution:\n"
                    "Step 1: X = 10 (mod 7) = 3\n"
                    "Step 2: Y = 4 (mod 7) = 4\n"
                    "Step 3: Z = X + Y = 3 + 4 = 7 (mod 7) = 0\n"
                    "[FINAL] 0 [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CONCRETE EXAMPLE 3:\n"
                    "═══════════════════════════════════════════\n"
                    "Input: P := 8\n"
                    "       Q := P\n"
                    "       R := Q + P\n"
                    "       Query: R = ?\n\n"
                    
                    "Solution:\n"
                    "Step 1: P = 8 (mod 7) = 1\n"
                    "Step 2: Q = P = 1 (mod 7) = 1\n"
                    "Step 3: R = Q + P = 1 + 1 = 2 (mod 7) = 2\n"
                    "[FINAL] 2 [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "OUTPUT FORMAT (YOU MUST FOLLOW THIS EXACTLY):\n"
                    "═══════════════════════════════════════════\n"
                    "Step 1: {var} = {value} (mod 7) = {result}\n"
                    "Step 2: {var} = {expression} = {computed} (mod 7) = {result}\n"
                    "Step 3: {var} = {expression} = {computed} (mod 7) = {result}\n"
                    "... (continue for ALL assignments)\n"
                    "Step N: {query_var} = {value} (mod 7) = {answer}\n"
                    "[FINAL] {answer} [END]\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "CRITICAL RULES:\n"
                    "═══════════════════════════════════════════\n"
                    "✓ Process EVERY assignment in order\n"
                    "✓ Substitute variable values immediately when they appear\n"
                    "✓ Show the arithmetic calculation BEFORE applying mod 7\n"
                    "✓ Final answer must be between 0 and 6 (inclusive)\n"
                    "✓ After finding the query variable, write [FINAL] {answer} [END]\n"
                    "✓ STOP immediately after [END]\n\n"
                    
                    "✗ DO NOT skip any assignments\n"
                    "✗ DO NOT forget to apply mod 7\n"
                    "✗ DO NOT output results outside the range 0-6\n"
                    "✗ DO NOT continue after finding the queried variable\n"
                    "✗ DO NOT add code, explanations, or commentary\n\n"
                    
                    "═══════════════════════════════════════════\n"
                    "OPERATION TYPES:\n"
                    "═══════════════════════════════════════════\n"
                    "Direct assignment: A := 5\n"
                    "  → A = 5 (mod 7) = 5\n\n"
                    
                    "Variable copy: B := A (where A = 5)\n"
                    "  → B = 5 (mod 7) = 5\n\n"
                    
                    "Addition: C := A + B (where A = 5, B = 4)\n"
                    "  → C = 5 + 4 = 9 (mod 7) = 2\n\n"
                    
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "⚠️ CRITICAL: After [FINAL] {answer} [END], STOP.\n"
                    "Do NOT generate: code, examples, explanations, or ANYTHING.\n"
                    "Your response ends at [END].\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    
                    "CORRECT OUTPUT:\n"
                    "Step 1: A = 5 (mod 7) = 5\n"
                    "Step 2: B = A + 3 = 5 + 3 = 8 (mod 7) = 1\n"
                    "[FINAL] 1 [END]\n\n"
                    
                    "INCORRECT OUTPUT (DO NOT DO THIS):\n"
                    "[FINAL] 1 [END] ```python\n"
                    "This is WRONG. Stop at [END].\n\n"
                    
                    "ALSO INCORRECT (DO NOT DO THIS):\n"
                    "[FINAL] 8 [END]\n"
                    "8 is NOT valid. Must apply mod 7 to get a result between 0-6."
                ),
                "force_start": "[FINAL]",
            }
        }

        self.task_templates = {}
        for task_type, config in task_configs.items():
            self.task_templates[task_type] = {
                "system": config["system"],
                "force_start_text": config["force_start"],
                "stop_sequences": ["[END]", "SAMPLE", "\n```", "```python", "def ", "#", "**Final", "Example usage"],
            }

        print("[+] Task templates with strict format enforcement pre-computed.")
        
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
        Correctly counts generated tokens for each sample in batch.
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
            messages = [
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": user_input},
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt += template["force_start_text"]
            prompts.append(prompt)

        # Tokenize with padding
        encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        input_lengths = attention_mask.sum(dim=1)  # Store original lengths

        default_config = self._get_optimal_generation_config(task_type)
        if generation_config:
            default_config.update(generation_config)

        # Start timing for each batch item separately
        start_time = time.perf_counter()
        default_config["stop_strings"] = template["stop_sequences"]
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
                generation_config=GenerationConfig(**default_config),
            )
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            error_results = [self._create_error_result(inp, ut_steps, str(e)) for inp in user_inputs]
            return error_results[0] if is_single else error_results

        total_generation_time = time.perf_counter() - start_time
        
        results = []
        for i in range(len(user_inputs)):
            # Get the unpadded generated tokens for this specific sample
            generated_ids = outputs.sequences[i, input_ids.shape[1]:]
            
            # Find where the padding starts (first pad_token_id)
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is not None:
                # Find indices of non-pad tokens
                non_pad_mask = generated_ids != pad_token_id
                if non_pad_mask.any():
                    # Last non-pad token index
                    last_non_pad = non_pad_mask.nonzero()[-1].item() + 1
                    generated_ids = generated_ids[:last_non_pad]
            
            # Decode only the actual generated tokens
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            full_response = template["force_start_text"] + " " + generated_text
            
            # Count actual generated tokens
            actual_generated_tokens = len(generated_ids)
            
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
                "generation_time": total_generation_time,  # Total batch time (not divided)
                "generated_tokens": actual_generated_tokens,  # Correct count
                "input_tokens": input_lengths[i].item(),  # Original unpadded length
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