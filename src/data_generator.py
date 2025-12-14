import random
import json
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

def create_test_datasets(config: dict) -> Dict[str, List[Dict]]:
    """
    Generate algorithmic test datasets strictly matching ICLR 2025 specs.
    Ensures prompts match the exact format expected by model templates.
    """
    test_data = {}

    if "n_ary" in config:
        n_ary_data = []
        ops_levels = config["n_ary"].get("ops_levels", [8, 16, 24, 32])
        num_samples = config["n_ary"].get("num_samples_per_level", 30)

        for n in ops_levels:
            for _ in range(num_samples):
                nums_int = [random.randint(0, 999) for _ in range(n)]
                nums_str = [str(x).zfill(3) for x in nums_int]
                # Format matching the few-shot examples: "100 + 200 + 300 ="
                prompt_str = " + ".join(nums_str) + " ="
                target_str = str(sum(nums_int))

                n_ary_data.append({
                    "prompt": prompt_str,
                    "expected_answer": target_str,
                    "difficulty": f"{n}_ops",
                    "task_type": "n_ary",
                    "numbers": nums_int,  # Keep for verification
                    "sum": sum(nums_int),
                })
        test_data["n_ary"] = n_ary_data

    if "p_hop" in config:
        p_hop_data = []
        alphabet = ["A", "B", "C", "D"]
        seq_len = config["p_hop"].get("sequence_length", 256)
        hop_levels = config["p_hop"].get("hop_levels", [16, 24, 32])
        num_samples = config["p_hop"].get("num_samples_per_level", 30)

        for p in hop_levels:
            for _ in range(num_samples):
                # Ensure indices fit within sequence
                if p + 1 >= seq_len:
                    # Adjust sequence length for large hops
                    seq_len_adjusted = max(seq_len, (p + 1) * 2)
                    indices = random.sample(range(seq_len_adjusted), p + 1)
                else:
                    indices = random.sample(range(seq_len), p + 1)
                
                indices.sort()
                
                # Generate chain that ensures each hop is possible
                chain = []
                # First element can be any letter
                chain.append(random.choice(alphabet))
                
                # Ensure each subsequent element follows the sequence pattern
                for i in range(1, p + 1):
                    # Make sure the next token exists in the alphabet
                    possible_next = [c for c in alphabet if c != chain[-1]] or alphabet
                    chain.append(random.choice(possible_next))
                
                # Create sequence with embedded chain
                seq = [random.choice(alphabet) for _ in range(seq_len)]
                for k, idx in enumerate(indices):
                    if idx < len(seq):
                        seq[idx] = chain[k]
                    else:
                        # Extend sequence if needed
                        seq.extend([random.choice(alphabet)] * (idx - len(seq) + 1))
                        seq[idx] = chain[k]

                seq_str = "".join(seq[:seq_len])  # Trim to exact length
                start_node = chain[0]
                expected = chain[-1]
                
                # Format matching few-shot: "Sequence: ABC... Start: A. Hop X times."
                full_prompt = f"Sequence: {seq_str}. Start: {start_node}. Hop {p} times."

                p_hop_data.append({
                    "prompt": full_prompt,
                    "expected_answer": expected,
                    "difficulty": f"{p}_hops",
                    "task_type": "p_hop",
                    "sequence": seq_str,
                    "chain": chain,
                    "indices": indices[:seq_len],
                })
        test_data["p_hop"] = p_hop_data

    if "igsm" in config:
        igsm_data = []
        num_total = config["igsm"].get("num_samples_total", 50)
        chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        def get_var_name(existing: set = None) -> str:
            """Get unique variable name"""
            while True:
                var = f"{random.choice(chars)}#{random.choice(chars)}"
                if existing is None or var not in existing:
                    return var
        
        def create_hierarchical_equations() -> Tuple[List[str], Dict[str, int], str, int]:
            """Create properly hierarchical equations with depth constraint"""
            all_vars = set()
            levels = defaultdict(list)
            var_values = {}
            equations = []
            
            # Level 0: Base assignments (values 0-6)
            num_base = random.randint(3, 5)
            for _ in range(num_base):
                var = get_var_name(all_vars)
                all_vars.add(var)
                val = random.randint(0, 6)
                var_values[var] = val
                levels[0].append(var)
                equations.append(f"{var} := {val}")
            
            # Levels 1-3: Dependencies on previous level only
            for level in range(1, 4):
                num_vars = random.randint(2, 4)
                for _ in range(num_vars):
                    var = get_var_name(all_vars)
                    all_vars.add(var)
                    
                    # Only reference variables from previous level
                    available_refs = levels[level - 1]
                    if not available_refs:
                        available_refs = levels[0]  # Fallback
                    
                    # Use 1 or 2 operands from previous level
                    num_operands = random.randint(1, 2)
                    operands = random.sample(available_refs, min(num_operands, len(available_refs)))
                    op_vals = [var_values[op] for op in operands]
                    
                    # Choose operation
                    if len(operands) == 1:
                        # Assignment from previous level
                        res = op_vals[0]
                        stmt = f"{var} := {operands[0]}"
                    else:
                        # Binary operation
                        op_type = random.choice(["+", "-", "*"])
                        if op_type == "+":
                            res = (op_vals[0] + op_vals[1]) % 7
                            stmt = f"{var} := {operands[0]} + {operands[1]}"
                        elif op_type == "-":
                            res = (op_vals[0] - op_vals[1]) % 7
                            stmt = f"{var} := {operands[0]} - {operands[1]}"
                        else:  # "*"
                            res = (op_vals[0] * op_vals[1]) % 7
                            stmt = f"{var} := {operands[0]} * {operands[1]}"
                    
                    var_values[var] = res
                    levels[level].append(var)
                    equations.append(stmt)
            
            # Target variable from deepest level
            target_level = 3 if levels[3] else (2 if levels[2] else 1)
            target_var = random.choice(levels[target_level])
            target_val = var_values[target_var]
            
            return equations, var_values, target_var, target_val
        
        for _ in range(num_total):
            equations, var_values, target_var, target_val = create_hierarchical_equations()
            
            # Shuffle equations but maintain solvability (base definitions first in practice)
            random.shuffle(equations)
            
            # Format matching few-shot: "Question. A#A := 4. A#B := A#A + 2. A#B?"
            full_prompt = "Question. " + ". ".join(equations) + f". {target_var}?"

            igsm_data.append({
                "prompt": full_prompt,
                "expected_answer": str(target_val),
                "difficulty": "depth_4_hierarchical_mod_7",
                "task_type": "igsm",
                "equations": equations,
                "target_var": target_var,
                "all_values": var_values,
            })
    
    test_data["igsm"] = igsm_data
    return test_data

def create_perplexity_data(num_samples: int = 30) -> List[str]:
    """Generate reasoning traces for perplexity calculation"""
    perplexity_texts = []
    
    # Format to match model's expected templates
    # N-ARY traces
    for _ in range(num_samples // 2):
        n = random.choice([4, 6, 8])
        nums = [str(random.randint(10, 999)).zfill(3) for _ in range(n)]
        # Match the exact format from n_ary template
        prompt = " + ".join(nums) + " ="
        trace = f"Current: 0\n"
        current_sum = 0
        for num in nums:
            num_int = int(num)
            current_sum += num_int
            trace += f"Add {num}: {current_sum - num_int} + {num_int} = {current_sum}\n"
            trace += f"Current: {current_sum}\n"
        trace += f"Final: {current_sum}"
        perplexity_texts.append(trace)
    
    # P-HOP traces
    alphabet = ["A", "B", "C", "D"]
    for _ in range(num_samples // 2):
        hops = random.choice([3, 4, 5])
        seq_len = 256
        indices = sorted(random.sample(range(seq_len), hops + 1))
        chain = [random.choice(alphabet) for _ in range(hops + 1)]
        
        # Create trace in model's expected format
        trace = f"Start at {chain[0]}.\n"
        current_idx = indices[0]
        for i in range(hops):
            # Find next occurrence after current position
            next_idx = indices[i + 1]
            trace += f"Found '{chain[i]}' at index {next_idx}.\n"
            if i < hops:
                trace += f"Next token is {chain[i + 1]}.\n"
        
        trace += f"Final: {chain[-1]}"
        perplexity_texts.append(trace)
    
    return perplexity_texts

def load_and_preprocess_data(file_path: str) -> Dict[str, List[Dict]]:
    """Load existing test data from JSON or CSV"""
    print(f"Loading data from: {file_path}")
    
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        raw_data = df.to_dict("records")
    else:
        raise ValueError("Unsupported format. Use .json or .csv")

    processed_data = {"n_ary": [], "p_hop": [], "igsm": []}
    stats = {"n_ary": 0, "p_hop": 0, "igsm": 0, "unknown": 0}
    
    for record in raw_data:
        task = record.get("task_type", "").lower()
        
        # Normalize task type names
        if "n_ary" in task or "addition" in task:
            task_key = "n_ary"
        elif "p_hop" in task or "induction" in task:
            task_key = "p_hop"
        elif "igsm" in task or "gsm" in task or "symbolic" in task:
            task_key = "igsm"
        else:
            stats["unknown"] += 1
            continue
        
        # Validate required fields
        required = ["prompt", "expected_answer"]
        if all(k in record for k in required):
            processed_data[task_key].append({
                "prompt": str(record["prompt"]),
                "expected_answer": str(record["expected_answer"]),
                "difficulty": record.get("difficulty", "unknown"),
                "task_type": task_key,
                "original_data": record  # Keep original for debugging
            })
            stats[task_key] += 1
    
    print(f"✅ Loaded - N-ary: {stats['n_ary']}, "
          f"P-Hop: {stats['p_hop']}, iGSM: {stats['igsm']}, "
          f"Skipped: {stats['unknown']}")
    
    return processed_data

def create_reasoning_primitives_data(config: dict) -> Dict[str, List[Dict]]:
    """
    Generates 'Reasoning Primitives' datasets as described.
    Now includes formats matching model templates.
    """
    if "reasoning_primitives" not in config:
        return {}

    primitives_data = {}
    cfg = config["reasoning_primitives"]
    num_samples = cfg.get("num_samples", 50)
    
    # Different prompt formats for testing generalization
    formats = {
        "code": {"assign": "{var} = {val}", "query": "print({var})", "sep": "\n"},
        "math": {"assign": "Let {var} = {val}.", "query": "What is {var}?", "sep": " "},
        "equation": {"assign": "{var} := {val}", "query": "{var}?", "sep": ". "},  # Matches i-GSM style
    }
    
    chars = list("abcdefghijklmnopqrstuvwxyz")
    
    for depth in [0, 1]:
        for variant in ["code", "math", "equation"]:
            task_name = f"var_assign_depth_{depth}_{variant}"
            samples = []
            fmt = formats[variant]
            
            for _ in range(num_samples):
                num_vars = random.randint(5, 8)
                vars_subset = random.sample(chars, num_vars)
                values = {v: random.randint(0, 99) for v in vars_subset}
                statements = []
                
                if depth == 0:
                    # Direct assignments only
                    for v in vars_subset:
                        stmt = fmt["assign"].format(var=v, val=values[v])
                        statements.append(stmt)
                    target_var = random.choice(vars_subset)
                    expected = str(values[target_var])
                    
                elif depth == 1:
                    # Two-level hierarchy
                    roots = vars_subset[:num_vars // 2]
                    pointers = vars_subset[num_vars // 2:]
                    
                    # Root assignments
                    for v in roots:
                        stmt = fmt["assign"].format(var=v, val=values[v])
                        statements.append(stmt)
                    
                    # Pointer assignments (reference roots)
                    for v in pointers:
                        src = random.choice(roots)
                        stmt = fmt["assign"].format(var=v, val=src)
                        statements.append(stmt)
                        values[v] = values[src]  # Track derived value
                    
                    target_var = random.choice(pointers)
                    expected = str(values[target_var])

                # Shuffle statements (model must handle arbitrary order)
                random.shuffle(statements)
                
                # Build prompt
                context = fmt["sep"].join(statements)
                query = fmt["query"].format(var=target_var)
                
                # Handle different separators for query
                if variant == "code":
                    full_prompt = f"{context}\n{query}"
                elif variant == "math":
                    full_prompt = f"{context} {query}"
                else:  # equation
                    full_prompt = f"{context}. {query}"
                
                samples.append({
                    "prompt": full_prompt,
                    "expected_answer": expected,
                    "difficulty": f"depth_{depth}",
                    "task_type": task_name,
                    "variant": variant,
                    "depth": depth,
                })
            
            primitives_data[task_name] = samples

    return primitives_data

def format_5_shot_prompt(task_samples: List[Dict], current_sample: Dict, 
                         template_format: str = "plain") -> str:
    """
    Helper to prepend 5 random examples for few-shot evaluation.
    Supports different formatting styles.
    """
    pool = [s for s in task_samples if s["prompt"] != current_sample["prompt"]]
    if len(pool) < 5:
        shots = pool
    else:
        shots = random.sample(pool, 5)
    
    if template_format == "chat":
        # Format for chat models
        demos = []
        for shot in shots:
            demos.append(f"User: {shot['prompt']}")
            demos.append(f"Assistant: {shot['expected_answer']}")
        
        few_shot_context = "\n".join(demos)
        final_prompt = f"{few_shot_context}\nUser: {current_sample['prompt']}\nAssistant:"
    
    elif template_format == "instruction":
        # Instruction-response format
        demos = []
        for shot in shots:
            demos.append(f"### Instruction:\n{shot['prompt']}\n\n### Response:\n{shot['expected_answer']}")
        
        few_shot_context = "\n\n".join(demos)
        final_prompt = f"{few_shot_context}\n\n### Instruction:\n{current_sample['prompt']}\n\n### Response:"
    
    else:  # plain format (default)
        demos = []
        for shot in shots:
            demos.append(f"{shot['prompt']}\nAnswer: {shot['expected_answer']}")
        
        few_shot_context = "\n\n".join(demos)
        final_prompt = f"{few_shot_context}\n\n{current_sample['prompt']}\nAnswer:"
    
    return final_prompt

def save_datasets(data_dict: Dict, output_dir: str = "./data"):
    """Save generated datasets to organized files"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for task_type, samples in data_dict.items():
        if samples:
            # Save as JSON
            json_path = os.path.join(output_dir, f"{task_type}_dataset.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)
            
            # Save as CSV for easy viewing
            csv_path = os.path.join(output_dir, f"{task_type}_dataset.csv")
            df = pd.DataFrame(samples)
            df.to_csv(csv_path, index=False)
            
            print(f"✅ Saved {len(samples)} {task_type} samples to {json_path}")