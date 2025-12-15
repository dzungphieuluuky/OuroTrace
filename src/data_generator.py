import random
import json
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

def create_test_datasets(config: dict) -> Dict[str, List[Dict]]:
    """
    Generate algorithmic test datasets strictly matching ICLR 2025 specs:
    1. N-ary Addition: Input-Output pairs, 3-digit operands, sum.
    2. P-hop Induction: Sequence length 256, Alphabet 4, Chain embedded at random sorted indices.
    3. Symbolic i-GSM: Hierarchy depth 4, strict Level i -> Level i+1 dependency, Modulo 7.
    """
    test_data = {}

    # 1. N-ARY ADDITION (Unchanged, matches paper description)
    if 'n_ary' in config:
        n_ary_data = []
        ops_levels = config['n_ary'].get('ops_levels', [8, 16, 24, 32])
        num_samples = config['n_ary'].get('num_samples_per_level', 30)

        for n in ops_levels:
            for _ in range(num_samples):
                nums_int = [random.randint(0, 999) for _ in range(n)]
                nums_str = [str(x).zfill(3) for x in nums_int]
                prompt_str = " + ".join(nums_str) + " ="
                target_str = str(sum(nums_int))

                n_ary_data.append({
                    "prompt": prompt_str,
                    "expected_answer": target_str,
                    "difficulty": f"{n}_ops",
                    "task_type": "n_ary"
                })
        test_data['n_ary'] = n_ary_data

    # 2. P-HOP INDUCTION
    if 'p_hop' in config:
        p_hop_data = []
        alphabet = ['A', 'B', 'C', 'D']
        seq_len = 256
        hop_levels = config['p_hop'].get('hop_levels', [16, 24, 32])
        num_samples = config['p_hop'].get('num_samples_per_level', 30)

        for p in hop_levels:
            for _ in range(num_samples):
                chain = [random.choice(alphabet) for _ in range(p + 1)]
                indices = random.sample(range(seq_len), p + 1)
                indices.sort()
                seq = [random.choice(alphabet) for _ in range(seq_len)]
                for k, idx in enumerate(indices):
                    seq[idx] = chain[k]
                seq_str = "".join(seq)
                start_node = chain[0]
                expected = chain[-1]
                full_prompt = f"Sequence: {seq_str}. Start: {start_node}. Hop {p} times."
                p_hop_data.append({
                    "prompt": full_prompt,
                    "expected_answer": expected,
                    "difficulty": f"{p}_hops",
                    "task_type": "p_hop"
                })
        test_data['p_hop'] = p_hop_data

    # 3. SYMBOLIC i-GSM
    if 'igsm' in config:
        igsm_data = []
        num_total = config['igsm'].get('num_samples_total', 50)
        chars = "ABCDEFGHIJKLMNOP"
        def get_var_name():
            return f"{random.choice(chars)}#{random.choice(chars)}"
        for _ in range(num_total):
            levels = {0: [], 1: [], 2: [], 3: [], 4: []}
            all_vars_data = {}
            equations = []
            for _ in range(4):
                name = get_var_name()
                val = random.randint(0, 6)
                levels[0].append(name)
                all_vars_data[name] = val
                equations.append(f"{name} := {val}")
            for i in range(1, 5):
                num_vars_in_level = random.randint(2, 4)
                for _ in range(num_vars_in_level):
                    target_var = get_var_name()
                    while target_var in all_vars_data:
                        target_var = get_var_name()
                    operands = random.choices(levels[i-1], k=random.randint(1, 2))
                    op_vals = [all_vars_data[op] for op in operands]
                    op_type = random.choice(['add', 'sub', 'mult', 'assign'])
                    stmt = ""
                    res = 0
                    if op_type == 'assign' or len(operands) < 2:
                        stmt = f"{target_var} := {operands[0]}"
                        res = op_vals[0]
                    elif op_type == 'add':
                        stmt = f"{target_var} := {operands[0]} + {operands[1]}"
                        res = (op_vals[0] + op_vals[1]) % 7
                    elif op_type == 'sub':
                        stmt = f"{target_var} := {operands[0]} - {operands[1]}"
                        res = (op_vals[0] - op_vals[1]) % 7
                    elif op_type == 'mult':
                        stmt = f"{target_var} := {operands[0]} * {operands[1]}"
                        res = (op_vals[0] * op_vals[1]) % 7
                    equations.append(stmt)
                    all_vars_data[target_var] = res
                    levels[i].append(target_var)
            target_var = random.choice(levels[4])
            target_val = all_vars_data[target_var]
            random.shuffle(equations)
            full_prompt = "Question. " + ". ".join(equations) + f". {target_var}?"
            igsm_data.append({
                "prompt": full_prompt,
                "expected_answer": str(target_val),
                "difficulty": "depth_4_hierarchical_mod_7",
                "task_type": "igsm"
            })
        test_data['igsm'] = igsm_data

    return test_data

def create_perplexity_data(num_samples: int = 30) -> List[str]:
    """Generate reasoning traces for perplexity calculation"""
    perplexity_texts = []

    # N-ARY traces
    for _ in range(num_samples // 2):
        n = random.choice([4, 6, 8])
        nums = [random.randint(10, 99) for _ in range(n)]
        trace = f"System: You are a calculation engine.\nUser: Sum: {nums}\nAssistant: Current Sum: 0\n"
        current_sum = 0
        for num in nums:
            prev_sum = current_sum
            current_sum += num
            trace += f"Add {num}: {prev_sum} + {num} = {current_sum}\nCurrent Sum: {current_sum}\n"
        trace += f"Final: {current_sum}"
        perplexity_texts.append(trace)

    # P-HOP traces
    all_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for _ in range(num_samples // 2):
        hops = random.choice([3, 4, 5])
        nodes = random.sample(all_letters, hops + 1)
        facts = [f"{nodes[i]}->{nodes[i+1]}" for i in range(len(nodes) - 1)]
        facts_str = ", ".join(facts)
        trace = f"System: Logic engine.\nUser: Facts: {facts_str}. Start: {nodes[0]}. Find: {nodes[-1]}.\n"
        trace += f"Assistant: Current Node: {nodes[0]}\n"
        for i in range(hops):
            trace += f"Rule Matches: {nodes[i]} -> {nodes[i+1]}\nNext Node: {nodes[i+1]}\n"
        trace += f"Final: {nodes[-1]}"
        perplexity_texts.append(trace)

    return perplexity_texts

def load_and_preprocess_data(file_path: str) -> Dict[str, List[Dict]]:
    """Load existing test data from JSON or CSV"""
    print(f"Loading data from: {file_path}")
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        raw_data = df.to_dict('records')
    else:
        raise ValueError("Unsupported format. Use .json or .csv")
    
    processed_data = {'n_ary': [], 'p_hop': [], 'igsm': []}
    
    for record in raw_data:
        task = record.get('task_type')
        if task in processed_data:
            if all(k in record for k in ['prompt', 'expected_answer', 'difficulty']):
                processed_data[task].append(record)
    
    print(f"✅ Loaded - N-ary: {len(processed_data['n_ary'])}, "
          f"P-Hop: {len(processed_data['p_hop'])}, iGSM: {len(processed_data['igsm'])}")
    
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