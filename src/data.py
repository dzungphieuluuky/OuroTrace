import random
import json
import pandas as pd
from typing import Dict, List

def create_test_datasets(config: dict) -> Dict[str, List[Dict]]:
    """Generate algorithmic test datasets (N-ary, P-hop, iGSM)"""
    test_data = {}
    
    # 1. N-ARY ADDITION
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
                    while target_var in all_vars_data: target_var = get_var_name()
                    
                    operands = random.choices(levels[i-1], k=random.randint(1, 2))
                    op_vals = [all_vars_data[op] for op in operands]
                    op_type = random.choice(['add', 'sub', 'mult', 'assign'])
                    
                    stmt, res = "", 0
                    if op_type == 'assign' or len(operands) < 2:
                        stmt, res = f"{target_var} := {operands[0]}", op_vals[0]
                    elif op_type == 'add':
                        stmt, res = f"{target_var} := {operands[0]} + {operands[1]}", (op_vals[0] + op_vals[1]) % 7
                    elif op_type == 'sub':
                        stmt, res = f"{target_var} := {operands[0]} - {operands[1]}", (op_vals[0] - op_vals[1]) % 7
                    elif op_type == 'mult':
                        stmt, res = f"{target_var} := {operands[0]} * {operands[1]}", (op_vals[0] * op_vals[1]) % 7
                        
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

def create_reasoning_primitives_data(config: dict) -> Dict[str, List[Dict]]:
    if 'reasoning_primitives' not in config: return {}
    primitives_data = {}
    cfg = config['reasoning_primitives']
    num_samples = cfg.get('num_samples', 50)
    
    formats = {
        'code': {'assign': "{var} = {val}", 'query': "print({var})", 'sep': "\n"},
        'math': {'assign': "Let {var} = {val}.", 'query': "What is {var}?", 'sep': " "}
    }
    chars = "abcdefghijklmnopqrstuvwxyz"
    
    for depth in [0, 1]:
        for variant in ['code', 'math']:
            task_name = f"var_assign_depth_{depth}_{variant}"
            samples = []
            fmt = formats[variant]
            
            for _ in range(num_samples):
                num_vars = random.randint(5, 8)
                vars_subset = random.sample(chars, num_vars)
                values = {v: random.randint(0, 99) for v in vars_subset}
                statements = []
                
                if depth == 0:
                    for v in vars_subset:
                        stmt = fmt['assign'].format(var=v, val=values[v])
                        statements.append(stmt)
                    target_var = random.choice(vars_subset)
                    expected = str(values[target_var])
                elif depth == 1:
                    roots = vars_subset[:num_vars//2]
                    for v in roots:
                        stmt = fmt['assign'].format(var=v, val=values[v])
                        statements.append(stmt)
                    pointers = vars_subset[num_vars//2:]
                    for v in pointers:
                        src = random.choice(roots)
                        stmt = fmt['assign'].format(var=v, val=src)
                        statements.append(stmt)
                        values[v] = values[src]
                    target_var = random.choice(pointers)
                    expected = str(values[target_var])

                random.shuffle(statements)
                full_prompt = f"{fmt['sep'].join(statements)}{fmt['sep']}{fmt['query'].format(var=target_var)}"
                samples.append({"prompt": full_prompt, "expected_answer": expected, "difficulty": f"depth_{depth}", "task_type": task_name})
            primitives_data[task_name] = samples
    return primitives_data

def format_5_shot_prompt(task_samples: List[Dict], current_sample: Dict) -> str:
    pool = [s for s in task_samples if s['prompt'] != current_sample['prompt']]
    shots = random.sample(pool, 5) if len(pool) >= 5 else pool
    demos = [f"{shot['prompt']}\nAnswer: {shot['expected_answer']}" for shot in shots]
    return "\n\n".join(demos) + f"\n\n{current_sample['prompt']}\nAnswer:"

def create_perplexity_data(num_samples: int = 30) -> List[str]:
    # Placeholder for simple logic, extended logic can be copied from original if needed
    return ["Sample text for perplexity calculation."]

def load_and_preprocess_data(file_path: str) -> Dict[str, List[Dict]]:
    print(f"Loading data from: {file_path}")
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f: raw_data = json.load(f)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        raw_data = df.to_dict('records')
    else: raise ValueError("Unsupported format.")
    
    processed_data = {'n_ary': [], 'p_hop': [], 'igsm': []}
    for record in raw_data:
        task = record.get('task_type')
        if task in processed_data: processed_data[task].append(record)
    return processed_data