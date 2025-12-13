import random
import json

def generate_variable_assignment(num_samples=1000, depth=0, num_vars=10, mode='math'):
    """
    Generates Depth-k Variable Assignment data.
    
    Args:
        num_samples: Number of samples to generate.
        depth: The 'k' in depth-k (chain length). 
               0 = direct assignment (x=1, y=x? -> No, y=2? Yes).
               Image example Depth 0: a=1, b=2, c=6, b=? -> 2 (Direct lookup)
               Image example Depth 1: a=1, b=2, c=a, d=b, d=? -> 2 (1 hop)
        num_vars: Total number of variables in the context window.
        mode: 'math' (symbolic) or 'code' (python-like).
    """
    data = []
    # Pool of variable names
    var_names = [chr(ord('a') + i) for i in range(26)] # a, b, c...
    
    for _ in range(num_samples):
        # 1. Create base values (constants)
        assignments = {}
        history = []
        
        # We need a chain of length 'depth' ending at the target
        # Chain: v_0 -> v_1 -> ... -> v_k
        
        # Pick random variables for the chain
        chain_vars = random.sample(var_names, depth + 1)
        
        # The base of the chain (v_0) gets a number
        base_val = random.randint(0, 9)
        assignments[chain_vars[0]] = base_val
        history.append((chain_vars[0], str(base_val)))
        
        # Build the chain (v_i = v_{i-1})
        for i in range(1, depth + 1):
            assignments[chain_vars[i]] = assignments[chain_vars[i-1]]
            history.append((chain_vars[i], chain_vars[i-1]))
            
        # 2. Add distractor variables (noise)
        # Distractors can be constants (x=5) or point to other existing vars (y=x)
        # To match the image "a=1, b=2, c=6", we mostly use constants for depth 0 distractions
        remaining_vars = [v for v in var_names if v not in chain_vars][:num_vars - (depth + 1)]
        
        for v in remaining_vars:
            if random.random() < 0.5 and len(assignments) > 0:
                # Assign to existing var
                ref_var = random.choice(list(assignments.keys()))
                assignments[v] = assignments[ref_var]
                history.append((v, ref_var))
            else:
                # Assign to constant
                val = random.randint(0, 9)
                assignments[v] = val
                history.append((v, str(val)))
                
        # Shuffle the context so the chain isn't obvious
        random.shuffle(history)
        
        # The query is the last variable in our chain
        query_var = chain_vars[-1]
        target = str(assignments[query_var])
        
        # Format the output based on mode
        if mode == 'math':
            # Format: "a=1, b=2, c=a, d=b, d=?"
            context_str = ", ".join([f"{k}={v}" for k, v in history])
            prompt = f"{context_str}, {query_var}=?"
        else: # code
            # Format: "a=1\nb=2\nc=a\nd=b\nprint(d)"
            context_str = "\n".join([f"{k}={v}" for k, v in history])
            prompt = f"{context_str}\nprint({query_var})"
            
        data.append({
            "task": f"var_assign_depth_{depth}_{mode}",
            "depth": depth,
            "prompt": prompt,
            "target": target
        })
        
    return data

# Generate Depth-0 and Depth-1 as per image description
dataset = []
print("Starting generating reasoning primitives sample...")
# Math variants
dataset.extend(generate_variable_assignment(100_000, depth=0, mode='math'))
dataset.extend(generate_variable_assignment(100_000, depth=1, mode='math'))

# Coding variants
dataset.extend(generate_variable_assignment(100_000, depth=0, mode='code'))
dataset.extend(generate_variable_assignment(100_000, depth=1, mode='code'))

# Save
with open("reasoning_primitives_data.json", "w") as f:
    json.dump(dataset, f, indent=2)
    
print(f"Generated {len(dataset)} reasoning primitive samples.")