import re
from datasets import Dataset
import json

def generate_arithmetic_cot(prompt: str, target: str) -> str:
    """
    Generates the Chain-of-Thought (CoT) string for a multi-step addition problem.
    
    Example: "711 + 234 + 666 + 474 =" -> "711 + 234 = 945. 945 + 666 = 1611. 1611 + 474 = 2085."
    """
    # 1. Extract all numbers from the prompt string
    numbers_str = prompt.replace('=', '').strip()
    numbers = [int(n) for n in re.findall(r'\d+', numbers_str)]
    
    if not numbers:
        return f"The final sum is: {target}" # Fallback
    
    intermediate_sum = 0
    cot_steps = []
    
    # 2. Iterate and generate step-by-step calculations
    for i, num in enumerate(numbers):
        if i == 0:
            # Start with the first number
            intermediate_sum = num
        else:
            # Current step is: previous sum + current number
            current_sum = intermediate_sum
            
            # Use Python's calculation to get the correct intermediate sum
            # Note: eval() is safe here as input is controlled numbers/operators.
            intermediate_sum = current_sum + num
            
            # Format the step as text for the model
            cot_steps.append(f"{current_sum} + {num} = {intermediate_sum}.")
            
    # 3. Combine steps into the final completion structure
    cot_string = " ".join(cot_steps)
    
    # Final structure for the model
    completion = (
        f"Let's calculate the sum step by step. {cot_string}. "
        f"The final sum is: {target}"
    )
    
    # Simple check to ensure the generated sum matches the target
    if str(intermediate_sum) != target:
        print(f"Warning: Calculated sum ({intermediate_sum}) does not match target ({target}) for prompt: {prompt}")
        
    return completion

# 1. Create the 'completion' field
raw_data = json.load(open("n-ary-data.json", "r"))
formatted_data = []
for item in raw_data:
    completion = generate_arithmetic_cot(item["prompt"], item["target"])
    formatted_data.append({
        "prompt": item["prompt"],
        "completion": completion
    })
print(formatted_data[:2])  # Print first 2 entries for verification
# 2. Convert to Hugging Face Dataset for training
dataset = Dataset.from_list(formatted_data)