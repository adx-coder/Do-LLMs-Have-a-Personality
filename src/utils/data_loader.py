
import json
from pathlib import Path


def load_prompt_pairs(base_path='data/base_prompts.json', 
                     cf_path='data/counterfactual_prompts.json'):
    base_path = Path(base_path)
    cf_path = Path(cf_path)
    
    with open(base_path, 'r') as f:
        base_prompts = json.load(f)
    with open(cf_path, 'r') as f:
        cf_prompts = json.load(f)
    
    base_dict = {item['prompt_id']: item for item in base_prompts}
    cf_dict = {item['prompt_id']: item for item in cf_prompts}
    
    prompt_pairs = []
    for prompt_id in base_dict.keys():
        if prompt_id in cf_dict:
            prompt_pairs.append({
                'prompt_id': prompt_id,
                'base_prompt': base_dict[prompt_id]['prompt'],
                'cf_prompt': cf_dict[prompt_id]['prompt'],
                'original_id': base_dict[prompt_id].get('original_id', ''),
                'ethical_option_base': base_dict[prompt_id].get('ethical_option', ''),
                'ethical_option_cf': cf_dict[prompt_id].get('ethical_option', '')
            })
    
    print(f"Loaded {len(prompt_pairs)} matched prompt pairs")
    return prompt_pairs


def save_prompt_pairs(prompt_pairs, output_path='data/prompt_pairs.json'):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(prompt_pairs, f, indent=2)
    
    print(f"Saved {len(prompt_pairs)} prompt pairs to {output_path}")


def validate_prompt_pairs(prompt_pairs):
    required_fields = ['prompt_id', 'base_prompt', 'cf_prompt']
    valid_pairs = []
    invalid_pairs = []
    
    for pair in prompt_pairs:
        if all(field in pair for field in required_fields):
            valid_pairs.append(pair)
        else:
            invalid_pairs.append(pair)
    
    if invalid_pairs:
        print(f"Warning: {len(invalid_pairs)} invalid prompt pairs found")
    
    return valid_pairs, invalid_pairs