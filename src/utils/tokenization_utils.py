def get_choice_tokens_robust(tokenizer):
    print("Analyzing tokenization of '1' and '2'...")
    
    # Test various contexts where '1' and '2' might appear
    test_contexts = [
        "Choice 1", 
        "Choice  1", 
        "Option 1:", 
        " 1 is", 
        "\n1)", 
        "1.",
        "Answer: 1",
        "Select 1"
    ]
    
    token_1_candidates = set()
    token_2_candidates = set()
    
    # Find tokens that differ between '1' and '2' contexts
    for context in test_contexts:
        full_1 = tokenizer.encode(context, add_special_tokens=False)
        full_2 = tokenizer.encode(context.replace('1', '2'), add_special_tokens=False)
        
        for i, (t1, t2) in enumerate(zip(full_1, full_2)):
            if t1 != t2:
                token_1_candidates.add(t1)
                token_2_candidates.add(t2)
    
    # Also test direct encoding
    test_string_1 = "Answer: 1"
    test_string_2 = "Answer: 2"
    
    tokens_1 = tokenizer.encode(test_string_1, add_special_tokens=False)
    tokens_2 = tokenizer.encode(test_string_2, add_special_tokens=False)
    
    if len(tokens_1) > 0 and len(tokens_2) > 0:
        token_1_candidates.add(tokens_1[-1])
        token_2_candidates.add(tokens_2[-1])
    
    # Find most common token IDs across contexts
    if token_1_candidates and token_2_candidates:
        token_1 = max(
            token_1_candidates, 
            key=lambda t: sum(
                1 for c in test_contexts 
                if t in tokenizer.encode(c, add_special_tokens=False)
            )
        )
        token_2 = max(
            token_2_candidates, 
            key=lambda t: sum(
                1 for c in test_contexts 
                if t in tokenizer.encode(c.replace('1', '2'), add_special_tokens=False)
            )
        )
        
        if token_1 != token_2:
            print(f"Found distinct tokens:")
            print(f"Token '1': ID={token_1} -> '{tokenizer.decode([token_1]).strip()}'")
            print(f"Token '2': ID={token_2} -> '{tokenizer.decode([token_2]).strip()}'")
            return token_1, token_2
    
    raise ValueError(
        "Cannot find distinct token IDs for '1' and '2'! "
        "This model may tokenize digits in an unexpected way."
    )


def get_token_id(tokenizer, token_str, context="Answer: "):
    full_string = f"{context}{token_str}"
    token_ids = tokenizer.encode(full_string, add_special_tokens=False)
    return token_ids[-1]


def decode_tokens(tokenizer, token_ids):
    return [tokenizer.decode([tid]).strip() for tid in token_ids]


def analyze_tokenization(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    return {
        'text': text,
        'num_tokens': len(token_ids),
        'token_ids': token_ids,
        'tokens': tokens,
        'decoded': tokenizer.decode(token_ids)
    }