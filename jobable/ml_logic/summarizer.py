from .model import tokenizer, model

def truncate_to_token_limit(text, max_tokens=450):
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt"
    )
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

def summarize_text(text: str, max_tokens: int = 200):
    safe_text = truncate_to_token_limit(text)

    prompt = f"""
    Summarize the following text into concise professional bullet points.
    Focus on key skills, experience, and measurable achievements.

    {safe_text}
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=max_tokens)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
