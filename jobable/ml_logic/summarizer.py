# # from .model import tokenizer as _default_tokenizer, model as _default_model
# import torch


# def truncate_to_token_limit(text, max_tokens=900, tokenizer=None):
#     tok = tokenizer if tokenizer is not None else _default_tokenizer
#     tokens = tok(
#         text,
#         truncation=True,
#         max_length=max_tokens,
#         return_tensors="pt"
#     )
#     return tok.decode(tokens["input_ids"][0], skip_special_tokens=True)


# def generate_text(prompt, max_tokens=150, tokenizer=None, model=None):
#     tok = tokenizer if tokenizer is not None else _default_tokenizer
#     mod = model if model is not None else _default_model
#     device = mod.device

#     inputs = tok(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         padding=True
#     )

#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = mod.generate(
#             **inputs,
#             max_new_tokens=120,
#             do_sample=True,
#             temperature=0.4
#         )

#     return tok.decode(outputs[0], skip_special_tokens=True)


# def summarize_resume(resume_text: str, tokenizer=None, model=None):
#     tok = tokenizer if tokenizer is not None else _default_tokenizer
#     safe_text = truncate_to_token_limit(resume_text, tokenizer=tok)

#     prompt = f"""
# Extract the most important professional highlights from this resume.

# Return 4-6 concise bullet points including:
# - years of experience
# - key skills
# - leadership or achievements
# - industries worked in

# Resume:
# {safe_text}

# Bullet Point Summary:
# """

#     mod = model if model is not None else _default_model
#     return generate_text(prompt, tokenizer=tok, model=mod)


# def summarize_job_description(jd_text: str, tokenizer=None, model=None):
#     tok = tokenizer if tokenizer is not None else _default_tokenizer
#     safe_text = truncate_to_token_limit(jd_text, tokenizer=tok)

#     prompt = f"""
# Summarize the key requirements of this job description.

# Return 4-6 bullet points including:
# - core skills required
# - technologies
# - experience expectations
# - main responsibilities

# Job Description:
# {safe_text}

# Key Requirements:
# """

#     mod = model if model is not None else _default_model
#     return generate_text(prompt, tokenizer=tok, model=mod)
