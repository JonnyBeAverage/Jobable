from .model import tokenizer, model
import torch


def truncate_to_token_limit(text, max_tokens=450):
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt"
    )
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)


def generate_text(prompt, max_tokens=150):

    device = model.device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.4
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def summarize_resume(resume_text: str):

    safe_text = truncate_to_token_limit(resume_text)

    prompt = f"""
Extract the most important professional highlights from this resume.

Return 4-6 concise bullet points including:
- years of experience
- key skills
- leadership or achievements
- industries worked in

Resume:
{safe_text}

Bullet Point Summary:
"""

    return generate_text(prompt)


def summarize_job_description(jd_text: str):

    safe_text = truncate_to_token_limit(jd_text)

    prompt = f"""
Summarize the key requirements of this job description.

Return 4-6 bullet points including:
- core skills required
- technologies
- experience expectations
- main responsibilities

Job Description:
{safe_text}

Key Requirements:
"""

    return generate_text(prompt)
