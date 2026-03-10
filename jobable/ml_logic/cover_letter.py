from jobable.ml_logic.summarizer import summarize_resume, summarize_job_description
from jobable.ml_logic.model import tokenizer, model, MODEL_PATH
from transformers import GenerationConfig
from pathlib import Path
import torch

generation_config = GenerationConfig.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

def create_cover_letter(cv_text: str, jd_text: str):

    summarized_cv = summarize_resume(cv_text)
    summarized_jd = summarize_job_description(jd_text)

    prompt = f"""
    Resume Highlights:
    {summarized_cv}

    Job Requirements:
    {summarized_jd}

    Write a professional cover letter tailored to this role.

    Cover Letter:
    Dear Hiring Manager,
    """

    device = model.device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    print(create_cover_letter(
        "Testing resume text",
        "Testing job description"
    ))
