from jobable.ml_logic.summarizer import summarize_resume, summarize_job_description
from jobable.ml_logic.model import tokenizer as _default_tokenizer, model as _default_model, MODEL_PATH
from transformers import GenerationConfig
from pathlib import Path
import torch
import re
import time

generation_config = GenerationConfig.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

def trim_cover_letter(text):
    pattern = r"Sincerely,\s*\n?\[Your Name\].*"
    return re.sub(pattern, "Sincerely,\nIsaac Shane", text, flags=re.IGNORECASE | re.DOTALL)

def create_cover_letter(cv_text: str, jd_text: str, tokenizer=None, model=None):
    """Pass tokenizer and model to reuse cached instances (e.g. from app get_cover_letter_model())."""
    tok = tokenizer if tokenizer is not None else _default_tokenizer
    mod = model if model is not None else _default_model

    summarized_cv = summarize_resume(cv_text, tokenizer=tok, model=mod)
    summarized_jd = summarize_job_description(jd_text, tokenizer=tok, model=mod)

    prompt = f"""
    Resume Highlights:
    {summarized_cv}

    Job Requirements:
    {summarized_jd}

    Write a one page (around 400 words) professional cover letter tailored to this role.
    Be entheusiastic, forward looking, and professional. Do not write any code.

    End the cover letter with
    "Sincerely,
    Isaac Shane"


    Cover Letter:
    Dear Hiring Manager,
    """

    device = mod.device
    inputs = tok(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        outputs = mod.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    return trim_cover_letter(tok.decode(generated_tokens, skip_special_tokens=True))


def fake_cover_letter():
    time.sleep(60)
    letter = (
    "My name is Isaac Shane, and I am writing to express my interest in the Network Engineer (Test Lab) role at your organization. "
    "I have been following your company's progress for several months now and was excited to see that you are looking for talented "
    "individuals to join your team. After reviewing the job description, I believe that I possess many of the skills and qualifications "
    "necessary to excel in this role.\n"
    "\tAs a Network Engineer with 5+ years of experience in the industry, I am confident that I would be able to make significant "
    "contributions to your organization. My expertise in networking, combined with my basic to intermediate programming language "
    "skills, would enable me to develop and execute automated Network OS compliance tests for network vendor hardware used in the "
    "corporate network environment. Additionally, I am proficient with circuit board and microscope soldering skills and rework, "
    "which would be beneficial when troubleshooting and fixing problems with Test Infrastructure.\n\n"
    "\tIn terms of education, I hold a Bachelor's degree in Computer Science, which has equipped me with the necessary knowledge "
    "and skills to succeed in the IT sector. Moreover, I am eager to learn more about DevOps and Test Automation frameworks, "
    "which could be valuable assets to your company.\n\n"
    "\tOverall, I am very interested in joining your team and contributing to your organization's continued success. Thank you for "
    "considering my application. Please do not hesitate to contact me if you require any further information.\n\n"
    "Best regards,\n\n"
    "Isaac Shane"
    )
    return letter

if __name__ == "__main__":
    print(create_cover_letter(
        "Testing resume text",
        "Testing job description"
    ))
    # With cached model (e.g. from app):
    # tok, mod = get_cover_letter_model()
    # print(create_cover_letter("Resume...", "JD...", tokenizer=tok, model=mod))



### ========ISAAC VERSION FOR TESTING============

# from .summarizer import summarize_resume, summarize_job_description
# from .model import tokenizer as _default_tokenizer, model as _default_model


# def create_cover_letter(cv_text: str, jd_text: str, tokenizer=None, model=None):
#     """Generate a cover letter. Pass tokenizer and model to reuse a cached model."""
#     tok = tokenizer if tokenizer is not None else _default_tokenizer
#     mod = model if model is not None else _default_model

#     summarized_cv = summarize_resume(cv_text, tokenizer=tok, model=mod)
#     summarized_jd = summarize_job_description(jd_text, tokenizer=tok, model=mod)

#     input_text = f"""
#     You are a professional executive career coach.

#     Write a tailored cover letter using the structure below.

#     Structure:
#     Paragraph 1:
#     - Express interest in the specific role.
#     - Mention years of leadership experience.

#     Paragraph 2:
#     - Connect 2-3 specific achievements to the job requirements.
#     - Be concrete and avoid repetition.

#     Paragraph 3:
#     - Reinforce cultural fit and leadership strengths.

#     Paragraph 4:
#     - Confident closing and call to action.

#     Rules:
#     - Begin with: Dear Hiring Manager,
#     - Do not repeat phrases.
#     - Avoid generic language.
#     - Do not restate the resume summary.
#     - Keep under 300 words.

#     Candidate Highlights:
#     {summarized_cv}

#     Role Requirements:
#     {summarized_jd}

#     Write the full letter now.
#     """
#     input_ids = tok(input_text, return_tensors="pt").input_ids
#     outputs = mod.generate(input_ids, max_new_tokens=350)

#     return tok.decode(outputs[0], skip_special_tokens=True)

# if __name__ == '__main__':
#     print(create_cover_letter('does this work? I am testing this', 'this does work. The test has succeeded'))
