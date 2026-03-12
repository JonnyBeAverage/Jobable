from .summarizer import summarize_resume, summarize_job_description
from .model import tokenizer as _default_tokenizer, model as _default_model



def create_cover_letter(cv_text: str, jd_text: str, tokenizer=None, model=None):
    '''Generate a cover letter. Pass tokenizer and model to reuse a cached model.'''
    tok = tokenizer if tokenizer is not None else _default_tokenizer
    mod = model if model is not None else _default_model

    summarized_cv = summarize_resume(cv_text, tokenizer=tok, model=mod)
    summarized_jd = summarize_job_description(jd_text, tokenizer=tok, model=mod)

    input_text = f"""
    You are a professional executive career coach.

    Write a tailored cover letter using the structure below.

    Structure:
    Paragraph 1:
    - Express interest in the specific role.
    - Mention years of leadership experience.

    Paragraph 2:
    - Connect 2-3 specific achievements to the job requirements.
    - Be concrete and avoid repetition.

    Paragraph 3:
    - Reinforce cultural fit and leadership strengths.

    Paragraph 4:
    - Confident closing and call to action.

    Rules:
    - Begin with: Dear Hiring Manager,
    - Do not repeat phrases.
    - Avoid generic language.
    - Do not restate the resume summary.
    - Keep under 300 words.

    Candidate Highlights:
    {summarized_cv}

    Role Requirements:
    {summarized_jd}

    Write the full letter now.
    """
    input_ids = tok(input_text, return_tensors="pt").input_ids
    outputs = mod.generate(input_ids, max_new_tokens=350)

    return tok.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(create_cover_letter("does this work? I am testing this", "this does work. The test has succeeded"))
