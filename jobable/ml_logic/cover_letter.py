from summarizer import summarize_text
from model import tokenizer, model

def create_cover_letter(cv_text: str, jd_text: str):

    summarized_cv = summarize_text(cv_text)
    summarized_jd = summarize_text(jd_text)

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
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=350)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
