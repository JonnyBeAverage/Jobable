from summarizer import summarize_text, truncate_to_token_limit
from model import generator

def create_cover_letter(cv_text: str, jd_text: str):

    summarized_cv = summarize_text(cv_text)
    summarized_jd = summarize_text(jd_text)

    raw_prompt = f"""
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

    safe_prompt = truncate_to_token_limit(raw_prompt)

    result = generator(
        safe_prompt,
        max_new_tokens=350,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return result[0]["generated_text"]
