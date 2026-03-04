"""
Jobable — Streamlit frontend shell.
Top: search + CV upload. Main: scrollable list of jobs with title and description preview.
Generate CV button runs create_cover_letter(cv_text, jd_text).
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO

# Optional: PDF/DOCX text extraction
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

from jobable.ml_logic.cover_letter import create_cover_letter

# Page config
st.set_page_config(page_title="Jobable", page_icon="💼", layout="wide", initial_sidebar_state="collapsed")

# Remove top padding and make page non-scrolling; only the job list iframe scrolls
st.markdown(
    """
    <style>
    /* Remove gap above title */
    .main .block-container { padding-top: 0; }
    /* Page fills viewport, no page scroll */
    .main { height: 100vh; overflow: hidden; }
    .main .block-container { display: flex; flex-direction: column; height: 100%; max-width: 100%; }
    .main .block-container > div { flex: 0 0 auto; }
    /* Job list container takes remaining space and scrolls inside iframe */
    .main .block-container > div:has(iframe) { flex: 1 1 auto; min-height: 0; overflow: hidden; }
    .main .block-container > div:has(iframe) iframe { height: 100% !important; display: block; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load jobs from CSV (cached)
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "jobable" / "data" / "job_title_des.csv"


@st.cache_data
def load_jobs_csv(path: Path):
    """Load job list from job_title_des.csv. Columns: index, Job Title, Job Description."""
    df = pd.read_csv(path)
    # Normalize: use 'Job Title' and 'Job Description'
    title_col = "Job Title"
    desc_col = "Job Description"
    if title_col not in df.columns or desc_col not in df.columns:
        return []
    jobs = []
    for i, row in df.iterrows():
        title = row[title_col]
        desc = row[desc_col]
        if pd.isna(title):
            title = ""
        if pd.isna(desc):
            desc = ""
        # Flatten multiline description to single line for preview
        desc_flat = " ".join(str(desc).split())
        jobs.append({
            "title": str(title).strip(),
            "company": f"Company {i + 1}",
            "description": desc_flat,
        })
    return jobs


JOBS = load_jobs_csv(DATA_PATH)

JOBS_PER_PAGE = 30
NO_CV_ERROR = "Error: No CV uploaded. Please upload a resume to generate a cover letter."


def get_cv_text(uploaded_file) -> str | None:
    """Extract plain text from uploaded CV (txt, pdf, or docx). Returns None if unreadable."""
    if uploaded_file is None:
        return None
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    name = (uploaded_file.name or "").lower()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="replace")
    if name.endswith(".pdf") and PdfReader is not None:
        try:
            reader = PdfReader(BytesIO(raw))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            return None
    if (name.endswith(".docx") or name.endswith(".doc")) and DocxDocument is not None:
        try:
            doc = DocxDocument(BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def job_preview_text(description: str, max_words: int = 12) -> str:
    words = description.strip().split()
    if len(words) <= max_words:
        return description
    return " ".join(words[:max_words]) + "…"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = 0
if "generate_for_job_index" not in st.session_state:
    st.session_state.generate_for_job_index = None
if "last_cover_letter" not in st.session_state:
    st.session_state.last_cover_letter = None
if "last_cover_letter_job_index" not in st.session_state:
    st.session_state.last_cover_letter_job_index = None

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("💼 Jobable")
st.caption("Find jobs that match your CV")

# ----- Top: Search + CV upload -----
with st.container():
    col_search, col_cv = st.columns([2, 1])
    with col_search:
        search_query = st.text_input(
            "Search jobs",
            placeholder="e.g. Data Scientist, Python, Remote",
            label_visibility="collapsed",
        )
    with col_cv:
        uploaded_cv = st.file_uploader(
            "Upload CV",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
        )
        if uploaded_cv is not None:
            st.caption(f"📄 {uploaded_cv.name}")

# ----- Cover letter: run on "Generate CV" click -----
idx = st.session_state.generate_for_job_index
if idx is not None:
    cv_text = get_cv_text(uploaded_cv) if uploaded_cv else None
    if not cv_text or not cv_text.strip():
        st.error(NO_CV_ERROR)
    else:
        job = JOBS[idx]
        with st.spinner("Generating cover letter…"):
            try:
                letter = create_cover_letter(cv_text, job["description"])
                st.session_state.last_cover_letter = letter
                st.session_state.last_cover_letter_job_index = idx
            except Exception as e:
                st.error(f"Error generating cover letter: {e}")
    st.session_state.generate_for_job_index = None

if st.session_state.last_cover_letter:
    with st.expander("📄 Generated cover letter", expanded=True):
        st.write(st.session_state.last_cover_letter)
    st.caption(f"Generated for job index {st.session_state.last_cover_letter_job_index}.")

# ----- Main: Scrollable jobs list (paginated, with Generate CV button) -----
st.subheader("Jobs")
st.divider()

num_pages = max(1, (len(JOBS) + JOBS_PER_PAGE - 1) // JOBS_PER_PAGE)
st.session_state.page = max(0, min(st.session_state.page, num_pages - 1))
page = st.session_state.page
start = page * JOBS_PER_PAGE
end = min(start + JOBS_PER_PAGE, len(JOBS))
page_jobs = [(start + i, JOBS[start + i]) for i in range(end - start)]

def set_generate_job(job_index: int):
    st.session_state.generate_for_job_index = job_index

with st.container():
    for job_index, job in page_jobs:
        col_content, col_btn = st.columns([4, 1])
        with col_content:
            st.markdown(f"**{job.get('company', '—')}**")
            st.caption(job.get("title", "—"))
            st.markdown(job_preview_text(job["description"]))
        with col_btn:
            st.button(
                "Generate CV",
                key=f"gen_cv_{job_index}",
                on_click=set_generate_job,
                args=(job_index,),
            )
        st.divider()

prev_col, info_col, next_col = st.columns([1, 2, 1])
with prev_col:
    if st.button("← Previous", disabled=(page <= 0)):
        st.session_state.page = page - 1
        st.rerun()
with info_col:
    st.caption(f"Page {page + 1} of {num_pages} · Jobs {start + 1}–{end} of {len(JOBS)}")
with next_col:
    if st.button("Next →", disabled=(page >= num_pages - 1)):
        st.session_state.page = page + 1
        st.rerun()

st.caption(f"Showing {len(JOBS)} jobs. Upload a CV and click « Generate CV » on a job to create a cover letter.")
