"""
Jobable — Streamlit frontend shell.
Top: search + CV upload. Main: scrollable list of jobs with title and description preview.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from jobable.ml_logic.cover_letter import create_cover_letter
from jobable.ml_logic.matching import compute_tfidf_similarity

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
JOBS_PER_PAGE = 20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def job_preview_text(description: str, max_words: int = 12) -> str:
    words = description.strip().split()
    if len(words) <= max_words:
        return description
    return " ".join(words[:max_words]) + "…"


def get_cv_text(uploaded_file) -> str | None:
    """Extract text from uploaded CV. Supports .txt; PDF/DOCX require pypdf and python-docx."""
    if uploaded_file is None:
        return None
    import io
    raw = uploaded_file.getvalue()
    name = (uploaded_file.name or "").lower()
    if name.endswith(".txt"):
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return None
    if name.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            return None
    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return None
    return None


def cover_letter_to_pdf(letter_text: str) -> bytes:
    """Build a PDF from cover letter text. Returns PDF bytes."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.set_auto_page_break(auto=True, margin=15)
    # Ensure we only pass bytes that the font can display (Helvetica is Latin-1)
    safe_text = letter_text.encode("latin-1", errors="replace").decode("latin-1")
    for line in safe_text.splitlines():
        pdf.multi_cell(0, 8, line or " ")
    return bytes(pdf.output())


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
            search_with_cv_clicked = st.button("Search with CV", key="search_with_cv")
        else:
            search_with_cv_clicked = False

if search_with_cv_clicked and uploaded_cv is not None:
    cv_text = get_cv_text(uploaded_cv)
    if cv_text and cv_text.strip():
        with st.spinner("Ranking jobs by CV match…"):
            scored = []
            for i, job in enumerate(JOBS):
                score = compute_tfidf_similarity(job["description"], cv_text)
                scored.append((score, i))
            scored.sort(key=lambda x: x[0], reverse=True)
            st.session_state["jobs_display_order"] = [i for _, i in scored]
            st.session_state["jobs_page"] = 0
        st.rerun()
    else:
        st.warning("Could not read CV text. Try uploading a .txt file.")

# ----- Main: Scrollable jobs list -----
st.subheader("Jobs")
if st.session_state.get("jobs_display_order") is not None:
    st.caption("Sorted by CV match (best first)")
st.divider()


def _safe_filename(s: str, max_len: int = 50) -> str:
    return "".join(c for c in s if c.isalnum() or c in " -_")[:max_len].strip() or "Job"


if "jobs_page" not in st.session_state:
    st.session_state["jobs_page"] = 0

# Use CV-sorted order if available, else default order
display_order = st.session_state.get("jobs_display_order")
if display_order is None or len(display_order) != len(JOBS):
    display_order = list(range(len(JOBS)))

total_jobs = len(display_order)
total_pages = max(1, (total_jobs + JOBS_PER_PAGE - 1) // JOBS_PER_PAGE)
current_page = max(0, min(st.session_state["jobs_page"], total_pages - 1))
if current_page != st.session_state["jobs_page"]:
    st.session_state["jobs_page"] = current_page

page_start = current_page * JOBS_PER_PAGE
page_end = min(page_start + JOBS_PER_PAGE, total_jobs)

for idx in range(page_start, page_end):
    i = display_order[idx]  # global index in JOBS
    job = JOBS[i]
    with st.container():
        st.markdown(f"**{job.get('company', '—')}** — *{job['title']}*")
        st.caption(job_preview_text(job["description"]))
        if st.button("Generate Cover Letter", key=f"jobable-cl-{i}"):
            if uploaded_cv is None:
                st.warning("Upload a CV first to generate a cover letter.")
            else:
                cv_text = get_cv_text(uploaded_cv)
                if not (cv_text and cv_text.strip()):
                    st.warning("Could not read CV text. Try uploading a .txt file.")
                else:
                    jd_text = job["description"]
                    with st.spinner("Generating cover letter…"):
                        try:
                            letter = create_cover_letter(cv_text, jd_text)
                            pdf_bytes = cover_letter_to_pdf(letter)
                            st.session_state["cover_letter_pdf_bytes"] = pdf_bytes
                            st.session_state["cover_letter_job_ix"] = i
                        except Exception as e:
                            st.error(f"Error generating cover letter: {e}")
        # Show download button in this card when the cover letter was generated for this job
        if (
            "cover_letter_pdf_bytes" in st.session_state
            and st.session_state.get("cover_letter_job_ix") == i
        ):
            job_label = _safe_filename(job["title"])
            st.download_button(
                label="Download cover letter",
                data=st.session_state["cover_letter_pdf_bytes"],
                file_name=f"cover_letter_{job_label}.pdf",
                mime="application/pdf",
                key=f"download_cover_letter_{i}",
            )
        st.divider()

# Pagination controls
col_prev, col_info, col_next = st.columns([1, 2, 1])
with col_prev:
    prev_clicked = st.button("← Previous", key="jobs_prev", disabled=(current_page == 0))
with col_info:
    st.caption(f"Page **{current_page + 1}** of **{total_pages}** — showing {page_start + 1}–{page_end} of {total_jobs} jobs")
with col_next:
    next_clicked = st.button("Next →", key="jobs_next", disabled=(current_page >= total_pages - 1))

if prev_clicked:
    st.session_state["jobs_page"] = max(0, current_page - 1)
    st.rerun()
if next_clicked:
    st.session_state["jobs_page"] = min(total_pages - 1, current_page + 1)
    st.rerun()

st.caption("Connect search and CV matching to filter and rank.")
