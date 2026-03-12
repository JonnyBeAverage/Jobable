"""
Jobable — Streamlit frontend shell.
Top: search + CV upload. Main: scrollable list of jobs with title and description preview.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

from jobable.ml_logic.cover_letter import create_cover_letter
from jobable.ml_logic.matching import compute_tfidf_similarity, keywords_missing, rank_jobs_by_embedding_similarity
from jobable.ml_logic.preprocess import preprocess_text


# ---------------------------------------------------------------------------
# Load encoder (search) and cover-letter models once (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_encoder_model():
    """SentenceTransformer for CV–job similarity; same as in test.ipynb."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_cover_letter_model():
    """(tokenizer, model) for cover letter generation; loaded once per session."""
    from jobable.ml_logic.model import tokenizer, model
    return (tokenizer, model)

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
    /* Keyword pills */
    .kw-wrap { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 6px; }
    .kw-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        background: linear-gradient(135deg, #e8eef4 0%, #dde5ed 100%);
        color: #334155;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }
    .kw-pill--missing {
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
        color: #7f1d1d;
        border: 1px solid rgba(185, 28, 28, 0.3);
    }
    /* Main title: centered, padding below */
    .jobable-main-title { text-align: center; margin-bottom: 1rem; }
    .jobable-main-title h1 { font-size: 2.25rem; font-weight: 600; margin: 0; }
    /* Jobs section: no padding below divider */
    .jobable-jobs-header { margin-bottom: 0 !important; padding-bottom: 0 !important; }
    .main .block-container > div:has(.jobable-jobs-header) { margin-bottom: 0.75rem !important; padding-bottom: 0 !important; }
    .jobable-jobs-header hr { margin-top: 0 !important; margin-bottom: 0 !important; }
    .jobable-jobs-header h2 { margin-bottom: 0 !important; margin-top: 0; font-size: 1.25rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load jobs from CSV (cached)
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "jobable" / "data" / "embeddings_dataframe.csv"

# Realistic company names (cycled by job index when CSV has no company column)
COMPANY_NAMES = [
    "Accenture", "Adobe", "Amazon", "Atlassian", "Bloomberg", "Capgemini", "Cisco",
    "Deloitte", "Dropbox", "Epam Systems", "Google", "IBM", "Infosys", "Intel",
    "JPMorgan Chase", "McKinsey & Company", "Microsoft", "Netflix", "Oracle",
    "Salesforce", "SAP", "Slack", "Spotify", "Stripe", "Tesla", "Twilio",
    "Uber", "VMware", "Wipro", "Zoom", "Airbnb", "Meta", "Shopify", "Square",
    "PayPal", "LinkedIn", "GitHub", "Notion", "Figma", "Canva", "HubSpot",
    "Snowflake", "Databricks", "MongoDB", "ServiceNow", "Workday", "Splunk",
    "Crowdstrike", "Okta", "Zscaler", "DocuSign", "RingCentral",
]


@st.cache_data
def load_jobs_csv(path: Path):
    """Load job list from embeddings_dataframe.csv. Columns: index, Job Title, Job Description."""
    df = pd.read_csv(path)
    # Normalize: use 'Job Title' and 'Job Description'
    title_col = "Job Title"
    desc_col = "Job Description"
    if title_col not in df.columns or desc_col not in df.columns:
        return []
    jobs = []
    for i, row in df.iterrows():
        if i == 0:
            continue  # Skip first entry
        title = row[title_col]
        desc = row[desc_col]
        if pd.isna(title):
            title = ""
        if pd.isna(desc):
            desc = ""
        # Flatten multiline description to single line for preview
        desc_flat = " ".join(str(desc).split())
        kw = sorted(preprocess_text(desc_flat))  # list of keyword phrases
        company = row["company"] if "company" in df.columns and pd.notna(row.get("company")) else COMPANY_NAMES[(i - 1) % len(COMPANY_NAMES)]
        jobs.append({
            "title": str(title).strip(),
            "company": str(company).strip(),
            "description": desc_flat,
            "kw": kw,
        })
    return jobs


JOBS = load_jobs_csv(DATA_PATH)
JOBS_PER_PAGE = 12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def job_preview_text(description: str, max_words: int = 20) -> str:
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
    #
    content_width = pdf.w - pdf.l_margin - pdf.r_margin
    # Ensure we only pass bytes that the font can display (Helvetica is Latin-1)
    safe_text = letter_text.encode("latin-1", errors="replace").decode("latin-1")
    for line in safe_text.splitlines():
        pdf.multi_cell(
            content_width, 8, line or " ",
            new_x="LMARGIN", new_y="NEXT",
        )
    return bytes(pdf.output())

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# Sync company page from URL (clickable company link uses ?company=<job_ix>)
if hasattr(st, "query_params") and "company" in st.query_params:
    try:
        st.session_state["company_page_ix"] = int(st.query_params["company"])
    except (ValueError, TypeError):
        pass

# ----- Company page (blank): show when navigated via company link -----
if st.session_state.get("company_page_ix") is not None:
    ix = st.session_state["company_page_ix"]
    if 0 <= ix < len(JOBS):
        job = JOBS[ix]
        company_name = job.get("company", "—")
        st.title("💼 Jobable")
        if st.button("← Back to jobs", key="back_from_company"):
            st.session_state["company_page_ix"] = None
            if hasattr(st, "query_params"):
                st.query_params.clear()
            st.rerun()
        st.subheader(company_name)
        desc = job.get("description", "")
        desc_esc = (
            str(desc)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        st.markdown(
            f'<div style="white-space: pre-wrap;">{desc_esc}</div>',
            unsafe_allow_html=True,
        )
        st.stop()
    else:
        st.session_state["company_page_ix"] = None

st.markdown(
    '<div class="jobable-main-title"><h1>Jobable</h1></div>',
    unsafe_allow_html=True,
)

# ----- Top: Search + CV upload -----
with st.container():
    # Centered box: search bar and CV controls share same width
    left_pad, center_box, right_pad = st.columns([1, 7, 1])
    with center_box:
        search_query = st.text_input(
            "Search jobs",
            placeholder="e.g. Data Scientist, Python, Remote",
            label_visibility="collapsed",
        )

        # Left = drag-and-drop uploader; middle = spacer; right = Search with CV button
        col_upload, col_spacer, col_btn = st.columns([3, 0.3, 1])
        with col_upload:
            uploaded_cv = st.file_uploader(
                "Upload CV",
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed",
            )
        with col_btn:
            if uploaded_cv is not None:
                search_with_cv_clicked = st.button("Search with CV", key="search_with_cv")
            else:
                search_with_cv_clicked = False

if search_with_cv_clicked and uploaded_cv is not None:
    cv_text = get_cv_text(uploaded_cv)
    if cv_text and cv_text.strip():
        with st.spinner("Ranking jobs by CV match…"):
            # Rank jobs by embedding similarity (same model as in test.ipynb: all-MiniLM-L6-v2)
            encoder_model = get_encoder_model()
            scored = rank_jobs_by_embedding_similarity(cv_text, DATA_PATH, model=encoder_model)
            if scored is not None:
                st.session_state["jobs_display_order"] = [i for _, i in scored]
                st.session_state["jobs_similarity_scores"] = {i: score for score, i in scored}
            else:
                # Fallback if embeddings CSV missing or no embeddings column: use TF-IDF sort
                scored = []
                for i, job in enumerate(JOBS):
                    score = compute_tfidf_similarity(job["description"], cv_text)
                    scored.append((score, i))
                scored.sort(key=lambda x: x[0], reverse=True)
                st.session_state["jobs_display_order"] = [i for _, i in scored]
                st.session_state["jobs_similarity_scores"] = {i: score for score, i in scored}
            # Previous sort (TF-IDF only): commented out in favor of embedding similarity
            # scored = []
            # for i, job in enumerate(JOBS):
            #     score = compute_tfidf_similarity(job["description"], cv_text)
            #     scored.append((score, i))
            # scored.sort(key=lambda x: x[0], reverse=True)
            # st.session_state["jobs_display_order"] = [i for _, i in scored]
            st.session_state["jobs_page"] = 0
            st.session_state["cv_search_clicked"] = True
            # Store resume keywords before rerun (file uploader often clears after rerun)
            st.session_state["resume_kw"] = sorted(preprocess_text(cv_text))
        st.rerun()
    else:
        st.warning("Could not read CV text. Try uploading a .txt file.")

# ----- Main: Scrollable jobs list -----
st.markdown('<div style="height: 3rem;"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="jobable-jobs-header"><h2>📝 Jobs Available</h2><hr /></div>',
    unsafe_allow_html=True,
)
st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)


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
        company = job.get("company", "—")
        company_esc = str(company).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        title_esc = str(job.get("title", "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        similarity_scores = st.session_state.get("jobs_similarity_scores") or {}
        match_html = f'<span style="font-weight: 600;">{similarity_scores[i] * 100:.2f}% match</span>' if i in similarity_scores else ""
        row_content = (
            f'<div style="display: flex; justify-content: space-between; align-items: center; gap: 1rem;">'
            f'<span><a href="?company={i}" style="font-weight: 700; color: inherit; text-decoration: none;">{company_esc}</a> — <em>{title_esc}</em></span>'
            f'<span>{match_html}</span>'
            f"</div>"
        )
        st.markdown(row_content, unsafe_allow_html=True)
        st.caption(job_preview_text(job["description"]))
        if st.button("🪄 Generate Cover Letter", key=f"jobable-cl-{i}"):
            if uploaded_cv is None:
                st.warning("Upload a CV first to generate a cover letter.")
            else:
                cv_text = get_cv_text(uploaded_cv)
                if not (cv_text and cv_text.strip()):
                    st.warning("Could not read CV text. Try uploading a .txt file.")
                else:
                    jd_text = job["description"]
                    with st.spinner("Generating Cover Letter…"):
                        try:
                            _tok, _mod = get_cover_letter_model()
                            letter = create_cover_letter(cv_text, jd_text, tokenizer=_tok, model=_mod)
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
                label="📥 Download Cover Letter",
                data=st.session_state["cover_letter_pdf_bytes"],
                file_name=f"cover_letter_{job_label}.pdf",
                mime="application/pdf",
                key=f"download_cover_letter_{i}",
            )
        # Keywords at bottom of card (from preprocess_text on description)
        # Red pill only for keywords missing from resume, and only after "Search with CV" was clicked
        kw = job.get("kw") or []
        if kw:
            resume_kw_set = set(st.session_state.get("resume_kw") or [])
            show_missing = bool(st.session_state.get("cv_search_clicked") and resume_kw_set)
            escaped = [str(k).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for k in kw]
            pill_class = lambda k: "kw-pill kw-pill--missing" if show_missing and k not in resume_kw_set else "kw-pill"
            pills = "".join(f'<span class="{pill_class(k)}">{p}</span>' for k, p in zip(kw, escaped))
            st.markdown(f'<div class="kw-wrap">{pills}</div>', unsafe_allow_html=True)
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
