"""
Jobable — Streamlit frontend shell.
Top: search + CV upload. Main: scrollable list of jobs with title and description preview.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def job_preview_text(description: str, max_words: int = 12) -> str:
    words = description.strip().split()
    if len(words) <= max_words:
        return description
    return " ".join(words[:max_words]) + "…"


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

# Placeholder for when we wire up: run search + CV matching here
# if uploaded_cv: ...
# if search_query: filter JOBS

# ----- Main: Scrollable jobs list -----
st.subheader("Jobs")
st.divider()

# Build HTML for job cards and render via components.html so it isn't escaped
job_cards_html = []
for job in JOBS:
    meta = job.get("company", "—")
    preview = job_preview_text(job["description"])
    # Escape for use inside HTML (in case job text contains < or &)
    title_esc = job["title"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    meta_esc = meta.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    preview_esc = preview.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    card = f"""
    <div class="jobable-job-card">
        <div class="jobable-job-title">{meta_esc}</div>
        <div class="jobable-job-meta">{title_esc}</div>
        <div class="jobable-job-desc">{preview_esc}</div>
    </div>
    """
    job_cards_html.append(card)

html_content = f"""
<!DOCTYPE html>
<html>
<head>
<style>
.jobable-job-list {{
    padding-top: 0;
    padding-right: 8px;
    font-family: inherit;
}}
.jobable-job-card {{
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid #0e1117;
}}
.jobable-job-title {{ font-weight: 600; font-size: 1.05rem; color: #0e1117; }}
.jobable-job-meta {{ font-size: 0.85rem; color: #666; margin-top: 0.25rem; }}
.jobable-job-desc {{ font-size: 0.9rem; color: #333; margin-top: 0.5rem; line-height: 1.4; }}
</style>
</head>
<body>
<div class="jobable-job-list">
{"".join(job_cards_html)}
</div>
</body>
</html>
"""
# Height is overridden by CSS so the iframe fills remaining viewport
st.components.v1.html(html_content, height=500, scrolling=True)

st.caption(f"Showing {len(JOBS)} jobs. Connect search and CV matching to filter and rank.")
