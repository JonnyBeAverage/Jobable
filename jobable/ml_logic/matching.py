import ast
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import preprocess_text
from sentence_transformers import SentenceTransformer, util

def compute_tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(vectors[0:1], vectors[1:2])
    return float(score[0][0])

def count_matching_keywords_no_repeats(resume_text, job_text):
    """
    job_text (str), resume_text (str): (order of inputs doesnt matter)
    counts number of overlapping unique data keywords

    """

    bag_of_resume = preprocess_text(resume_text) ##gets list of key words
    bag_of_job = preprocess_text(job_text) ##gets list of key words

    matching = set()

    for key_word in bag_of_job:
        if key_word in bag_of_resume:
            matching.add(key_word)

    return len(matching)


def encoder_scoring(resume_text, job_text, model=None):
    '''
    transformer model encodes both texts into a 1D tensor/vectors
    and finds consine similarity of the two tensors
    '''

    if not model:
        model = SentenceTransformer("all-MiniLM-L6-v2")


    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_embs = model.encode(job_text, convert_to_tensor=True)

    sim = util.cos_sim(resume_emb, job_embs)[0].cpu().numpy()

    return sim


def _parse_embedding_str(s):
    """Parse embeddings column from CSV (string repr of array or list) into 1D numpy array."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    if not s or s in ("[]", "nan"):
        return None
    try:
        # Try list literal (e.g. "[0.1, -0.2, ...]")
        out = np.array(ast.literal_eval(s), dtype=np.float32)
        return out
    except (ValueError, SyntaxError):
        pass
    try:
        # Fallback: space-separated numbers inside brackets
        s_clean = s.replace("[", "").replace("]", "").replace("\n", " ")
        parts = s_clean.split()
        out = np.array([float(x) for x in parts], dtype=np.float32)
        return out
    except Exception:
        return None


def rank_jobs_by_embedding_similarity(cv_text: str, embeddings_csv_path: Path):
    """
    Load job embeddings from embeddings_dataframe.csv (created with SentenceTransformer
    'all-MiniLM-L6-v2' on Job Description), encode cv_text with the same model, compute
    cosine similarity, and return job indices ordered by highest similarity first.
    """
    path = Path(embeddings_csv_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "embeddings" not in df.columns:
        return None
    # Parse embeddings column into a matrix (n_jobs x dim)
    embs = []
    for _, row in df.iterrows():
        e = _parse_embedding_str(row.get("embeddings"))
        if e is None:
            return None
        embs.append(e)
    job_embeddings = np.stack(embs).astype(np.float32)
    # Same model as in test.ipynb
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cv_embedding = model.encode(cv_text, convert_to_numpy=True).astype(np.float32)
    cv_embedding = cv_embedding.reshape(1, -1)
    # Cosine similarity: (1 x dim) @ (dim x n) -> (1 x n_jobs)
    sim = cosine_similarity(cv_embedding, job_embeddings)[0]
    # Order by highest similarity first
    order = np.argsort(sim)[::-1]
    return [(float(sim[i]), int(i)) for i in order]


def keywords_missing(job_text, resume_text=None, kw_job=None, kw_resume=None):
    """
    finds keywords in job but not in resume (can also pass in kw_resume for efficiency)
    """
    if not kw_resume and not resume_text:
        return
    if not kw_job and not job_text:
        return 


    if not kw_resume:
        kw_resume = preprocess_text(resume_text)
    if not kw_job:
        kw_job = preprocess_text(job_text)


    return set(w for w in kw_job if w not in kw_resume)


# model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = model.encode(job_df['Job Description'])
# def embed_to_column(row):
#     row_indx = row.name
#     row['embeddings'] = embeddings[row_indx]
#     return row
    

    