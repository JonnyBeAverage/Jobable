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

    return len(key_word)


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
