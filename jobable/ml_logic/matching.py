from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_text
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(vectors[0:1], vectors[1:2])
    return float(score[0][0])



def count_matching_keywords_no_repeats(resume_text, job_text):
    """
    job (Series), resume (Series): counts number of overlapping data keywords
    count_score(job, resume): counts repeats
    count_score(resume, job): doesn’t counts repeats
    """

    bag_of_resume = preprocess_text(resume_text) ##gets list of key words
    bag_of_job = preprocess_text(job_text) ##gets list of key words

    matching = set()

    for key_word in bag_of_job:
        if key_word in bag_of_resume:
            matching.add(key_word)

    return len(key_word)
