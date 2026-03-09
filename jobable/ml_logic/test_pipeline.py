from load_data import load_data
from preprocess import add_bag_of_words_column
from frequency import get_wordcounts
from matching import compute_tfidf_similarity, count_matching_keywords_no_repeats, encoder_scoring

import os
from recommendation import recommend_skills
import pandas as pd
from sentence_transformers import SentenceTransformer

def run_pipeline():

    # Load Data, for testing change the path to the data
    df = load_data("/Users/jonny/code/JonnyBeAverage/Jobable/data/Jobs.csv")
    print("Data loaded:", df.shape)

    # Preprocess, may need to change the name of the column from 'description'
    df = add_bag_of_words_column(df, "description")
    print("Token column added.")

    # Word Frequency
    freq_dict = get_wordcounts(df["bag_of_words"])
    print("Top 5 words:", list(freq_dict.items())[:5])

    # Test similarity
    sample_cv = "Python SQL data analysis"
    sample_jd = "Looking for Python and SQL experience"

    similarity_score = compute_tfidf_similarity(sample_cv, sample_jd)
    print("Similarity score:", similarity_score)

    # Test recommendation
    cv_skills = ["python"]
    jd_skills = ["python", "sql", "aws"]

    recommendations = recommend_skills(cv_skills, jd_skills, freq_dict)
    print("Recommended skills:", recommendations)



###
### REAL DATA TESTS
###

# load data CHANGE THIS TO YOUR LOCAL PATH FOR NOW
job_df = load_data('jobable/data/job_title_des.csv')
resume_df = load_data('jobable/data/resume.csv')

#select resume and job instance from df
test_job_instance = job_df.iloc[9, 2] #change first number to change resume instance (e.g [10,2] for the resume in index 10)

# test_resume_instance = resume_df.iloc[5,1] #change first number to change resume instance (e.g [10,1] for the resume in index 10)
test_resume_instance = resume_df[resume_df['Category']=='ENGINEERING'].iloc[5,1]


#use category=engineering to get resumes with datascience keywords

test_resume_id = resume_df[resume_df['Resume_str']==test_resume_instance].iloc[0,0]


def test_tf_idf(test_job=test_job_instance, test_resume=test_resume_instance):
    return compute_tfidf_similarity(test_job, test_resume)

def test_count_matching_keywords_no_repeats(test_job=test_job_instance, test_resume=test_resume_instance):
    return count_matching_keywords_no_repeats(test_job, test_resume)

def test_encoder_scoring(test_job=test_job_instance, test_resume=test_resume_instance, model=None):
    return encoder_scoring(test_job, test_resume, model)



###
### TEST ALL DOCUMENTS
### finds best scoring documents for a given resume
###

def test_all_scoring_functions(test_resume=test_resume_instance, job_df=job_df, save_csv=True):

    top10_df = pd.DataFrame()

    #TFIDF
    job_df['tfidf_score'] = job_df['Job Description'].apply(lambda x: test_tf_idf(x, test_resume))
    t = job_df.sort_values(by='tfidf_score', ascending=False).head(10) #higher is better

    top10_df['tfidf_job_index'] = t.index
    top10_df[['tfidf_job_descriptions', 'tfidf_scores']] = t[['Job Description','tfidf_score']].values
    print('M: finished TFIDF tests')


    #KEYWORD MATCHING
    job_df['matching_score'] = job_df['Job Description'].apply(lambda x: test_count_matching_keywords_no_repeats(x, test_resume))
    t = job_df.sort_values(by='matching_score', ascending=False).head(10)

    top10_df['keyword_matching_job_index'] = t.index
    top10_df[['keyword_matching_job_descriptions', 'keyword_matching_scores']] = t[['Job Description','matching_score']].values

    print('M: finished keyword matching tests')

    #ENCODER
    model = SentenceTransformer("all-MiniLM-L6-v2")
    job_df['encoder_scores'] = job_df['Job Description'].apply(lambda x: test_encoder_scoring(x, test_resume, model))
    t= job_df.sort_values(by='encoder_scores', ascending=False).head(10)

    top10_df['encoder_job_index'] = t.index
    top10_df[['encoder_job_descriptions', 'encoder_scores']] = t[['Job Description','encoder_scores']].values
    print('M: finished keyword encoder tests')

    if save_csv:
        os.makedirs('jobable/test_results', exist_ok=True)
        top10_df.to_csv(f'jobable/test_results/all_tests_resumeid={test_resume_id}.csv')

    return top10_df

if __name__ == "__main__":
    print('running')
    print(test_all_scoring_functions())
