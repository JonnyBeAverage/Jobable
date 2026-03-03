from load_data import load_data
from preprocess import add_bag_of_words_column
from frequency import get_wordcounts
from matching import compute_similarity, count_matching_keywords_no_repeats, encoder_scoring
from recommendation import recommend_skills

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

    similarity_score = compute_similarity(sample_cv, sample_jd)
    print("Similarity score:", similarity_score)

    # Test recommendation
    cv_skills = ["python"]
    jd_skills = ["python", "sql", "aws"]

    recommendations = recommend_skills(cv_skills, jd_skills, freq_dict)
    print("Recommended skills:", recommendations)



###
### REAL DATA TESTS
###

# load data
job_df = load_data('/Users/isaac/code/ishane0620/jobable/data/job_title_des.csv')
resume_df = load_data('/Users/isaac/code/ishane0620/jobable/data/Resume.csv')

#select resume and job instance from df
test_job_instance = job_df.iloc[9, 2] #change first number to change resume instance (e.g [10,2] for the resume in index 10)

test_resume_instance = resume_df.iloc[5,1] #change first number to change resume instance (e.g [10,1] for the resume in index 10)
# test_resume_instance = resume_df[resume_df['Category']=='ENGINEERING'].iloc[5,1]
#use category=engineering to get resumes with datascience keywords


def test_count_matching_keywords_no_repeats(test_job= test_job_instance,test_resume=test_resume_instance):
    return count_matching_keywords_no_repeats(test_job, test_resume)

def test_encoder_scoring(test_job= test_job_instance,test_resume=test_resume_instance):
    return encoder_scoring(test_job, test_resume)


if __name__ == "__main__":
    print('running')
    print(test_count_matching_keywords_no_repeats())
    print(test_encoder_scoring())
