from load_data import load_data
from preprocess import add_bag_of_words_column
from frequency import get_wordcounts
from matching import compute_similarity
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

if __name__ == "__main__":
    run_pipeline()
