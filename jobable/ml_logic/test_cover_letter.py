from jobable.ml_logic.load_data import load_data
from jobable.ml_logic.cover_letter import create_cover_letter

def run_cover_letter_test():

    #Load data and get 1 cv and 1 job descriptiom
    jd_df = load_data("/Users/jonny/code/JonnyBeAverage/Jobable/data/Jobs.csv")
    resume_df = load_data("/Users/jonny/code/JonnyBeAverage/Jobable/data/Resume.csv")
    print("Data loaded:", jd_df.shape, resume_df.shape)

    single_jd_description = jd_df.iloc[0]['description']
    single_resume_description = resume_df.iloc[0]['Resume_str']
    print("Extracted Resume and JD details")

    cover_letter = create_cover_letter(single_resume_description, single_jd_description)
    print("Created cover letter")
    print(cover_letter)

if __name__ == "__main__":
    run_cover_letter_test()
