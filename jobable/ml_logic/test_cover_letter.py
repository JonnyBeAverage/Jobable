from jobable.ml_logic.load_data import load_data
from jobable.ml_logic.cover_letter import create_cover_letter

def run_cover_letter_test():

    #Load data and get 1 cv and 1 job descriptiom
    jd_df = load_data("jobable/data/job_title_des.csv")
    resume_df = load_data("jobable/data/Resume.csv")
    print("Data loaded:", jd_df.shape, resume_df.shape)

    single_jd_description = jd_df.sample(1)['Job Description'].iloc[0]
    single_resume_description = resume_df.sample(1)['Resume_str'].iloc[0]
    print("Extracted Resume and JD details")

    cover_letter = create_cover_letter(single_resume_description, single_jd_description)
    print("Created cover letter")
    print(cover_letter)

if __name__ == "__main__":
    run_cover_letter_test()
