def recommend_skills(cv_skills, jd_skills, freq_dict):
    missing = set(jd_skills) - set(cv_skills)
    ranked = sorted(missing, key=lambda x: freq_dict.get(x, 0), reverse=True)
    return ranked[:3]
