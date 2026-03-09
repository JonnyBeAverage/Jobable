from sklearn.feature_extraction.text import CountVectorizer
from load_data import load_data
import pandas as pd
from jobable.ml_logic.data_keywords import list_of_keywords

### import dataframe
job_df = load_data('../data/job_title_des.csv')
job_df

def skill_frequ(dataframe):

    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=5)
    X = vectorizer.fit_transform(job_df['Job Description'])

    ### convert sprase matrix to array to convert it into a dataframe
    X = X.toarray()
    X = pd.DataFrame(X, columns=vectorizer.get_feature_names_out())

    ### create list of keywords from set of keywords in order to iterate over elements
    list_keys = list(list_of_keywords)


    result = dict()

    for kw in list_keys:
        if kw in X.columns:
            result[kw] = X[kw].sum()

    list_count = []
    list_count = result.items()
    list_count = sorted(list_count, key = lambda item: item[1], reverse=True)

    ### returns 30 most frequently demanded skills/attributes
    return list_count[:30]
