import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from jobable.ml_logic.data_keywords import DATA_KEYWORDS

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text):

    text = text.strip().lower()
    text = ''.join(char for char in text if not char.isdigit())
    text = ''.join(char for char in text if char not in string.punctuation)

    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(word, pos="v") for word in tokens]
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]

    return [word for word in tokens if word in DATA_KEYWORDS]

def add_bag_of_words_column(df, column_name):
    df["bag_of_words"] = df[column_name].apply(preprocess_text)
    return df

def get_wordcounts(series):
    word_counts = {}
    for sentence in series:
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts
