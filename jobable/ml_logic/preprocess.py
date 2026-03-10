import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from data_keywords import DATA_KEYWORDS

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text):

    text = text.strip().lower()

    # Keep + and # (for c++, c#)
    text = re.sub(r"[^\w\s+#']", " ", text)
    words = text.split()

    matches = []

    # Find longest keyword length dynamically
    max_len = max(len(k.split()) for k in DATA_KEYWORDS)

    # Check phrases from 1 word up to max_len
    for n in range(1, max_len + 1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            if phrase in DATA_KEYWORDS:
                matches.append(phrase)

    return set(matches)

def add_bag_of_words_column(df, column_name):
    df["bag_of_words"] = df[column_name].apply(preprocess_text)
    return df

def get_wordcounts(series):
    word_counts = {}
    for sentence in series:
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts
