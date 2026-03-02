def get_wordcounts(series):
    word_counts = {}
    for sentence in series:
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts
