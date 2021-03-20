from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd


def ngram_feature(X_train: pd.Series):
    tfidf_ngram_vectorizer = TfidfVectorizer(
        max_features=100, ngram_range=(2, 3))
    tfidf_ngram_vectors: tuple = tfidf_ngram_vectorizer.fit_transform(X_train)
    tfidf_ngram: pd.DataFrame = pd.DataFrame(data=tfidf_ngram_vectors.todense(
    ), columns=tfidf_ngram_vectorizer.get_feature_names())
    return tfidf_ngram
