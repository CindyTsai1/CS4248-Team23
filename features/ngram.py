from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def ngram_feature(X_train: pd.Series, X_validation: pd.Series, X_test: pd.Series):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 6))
    X_train_tfidf = vectorizer.fit_transform(X_train['text'])
    X_validation_tfidf = vectorizer.transform(X_validation['text'])
    X_test_tfidf = vectorizer.transform(X_test['text'])
    
    return X_train_tfidf, X_validation_tfidf, X_test_tfidf
