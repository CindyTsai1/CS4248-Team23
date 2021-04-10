from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def bow_feature(X_train: pd.Series, X_validation: pd.Series, X_test: pd.Series):
    vectorizer = CountVectorizer(max_features=1000)
    X_train_bow = vectorizer.fit_transform(X_train['text'])
    X_validation_bow = vectorizer.transform(X_validation['text'])
    X_test_bow = vectorizer.transform(X_test['text'])

    # X_train_tfidf = pd.DataFrame(data=X_train_tfidf.todense(), columns=vectorizer.get_feature_names())
    # X_validation_tfidf = pd.DataFrame(data=X_validation_tfidf.todense(), columns=vectorizer.get_feature_names())
    # X_test_tfidf = pd.DataFrame(data=X_test_tfidf.todense(), columns=vectorizer.get_feature_names())

    return X_train_bow, X_validation_bow, X_test_bow