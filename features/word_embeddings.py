from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

def word_embeddings_feature(X_train: pd.Series, X_validation: pd.Series, X_test: pd.Series):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    tokenizer.fit_on_texts(X_validation)
    tokenizer.fit_on_texts(X_test)

    word_embeddings_train = tokenizer.texts_to_sequences(X_train)
    word_embeddings_validation = tokenizer.texts_to_sequences(X_validation)
    word_embeddings_test = tokenizer.texts_to_sequences(X_test)
    sent_len = lambda lst: len(lst)
    max_len = max(max(map(sent_len, word_embeddings_train)), 
        max(map(sent_len, word_embeddings_validation)), 
        max(map(sent_len, word_embeddings_test)))
    word_embeddings_train = pad_sequences(word_embeddings_train, maxlen=max_len)
    word_embeddings_validation = pad_sequences(word_embeddings_validation, maxlen=max_len)
    word_embeddings_test = pad_sequences(word_embeddings_test, maxlen=max_len)

    return word_embeddings_train, word_embeddings_validation, word_embeddings_test