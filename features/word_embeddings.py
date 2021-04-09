from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

def word_embeddings_feature(X_train: pd.Series):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    word_embeddings_train = tokenizer.texts_to_sequences(X_train)
    sent_len = lambda lst: len(lst)
    max_len = max(map(sent_len, word_embeddings_train))
    word_embeddings_train = pad_sequences(word_embeddings_train, maxlen=max_len)

    return word_embeddings_train