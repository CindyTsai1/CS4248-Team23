
import csv
from models.logistic_regression import logistic_regression
from os import remove
from re import T

import pandas as pd
import spicy as sp
from pycontractions import Contractions
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from copy import deepcopy

from features.ngram import ngram_feature
from features.word_embeddings import word_embeddings_feature
from features.singlish import singlish_feature
from preprocessing.correct_spelling import correct_spelling_preprocessing
from preprocessing.expand_contraction import expand_contraction_preprocessing
from preprocessing.expand_short_form_words import expand_short_form_preprocessing
from preprocessing.lemmatization import lemmatization_preprocessing
from preprocessing.preprocessingFunctions import (convert_to_lowercase,
                                                  punctuation_removal,
                                                  stemming, stopwords_removal)
from preprocessing.remove_digit import remove_digit_preprocessing
from preprocessing.remove_link import remove_link_preprocessing
from preprocessing.remove_newline import remove_newline_preprocessing
from preprocessing.remove_non_english import remove_non_english_preprocessing
from models.logistic_regression import logistic_regression
from models.nn import nn

preprocessing_not_done: bool = False
feature_extraction: bool = True
model_training: bool = False
if preprocessing_not_done:
    cont: Contractions = Contractions('/Users/yuwen/Desktop/NUS/Year5Sem2/CS4248/Project/GoogleNews-vectors-negative300.bin.gz')
    with open('/Users/yuwen/Desktop/NUS/Year5Sem2/CS4248/Project/CS4248-Team23/preprocessing/slang.txt', 'r') as myCSVfile:
        short_form_dict:dict = dict([pair for pair in csv.reader(myCSVfile, delimiter="=")])
def preprocessing(sentence: str, flags: list):
    ''' 
    takes in a row of data, preprocess it, return the processed row.

    Format:
    if remove_punctuations:
        sentence = sub(r' +', '  ', sentence)
        sentence = sub(rf'^[{punctuation}]+| [{punctuation}]+|[{punctuation}]+ |[{punctuations}]+$', ' ', sentence)
    
    Can write the functions in a separate file, import and execute here / or just write here since we didn't split this job
    '''
    lowercased: bool = flags[0]
    remove_punctuation: bool = flags[1]
    remove_newline: bool = flags[2]
    remove_non_english: bool = flags[3]
    remove_digit: bool = flags[4]
    correct_spelling: bool = flags[5]
    expand_contraction: bool = flags[6]
    replace_short_form_slang: bool = flags[7]
    remove_stopwords: bool = flags[8]
    lemmatization: bool = flags[9]
    stemming_flag: bool = flags[10]
    
    sentence = remove_link_preprocessing(sentence)

    if lowercased:
        sentence = convert_to_lowercase(sentence)

    if expand_contraction:
        sentence = expand_contraction_preprocessing(sentence, cont)
    
    if remove_punctuation:
        sentence = ' '.join(punctuation_removal(sentence.split()))

    if remove_newline:
        sentence = remove_newline_preprocessing(sentence)

    if remove_non_english:
        sentence = remove_non_english_preprocessing(sentence)

    if remove_digit:
        sentence = remove_digit_preprocessing(sentence)

    if correct_spelling:
        sentence = correct_spelling_preprocessing(sentence)
    
    if replace_short_form_slang:
        sentence = expand_short_form_preprocessing(sentence, short_form_dict)

    if remove_stopwords:
        sentence = ' '.join(stopwords_removal(sentence.split()))

    if lemmatization:
        sentence = lemmatization_preprocessing(sentence)

    if stemming_flag:
        sentence = ' '.join(stemming(sentence.split()))

    return sentence

def feature_engineering(data: pd.DataFrame):
    '''
    Flags to be written here
    n_gram_feature: bool = False
    '''
    singlish: bool = True
    bow: bool = False
    ''' 
    Format:
    from features.ngram import ngram_feature
    if n_gram_feature:
        features = pd.concat([features, ngram(X_train)], axis=1)
    
    Write your functions in separate python files in folder features and import them here to use, eg in features/ngram.py
    '''
    features: pd.DataFrame = pd.DataFrame()
    if singlish:
        features = pd.concat([features, singlish_feature(data['text']).rename('singlish_negativity')], axis=1)
    if bow:
        print("bow")
        vectorizer = CountVectorizer(max_features=500)
        word_count_vector = vectorizer.fit_transform(data['text'])
        features = pd.concat([features, pd.DataFrame(data=word_count_vector.todense(), columns=vectorizer.get_feature_names())], axis=1)
    return features

def feature_engineering2(X_train: pd.DataFrame):
    X_train_feature = feature_engineering(X_train)

    n_gram_feature: bool = False
    word_embedding: bool = False

    if n_gram_feature:
        X_train_tfidf = ngram_feature(X_train['text'])
        X_train_feature = sp.sparse.hstack((X_train_tfidf, X_train_feature))
    
    if word_embedding:
        X_train_word_embedding = word_embeddings_feature(X_train['text'])
        X_train_feature = sp.sparse.hstack((X_train_word_embedding, X_train_feature))

    return X_train_feature
   

def train_model(model, train_features: pd.DataFrame, validation_features: pd.DataFrame, train_label: pd.Series, validation_label:pd.Series):
    '''
    Flags to be written here
    n_gram_model: bool = False
    '''
    ''' 
    Format:
    from models.ngram import ngram_model
    if n_gram_model:
        model = ngram_model(model, X_train_features, y_train)
    
    Write your functions in separate python files in folder models and import them here to use
    '''
    f1_scorer = make_scorer(f1_score, average='macro')
    naive_bayes: bool = False
    logistic: bool = False
    neural_network: bool = False

    if naive_bayes:
        print("naive bayes")
        model = MultinomialNB().fit(train_features, train_label)
    elif logistic:
        print("logistic regression")
        model = logistic_regression(train_features, train_label, f1_scorer)
    elif neural_network:
        print("neural network")
        model = nn(train_features, train_label, validation_features, validation_label)
    return model

def predict(model: MultinomialNB, X_test_features: pd.DataFrame):
    return pd.Series(model.predict(X_test_features))

def generate_result(test: pd.DataFrame, y_pred: pd.Series, filename: str):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.to_csv(filename, index=False)

def main():
    '''
    If loading processsed data v6_remove_punctuation_remove_non_english_correct_spelling_replace_short_form_slang.csv, set preprocessing_not_done to False
    If loading feature csv, set feature_extraction to False and change the loaded feature file name
    If training model, set model_training to True
    '''
    old_train: pd.DataFrame = pd.read_csv('data/v6_remove_punctuation_remove_non_english_correct_spelling_replace_short_form_slang.csv')
    old_train = old_train.dropna(axis = 0, subset=['text'], inplace=False)
    label: pd.Series = old_train['label']
    train: pd.DataFrame = deepcopy(old_train)
    print("loaded data")
    if preprocessing_not_done:
        flag_names: list = ["lowercased","remove_punctuation","remove_newline","remove_non_english","remove_digit","correct_spelling","expand_contraction",
            "replace_short_form_slang","remove_stopwords","lemmatization","stemming"]
        #cont.load_models()
        print("loaded contraction model")
        scores: pd.DataFrame = pd.DataFrame(pd.read_csv('preprocessing/scores.csv'), columns=flag_names+["training_score","test_score"])
        flags = [False,True,False,True,False,True,False,False,True,True,False]
        # pre-processing
        print("start preprocessing")
        train['text'] = old_train['text'].copy()
        train['text'] = train['text'].apply(preprocessing, args=(flags,))
        filename = "Project/CS4248-Team23/data/v6_" + '_'.join(filter(lambda a: flags[flag_names.index(a)], flag_names)) + ".csv"
        train.to_csv(filename, index=False)
        print("stored preprocessing")
        
    # features
    if feature_extraction:
        print("start feature extraction")
        train_features: pd.DataFrame = feature_engineering2(train)
        train_features.to_csv('features/singlish_negativity.csv', index=False)
        print("finish features")
    else: 
        train_features: pd.DataFrame = pd.read_csv('features/singlish_negativity.csv')
        # uncomment and repeat the following row to input multiple feature files and concatenate the features into one dataframe
        # train_features = pd.concat([train_features, pd.read_csv('features/<your feature name>.csv')], axis=1)
        print("loaded features")

    if model_training:
        # split data into train, validation, test set
        train_features, validation_features, train_label, validation_label = train_test_split(train_features, label, test_size=0.2, random_state=10)
        test_features, validation_features, test_label, validation_label = train_test_split(validation_features, validation_label, test_size=0.5, random_state=10)
        
        # The following was used when reloading the model to further train
        # model = load_model('my_model')
        # GoEmotions pre-trained model can be imported here
        model = None

        model = train_model(model, train_features, validation_features, train_label, validation_label)
        # test your model
        print("start prediction")
        y_pred: pd.Series = predict(model, train_features)

        # Use f1-macro as the metric
        score: float = f1_score(train_label, y_pred, average='macro')
        print('score on validation = {}'.format(score))

        # generate prediction on test data
        y_pred: pd.Series = predict(model, test_features)
        score2: float = f1_score(test_label, y_pred, average='macro')
        print('score on test = {}'.format(score2))

        row: dict = dict(zip(flag_names, flags))
        row['training_score'] = score
        row['test_score'] = score2
        scores = scores.append(row, ignore_index=True)
        scores.to_csv('preprocessing/scores.csv', index=False)

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
