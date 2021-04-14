
import csv
from os import remove
from re import T

import numpy as np
import pandas as pd
import scipy as sp
# from pycontractions import Contractions
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from copy import deepcopy

from features.bow import bow_feature
from features.ngram import ngram_feature
from features.word_embeddings import word_embeddings_feature
from features.singlish import singlish_feature
#from features.bert_embeddings import bert_embeddings_feature
from preprocessing.correct_spelling import correct_spelling_preprocessing
# from preprocessing.expand_contraction import expand_contraction_preprocessing
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

preprocessing_not_done: bool = True
feature_extraction: bool = False # refers to feature extraction before splitting of data (i.e. does not include bow/tfidf)
bow: bool = False # set specifically for bow
tfidf: bool = False # set specifically for tfidf
model_training: bool = False # False
num_classes: int = 5 # 5 levels of negativity

# models - set only one of it to true
naive_bayes: bool = False # True # False
logistic: bool = False
neural_network: bool = True

if preprocessing_not_done:
    # cont: Contractions = Contractions('/Users/yuwen/Desktop/NUS/Year5Sem2/CS4248/Project/GoogleNews-vectors-negative300.bin.gz')
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
    
    if remove_stopwords:
        sentence = ' '.join(stopwords_removal(sentence.split()))
    
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
    singlish: bool = False
    word_embedding: bool = False
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

    elif word_embedding:
        X_train_word_embedding = word_embeddings_feature(data['text'])
        features = sp.sparse.hstack((X_train_word_embedding, features))

    return features

def feature_engineering2(X_train: pd.DataFrame, X_validation: pd.DataFrame, X_test: pd.DataFrame):
    X_train_feature: pd.DataFrame = pd.DataFrame()
    X_validation_feature: pd.DataFrame = pd.DataFrame()
    X_test_feature: pd.DataFrame = pd.DataFrame()

    # bow and tfidf booleans moved to the top of the source code now
    # bow: bool = False
    # tfidf: bool = False

    if bow:
        print("bow")
        X_train_feature, X_validation_feature, X_test_feature = bow_feature(X_train, X_validation, X_test)
       
    elif tfidf:
        print("tfidf")
        X_train_feature, X_validation_feature, X_test_feature = ngram_feature(X_train, X_validation, X_test)

    return X_train_feature, X_validation_feature, X_test_feature
   

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

    if naive_bayes:
        print("naive bayes")
        model = MultinomialNB().fit(train_features, train_label)
    elif logistic:
        print("logistic regression")
        model = LogisticRegression().fit(train_features, train_label) # simple testing (checking that embedding code works)
        # model = logistic_regression(train_features, train_label, f1_scorer)
    elif neural_network:
        print("neural network")
        model = nn(train_features, train_label, validation_features, validation_label, num_classes)
    return model

def predict(model: MultinomialNB, X_test_features: pd.DataFrame):
    if neural_network:
      return pd.Series(np.argmax(model.predict(X_test_features), axis=1))
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
    #old_train: pd.DataFrame = pd.read_csv('data/v6_remove_punctuation_remove_non_english_correct_spelling_replace_short_form_slang.csv')
    old_train: pd.DataFrame = pd.read_csv('data/v6_.csv')
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
        flags = [False,True,False,True,False,True,False,True,True,False,False]
        # pre-processing
        print("start preprocessing")
        train['text'] = old_train['text'].copy()
        train['text'] = train['text'].apply(preprocessing, args=(flags,))
        filename = "data/v6_" + '_'.join(filter(lambda a: flags[flag_names.index(a)], flag_names)) + ".csv"
        train.to_csv(filename, index=False)
        print("stored preprocessing")
        
    # feature extraction before splitting of data
    if feature_extraction:
        print("start feature extraction part 1 (before splitting of data)")
        train_features: pd.DataFrame = feature_engineering(train)
        train_features.to_csv('features/some_name.csv', index=False) # take note of overwriting ~ 

        # add 'text' col to train_features for bow/tfidf in feature extraction part II
        train_features = pd.concat([train_features, train['text']], axis=1)
        print("finished feature extraction part 1")
    else: 
        train_features = train[['text']]
        # train_features: pd.DataFrame = pd.read_csv('features/singlish_negativity.csv')
        
        # uncomment and repeat the following row to input multiple feature files and concatenate the features into one dataframe
        # train_features = pd.concat([train_features, pd.read_csv('features/<your feature name>.csv')], axis=1)

        ## -- uncomment to include Singlish Negativity  --
        train_features = pd.concat([train_features, pd.read_csv('features/singlish_negativity.csv')], axis=1)

        ## -- uncomment to include bert embeddings --
        ## use 'pt' for original BERT, 'nw' for NUSWhispers fine-tuned BERT, or 
        ## 'genw' for BERT fine-tuned on both GoEmotions and NUSWhispers.
        train_features = pd.concat([train_features, bert_embeddings_feature('nw')], axis=1)

        # print(train_features.head())
        # print(train_features.info())
        print("loaded features (Part 1)")

    # split data into train, validation, test set
    train_features, validation_features, train_label, validation_label = train_test_split(train_features, label, test_size=0.2, random_state=10)
    test_features, validation_features, test_label, validation_label = train_test_split(validation_features, validation_label, test_size=0.5, random_state=10)

    # feature extraction part II for bow/tfidf
    if bow or tfidf:
        print("start feature extraction part 2 (after splitting of data)")
        train_vector, validation_vector, test_vector = feature_engineering2(train_features, validation_features, test_features)

        train_features.drop(['text'], axis=1, inplace=True)
        validation_features.drop(['text'], axis=1, inplace=True)
        test_features.drop(['text'], axis=1, inplace=True)

        train_features = sp.sparse.hstack((train_vector, train_features))
        validation_features = sp.sparse.hstack((validation_vector, validation_features))
        test_features = sp.sparse.hstack((test_vector, test_features))
        print("finished feature extraction part 2")
    else:
        train_features.drop(['text'], axis=1, inplace=True)
        validation_features.drop(['text'], axis=1, inplace=True)
        test_features.drop(['text'], axis=1, inplace=True)

    if neural_network and isinstance(train_features, sp.sparse.coo.coo_matrix):
        train_features = pd.DataFrame(train_features.todense())
        validation_features = pd.DataFrame(validation_features.todense())
        test_features = pd.DataFrame(test_features.todense())
    
    if model_training:
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

        # row: dict = dict(zip(flag_names, flags))
        # row['training_score'] = score
        # row['test_score'] = score2
        # scores = scores.append(row, ignore_index=True)
        # scores.to_csv('preprocessing/scores.csv', index=False)

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
