
import csv
from os import remove
from re import T, compile, escape, split

import numpy as np
from numpy.core.numeric import True_
import pandas as pd
import scipy as sp
from pycontractions import Contractions
from sklearn.metrics import f1_score, make_scorer, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from imblearn.metrics import macro_averaged_mean_absolute_error
import string
import spacy
from autocorrect import Speller

from features.bow import bow_feature
from features.ngram import ngram_feature
from features.singlish import singlish_feature
from features.bert_embeddings import bert_embeddings_feature
from preprocessing.correct_spelling import correct_spelling_preprocessing
from preprocessing.expand_contraction import expand_contraction_preprocessing
from preprocessing.expand_short_form_words import expand_short_form_preprocessing
from preprocessing.lemmatization import lemmatization_preprocessing
from preprocessing.preprocessingFunctions import convert_to_lowercase, stopwords_removal
from preprocessing.remove_digit import remove_digit_preprocessing
from preprocessing.remove_link import remove_link_preprocessing
from preprocessing.remove_newline import remove_newline_preprocessing
from preprocessing.remove_non_english import remove_non_english_preprocessing
from models.nn import nn
from models.logistic_regression import logistic_regression

preprocessing_not_done: bool = False
feature_extraction: bool = False # refers to feature extraction before splitting of data (i.e. does not include bow/tfidf)
bow: bool = False # set specifically for bow
tfidf: bool = False # set specifically for tfidf
model_training: bool = True # False
num_classes: int = 5 # 5 levels of negativity

# models - set only one of it to true
naive_bayes: bool = False # True # False
logistic: bool = False
neural_network: bool = True

if preprocessing_not_done:
    cont: Contractions = Contractions('../GoogleNews-vectors-negative300.bin.gz')
    with open('preprocessing/slang.txt', 'r') as myCSVfile:
        short_form_dict:dict = dict([pair for pair in csv.reader(myCSVfile, delimiter="=")])
    regex = compile('[%s]' % escape(string.punctuation+"“”‘’"))
    nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])
    spell = Speller(lang='en')
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
    expand_contraction: bool = flags[1]
    remove_punctuation: bool = flags[2]
    remove_stopwords: bool = flags[3]
    remove_non_english: bool = flags[4]
    remove_digit: bool = flags[5]
    correct_spelling: bool = flags[6]
    replace_short_form_slang: bool = flags[7]
    lemmatization: bool = flags[8]
    
    sentence = remove_link_preprocessing(sentence)
    sentence = remove_newline_preprocessing(sentence)

    if lowercased:
        sentence = convert_to_lowercase(sentence)

    if expand_contraction:
        sentence = expand_contraction_preprocessing(sentence, cont)
    
    if remove_punctuation:
        tokens:list = [t for word in sentence.split() for t in split(regex, word)]
        if remove_stopwords:
            tokens = stopwords_removal(tokens)
        sentence = ' '.join(tokens)
    elif remove_stopwords:
        sentence = ' '.join(stopwords_removal(sentence.split()))
        
    if remove_non_english:
        sentence = remove_non_english_preprocessing(sentence)

    if remove_digit:
        sentence = remove_digit_preprocessing(sentence)

    if correct_spelling:
        sentence = correct_spelling_preprocessing(sentence, spell)

    if replace_short_form_slang:
        sentence = expand_short_form_preprocessing(sentence, short_form_dict)
        if remove_stopwords:
            sentence = ' '.join(stopwords_removal(sentence.split()))
        if lowercased:
            sentence = convert_to_lowercase(sentence)

    if lemmatization:
        sentence = lemmatization_preprocessing(sentence, nlp)

    return sentence.strip()

def feature_engineering(data: pd.DataFrame):
    '''
    Flags to be written here
    n_gram_feature: bool = False
    '''
    singlish: bool = True
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

    return features

def feature_engineering2(X_train: pd.DataFrame, X_validation: pd.DataFrame, X_test: pd.DataFrame):
    X_train_feature: pd.DataFrame = pd.DataFrame()
    X_validation_feature: pd.DataFrame = pd.DataFrame()
    X_test_feature: pd.DataFrame = pd.DataFrame()

    if bow:
        print("bow")
        X_train_feature, X_validation_feature, X_test_feature = bow_feature(X_train, X_validation, X_test)
       
    if tfidf:
        print("tfidf")
        X_train_tfidf, X_validation_tfidf, X_test_tfidf = ngram_feature(X_train, X_validation, X_test)
        X_train_feature = sp.sparse.hstack((X_train_feature, X_train_tfidf))
        X_validation_feature = sp.sparse.hstack((X_validation_feature, X_validation_tfidf))
        X_test_feature = sp.sparse.hstack((X_test_feature, X_test_tfidf))

    return X_train_feature, X_validation_feature, X_test_feature
   

def train_model(model, train_features: pd.DataFrame, validation_features: pd.DataFrame, train_label: pd.Series, validation_label:pd.Series):
    '''
    Flags to be written at the top as global variables
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
        model = logistic_regression(train_features, train_label, f1_scorer)
    elif neural_network:
        print("neural network")
        model = nn(train_features, train_label, validation_features, validation_label, num_classes)
    return model

def predict(model, X_test_features: pd.DataFrame):
    if neural_network:
      return pd.Series(np.argmax(model.predict(X_test_features), axis=1))
    return pd.Series(model.predict(X_test_features))

def scoring(train_or_test, y_label, y_pred):
    row: dict = {}
    # Use f1-macro as the metric
    score: float = f1_score(y_label, y_pred, average='macro')
    print('macro f1 score on {} = {}'.format(train_or_test, score))
    row[f'{train_or_test}_f1_score'] = score
    
    score = macro_averaged_mean_absolute_error(y_label, y_pred)
    print('macro MAE score on {} = {}'.format(train_or_test, score))
    row[f'{train_or_test}_MAE'] = score

    score = balanced_accuracy_score(y_label,y_pred)
    print('balance accuracy score on {} = {}'.format(train_or_test, score))
    row[f'{train_or_test}_acc'] = score
    return row

def main():
    '''
    If loading processsed data v6_remove_punctuation_remove_non_english_correct_spelling_replace_short_form_slang_remove_stopwords.csv, set preprocessing_not_done to False
    If loading feature csv, set feature_extraction to False and change the loaded feature file name
    If training model, set model_training to True
    '''
    old_train: pd.DataFrame = pd.read_csv('data/v6_remove_punctuation_remove_non_english_correct_spelling_replace_short_form_slang_remove_stopwords.csv')
    
    old_train = old_train.dropna(axis = 0, subset=['text'], inplace=False)
    label: pd.Series = old_train['label']
    train: pd.DataFrame = deepcopy(old_train)
    print("loaded data")

    if preprocessing_not_done:
        flag_names: list = ["lowercased","expand_contraction","remove_punctuation","remove_stopwords",
            "remove_non_english","remove_digit","correct_spelling","replace_short_form_slang","lemmatization"]
        cont.load_models()
        print("loaded contraction model")
        scores: pd.DataFrame = pd.DataFrame(
            pd.read_csv('preprocessing/scores.csv'), 
            columns=flag_names+["train_f1_score","train_MAE","train_acc","test_f1_score","test_MAE","test_acc"]
        )
        flags = [False,False,True,True,True,False,True,True,True]
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
        # change flags in feature_engineering to decide on the feature to extract
        train_features: pd.DataFrame = feature_engineering(train)
        # remember to change the file name for the feature extracted
        train_features.to_csv('features/singlish_negativity.csv', index=False) # take note of overwriting ~ 

        # add 'text' col to train_features for bow/tfidf in feature extraction part II
        train_features = pd.concat([train_features, train['text']], axis=1)
        print("finished feature extraction part 1")
    else: 
        print("no feature extraction part 1")
        train_features = pd.DataFrame(train['text'])
        ## uncomment to include features extracted previously
        ## -- uncomment to include Singlish Negativity  --
        # train_features = pd.concat([train_features.reset_index(drop=True), pd.read_csv('features/singlish_negativity.csv').reset_index(drop=True)], axis=1)

        ## -- uncomment to include question_mark  --
        # train_features = pd.concat([train_features, pd.read_csv('features/question_mark_count.csv')], axis=1)

        ## -- uncomment to include reply  --
        # train_features = pd.concat([train_features, pd.read_csv('features/is_not_reply.csv')], axis=1)

        ## -- uncomment to include sad_face  --
        # train_features = pd.concat([train_features, pd.read_csv('features/is_sad_face.csv')], axis=1)
        
        ## -- uncomment to include bert embeddings --
        ## use 'pt' for original BERT, 'nw' for NUSWhispers fine-tuned BERT, or 
        ## 'ge_nw' for BERT fine-tuned on both GoEmotions and NUSWhispers.
     
        # train_features = pd.concat([train_features.reset_index(drop=True), bert_embeddings_feature('pt').reset_index(drop=True)], axis=1)
        # train_features = pd.concat([train_features.reset_index(drop=True), bert_embeddings_feature('nw').reset_index(drop=True)], axis=1)
        # train_features = pd.concat([train_features.reset_index(drop=True), bert_embeddings_feature('ge_nw').reset_index(drop=True)], axis=1)

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
        model = None
        model = train_model(model, train_features, validation_features, train_label, validation_label)
        
        # test your model
        print("start prediction")
        y_pred: pd.Series = predict(model, train_features)
        train_metrics = scoring('train', train_label, y_pred)
        print(train_metrics.values())

        print("\n")

        y_pred: pd.Series = predict(model, test_features)
        test_metrics = scoring('test', test_label, y_pred)
        print(test_metrics.values())
        
        ## uncomment when testing and storing the perfomance of different pre-processing methods
        # row: dict = dict(zip(flag_names, flags))
        # scores = scores.append({**row,**train_metrics,**test_metrics}, ignore_index=True)
        # scores.to_csv('preprocessing/scores1.csv', index=False)

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()