
import matplotlib.pyplot as pyplot
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from pycontractions import Contractions
from features.num_like import num_like_feature
from features.singlish import singlish_feature
from preprocessing.remove_digit import remove_digit_preprocessing
from preprocessing.expand_contraction import expand_contraction_preprocessing

cont: Contractions = Contractions('/Users/yuwen/Desktop/NUS/Year5Sem2/CS4248/Project/CS4248-Team23/preprocessing/GoogleNews-vectors-negative300.bin.gz')
def preprocessing(sentence: str):
    ''' 
    takes in a row of data, preprocess it, return the processed row.

    Format:
    if remove_punctuations:
        sentence = sub(r' +', '  ', sentence)
        sentence = sub(rf'^[{punctuation}]+| [{punctuation}]+|[{punctuation}]+ |[{punctuations}]+$', ' ', sentence)
    
    Can write the functions in a separate file, import and execute here / or just write here since we didn't split this job
    '''
    remove_digit: bool = False
    expand_contraction: bool = False
    if remove_digit:
        sentence = remove_digit_preprocessing(sentence)
    if expand_contraction:
        sentence = expand_contraction_preprocessing(sentence, cont)
    return sentence

def feature_engineering(data: pd.DataFrame):
    '''
    Flags to be written here
    n_gram_feature: bool = False
    '''
    num_like: bool = False
    singlish: bool = False
    ''' 
    Format:
    from features.ngram import ngram_feature
    if n_gram_feature:
        features = pd.concat([features, ngram(X_train)], axis=1)
    
    Write your functions in separate python files in folder features and import them here to use, eg in features/ngram.py
    '''
    features: pd.DataFrame = pd.DataFrame()
    if num_like:
        features = pd.concat([features, num_like_feature(data)], axis=1)
    if singlish:
        features = pd.concat([features, singlish_feature(data['text'])], axis=1)
    return features

def train_model(model, train_features: pd.DataFrame, validation_features: pd.DataFrame):
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
    return model

def plot(history):
    x = range(1, len(history.history['accuracy']) + 1)
    pyplot.style.use('ggplot')
    pyplot.figure(figsize=(12, 5))
    pyplot.subplot(1, 2, 1)
    pyplot.plot(x, history.history['accuracy'], 'b', label='Training acc')
    pyplot.plot(x, history.history['validation_accuracy'], 'r', label='Validation acc')
    pyplot.title('Training and validation accuracy')
    pyplot.legend()
    pyplot.subplot(1, 2, 2)
    pyplot.plot(x, history.history['loss'], 'b', label='Training loss')
    pyplot.plot(x, history.history['validation_loss'], 'r', label='Validation loss')
    pyplot.title('Training and validation loss')
    pyplot.legend()
    pyplot.savefig('history.png')

def predict(model, X_test_features: pd.DataFrame):
    return pd.Series(model.predict(X_test_features))

def generate_result(test: pd.DataFrame, y_pred: pd.Series, filename: str):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    train: pd.DataFrame = pd.read_csv('v4.csv')
    train.drop(train['text'].str.len == 0)
    
    # pre-processing
    cont.load_models()
    train['text'] = train['text'].apply(preprocessing)

    # split data into train, validation, test set
    train, validation = train_test_split(train, test_size=0.2, random_state=10)
    test, validation = train_test_split(validation, test_size=0.5, random_state=10)
    
    # features
    train_features: pd.DataFrame = feature_engineering(train)
    validation_features: pd.DataFrame = feature_engineering(validation)
    test_features: pd.DataFrame = feature_engineering(test)
    
    # The following was used when reloading the model to further train
    # model = load_model('my_model')
    # GoEmotions pre-trained model can be imported here
    model = None

    model = train_model(model, train_features, validation_features)
    # test your model
    y_pred: pd.Series = predict(model, train_features)

    # Use f1-macro as the metric
    # score: float = f1_score(y_train, y_pred, average='macro')
    # print('score on validation = {}'.format(score))

    # generate prediction on test data
    y_pred: pd.Series = predict(model, test_features)
    generate_result(test, y_pred, "result.csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()