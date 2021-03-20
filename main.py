
import matplotlib.pyplot as pyplot
import pandas as pd
from sklearn.metrics import f1_score

def preprocessing(sentence: str):
    ''' 
    takes in a row of data, preprocess it, return the processed row.

    Format:
    if remove_punctuations:
        sentence = sub(r' +', '  ', sentence)
        sentence = sub(rf'^[{punctuation}]+| [{punctuation}]+|[{punctuation}]+ |[{punctuations}]+$', ' ', sentence)
    
    Can write the functions in a separate file, import and execute here / or just write here since we didn't split this job
    '''
    return sentence

def feature_engineering(X_train: pd.Series):
    '''
    Flags to be written here
    n_gram_feature: bool = False
    '''
    ''' 
    Format:
    from features.ngram import ngram_feature
    if n_gram_feature:
        features = pd.concat([features, ngram(X_train)], axis=1)
    
    Write your functions in separate python files in folder features and import them here to use, eg in features/ngram.py
    '''
    features: pd.DataFrame = pd.DataFrame()
    return features

def train_model(model, X_train_features: pd.DataFrame, y_train: pd.Series):
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
    # split data into train, validation, test set
    train: pd.DataFrame = pd.read_csv('train.csv')
    test: pd.DataFrame = pd.read_csv('test.csv')
    
    X_train: pd.Series = train['Text']
    X_test: pd.Series = test['Text']
    # pre-processing
    X_train = X_train.apply(preprocessing)
    X_test = X_test.apply(preprocessing)
    # features
    features: pd.DataFrame = feature_engineering(X_train.append(X_test, ignore_index=True))
    X_train_features: pd.DataFrame = features[:X_train.size]
    X_test_features: pd.DataFrame = features[X_train.size:]
    
    y_train: pd.Series = train['Verdict']

    # The following was used when reloading the model to further train
    # model = load_model('my_model')
    # GoEmotions pre-trained model can be imported here
    model = None

    model = train_model(model, X_train_features, y_train)
    # test your model
    y_pred: pd.Series = predict(model, X_train_features)

    # Use f1-macro as the metric
    score: float = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    # generate prediction on test data
    y_pred: pd.Series = predict(model, X_test_features)
    generate_result(test, y_pred, "result.csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()