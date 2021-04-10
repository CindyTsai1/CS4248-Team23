from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.layers import Dense, Embedding
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

def create_model(X_train_features: pd.DataFrame, y_train: pd.Series, initializer: str, regularization: str):
    opt = SGD(lr=0.001, momentum=0.9)
    model = Sequential()
    model.add(Dense(50, input_dim=X_train_features.shape[1], activation='relu',
                    kernel_regularizer=regularization, kernel_initializer=initializer))
    model.add(Dense(10, activation='relu',
                    kernel_regularizer=regularization, kernel_initializer=initializer))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
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

def nn(X_train_features: pd.DataFrame, y_train: pd.Series, X_validate_features: pd.DataFrame, y_validate: pd.Series):
    model = create_model(X_train_features, y_train, 'random_normal', 'l2')
    history = model.fit(
        np.asarray(X_train_features), 
        np.asarray(y_train), 
        epochs=300, 
        batch_size=X_train_features.shape[0]//100, 
        validation_data=(np.asarray(X_validate_features), np.asarray(y_validate)))
    plot(history)
    return model