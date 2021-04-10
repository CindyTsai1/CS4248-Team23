from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd

_max_iter: int = 200
def logistic_regression(X_train_features: pd.DataFrame, y_train: pd.Series, f1_scorer):
    param_grid = {'penalty': ['l2', 'l1', 'none'], 'C': [1, 0.5, 0.1]}
    model = GridSearchCV(
        LogisticRegression(max_iter=_max_iter,
                           solver='sag', multi_class='multinomial'),
        param_grid,
        scoring=f1_scorer,
        cv=5,
        verbose=10
    ).fit(X_train_features, y_train)
    print(model.best_params_)
    print(model.best_estimator_)
    return model
