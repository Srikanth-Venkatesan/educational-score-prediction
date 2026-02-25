import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    return np.sqrt(root_mean_squared_error(y_true, y_pred))


def mad(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
