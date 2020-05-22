import numpy as np


def argsort(y_pred, y_true, descent=True):
    if descent:
        sorted_index = np.argsort(-1 * y_pred)
    else:
        sorted_index = np.argsort(y_pred)

    return y_pred[sorted_index], y_true[sorted_index]



def ROC(X, y, model):



    return 0
