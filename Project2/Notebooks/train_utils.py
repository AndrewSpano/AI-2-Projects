import torch
import statistics
import numpy as np


def average_validation_error_has_not_decreased_enough(histories, tolerance=0.01):
    """ Function that returns true if the average validation error of the
        penultimate iteration has not dropped by a significant amount """
  
    # get the corresponding histories
    last_history = histories[-1]
    penultimate_history = histories[-2]

    # get the corresponding metrics (validation losses)
    last_val_losses = last_history["val_losses"]
    penultimate_val_losses = penultimate_history["val_losses"]

    # return the value of the condition which is what we want
    return statistics.mean(last_val_losses) > statistics.mean(penultimate_val_losses) - tolerance



def get_kth_fold_data(X_train, y_train, k, iteration):
    """ Function used to extract the datasets for the k-th iteration of the k-fold cross validation algorithm """
    # determine how many examples each fold gets
    portion = X_train.shape[0] // k

    # distinguish the cases and act accordingly
    if iteration == 0:
        X_val = X_train[0 : portion]
        y_val = y_train[0 : portion]

        _X_train = X_train[portion:]
        _y_train = y_train[portion:]

    elif iteration < k - 1:
        X_val = X_train[iteration * portion : (iteration + 1) * portion]
        y_val = y_train[iteration * portion : (iteration + 1) * portion]

        left_X = X_train[:iteration * portion]
        right_X = X_train[(iteration + 1) * portion:]
        _X_train = torch.cat((left_X, right_X), 0)

        left_y = y_train[:iteration * portion]
        right_y = y_train[(iteration + 1) * portion:]
        _y_train = torch.cat((left_y, right_y), 0)

    else:
        X_val = X_train[-portion:]
        y_val = y_train[-portion:]

        _X_train = X_train[:-portion]
        _y_train = y_train[:-portion]


    # return the target datasets
    return _X_train, _y_train, X_val, y_val