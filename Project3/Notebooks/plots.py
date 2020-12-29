import torch
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt

from metrics import *


def plot_metrics(train_metric, val_metric, train_metric_name, val_metric_name, yax, title):
    """
    :param list train_metric:      The list containing the training metrics.
    :param list val_metric:        The list containing the validation metrics.
    :param str train_metric_name:  The name of the training metric that is plotted.
    :param str val_metric_name:    The name of the validation metric that is plotted.
    :param str yax:                The string describing the values of the y-axis.
    :param str title:              The title of the plot.

    :return:  None.
    """
    plt.figure(figsize=(25, 13))
    plt.plot(train_metric, color='royalblue', label=train_metric_name)
    plt.plot(val_metric, color='maroon', linestyle="--", label=val_metric_name, marker='o')
    plt.yticks(np.arange(0.0, 1.01, 0.1))
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(yax)
    plt.title(title)
    plt.show()



def plot_roc_curve(model, test_iterator, threshold_step=0.0001):
    """ plot the ROC curve of the given models on a specific dataset """
    # define the thresholds that will be used to compute the ROC curve
    thresholds = np.arange(threshold_step, 1.0, threshold_step)

    # define the list with the values of (sensitivity and 1 - specificity)
    recalls = []
    fall_outs = []

    # get the information of the test set
    X_test, lengths = None, None
    y_test = None

    for full_testset in test_iterator:
        X_test, lengths = full_testset.text
        y_test = full_testset.label

    # make the prediction
    y_test_pred = model(X_test, lengths)

    # compute the metrics for every threshold
    for threshold in thresholds:

        # get the roc metrics
        recall, fall_out = roc_metrics(y_test_pred, y_test, threshold=threshold)

        # append to the corresponding lists
        recalls.append(recall)
        fall_outs.append(fall_out)

    # configure the size of the ROC curve plots
    plt.rcParams["figure.figsize"] = [15, 10]
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14

    # plot the ROC curve
    plt.plot(fall_outs, recalls, color="dodgeblue", label="RNN Classifier")

    # plot y = x for comparison
    x = np.arange(0, 1.01, 0.1)
    plt.plot(x, x, color="brown", linestyle="--", label=r"$y\;=\;x$")

    # add legend, labels and title
    plt.legend()
    plt.xlabel(r"1 - Specificity", fontsize=20)
    plt.ylabel(r"Sensitivity", fontsize=20)
    plt.title("ROC curve\n", fontsize=25)
    plt.show()
