import torch
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt

from metrics import *


def plot_metrics(checkpoints, train_metric, val_metric, train_metric_name, val_metric_name, yax,
                 title):
    """
    :param list checkpoints:       A list contaning in which steps the validation metrics where
                                      calculated.
    :param list train_metric:      The list containing the training metrics.
    :param list val_metric:        The list containing the validation metrics.
    :param str train_metric_name:  The name of the training metric that is plotted.
    :param str val_metric_name:    The name of the validation metric that is plotted.
    :param str yax:                The string describing the values of the y-axis.
    :param str title:              The title of the plot.

    :return:  None.
    """
    plt.figure(figsize=(20, 10))
    plt.plot(checkpoints, train_metric, color='royalblue', label=train_metric_name)
    plt.plot(checkpoints, val_metric, color='maroon', linestyle="--", label=val_metric_name, marker='o')
    plt.yticks(np.arange(0.0, 1.01, 0.1))
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel(yax)
    plt.title(title)
    plt.show()



def plot_roc_curves(models, modelnames, colors, Xs, ys, threshold_step=0.0001):
    """ plot the ROC curve of the given models on a specific dataset """
    # define the thresholds that will be used to compute the ROC curve
    thresholds = np.arange(threshold_step, 1.0, threshold_step)

    # define the list with the values of (sensitivity and 1 - specificity)
    recalls = {model: [] for model in models}
    fall_outs = {model: [] for model in models}

    # make the prediction
    y_pred = {model: model(Xs[model]) for model in models}

    # compute the metrics for every threshold
    for threshold in thresholds:

        # for each model
        for model in models:

            # get the roc metrics
            recall, fall_out = roc_metrics(y_pred[model], ys[model], threshold=threshold)

            # append to the corresponding lists
            recalls[model].append(recall)
            fall_outs[model].append(fall_out)

    # configure the size of the ROC curve plots
    plt.rcParams["figure.figsize"] = [15, 10]
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14

    # for every model
    for model in models:
        # plot its ROC curve
        color = colors[model] if colors is not None else "royalblue"
        plt.plot(fall_outs[model], recalls[model], color=color, label=modelnames[model])

    # plot y = x for comparison
    x = np.arange(0, 1.01, 0.1)
    plt.plot(x, x, color="brown", linestyle="--", label=r"$y\;=\;x$")

    # add legend, labels and title
    plt.legend()
    plt.xlabel(r"1 - Specificity", fontsize=20)
    plt.ylabel(r"Sensitivity", fontsize=20)
    plt.title("ROC curve\n", fontsize=25)
    plt.show()
