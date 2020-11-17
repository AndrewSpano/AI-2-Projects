import torch
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt

from metrics import *

def plot_metrics(histories):
    """ given a dictionary with info about specific metrics, plot them with matplotlib """
    # define some lists that will be the concatenation of every metric for all the folds
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1s = []
    val_f1s = []

    # for each fold
    for history in histories:

        # append the metrics of that fold to the above lists
        train_losses.extend(history["train_losses"])
        val_losses.extend(history["val_losses"])
        train_accuracies.extend(history["train_accuracies"])
        val_accuracies.extend(history["val_accuracies"])
        train_f1s.extend(history["train_f1"])
        val_f1s.extend(history["val_f1"])

    # determine the number of total epochs: epochs per fold * folds
    epochs = len(train_losses)

    # define the figure and itds size
    fig = plt.figure(figsize=(40, 40))

    # create a gridspec object
    gs1 = gridspec.GridSpec(2, 2)
    # configure the space between the plots
    gs1.update(wspace=0.135)
    # axis for losses
    ax1 = plt.subplot(gs1[0, 0])
    # axis for accuracies
    ax2 = plt.subplot(gs1[0, 1])
    # axis for F1 scores
    ax3 = plt.subplot(gs1[1, :])

    # make sure that epoch ticks are integers
    plt.xticks(range(1, epochs))
    xtick_frequency = max(epochs // 10, 1)
    ax1.set_xticks(np.arange(0, epochs, xtick_frequency))
    ax2.set_xticks(np.arange(0, epochs, xtick_frequency))
    ax3.set_xticks(np.arange(0, epochs, xtick_frequency))
    # make tick fontsize a bit bigger than default
    ax1.tick_params(axis="both", which="major", labelsize=21)
    ax2.tick_params(axis="both", which="major", labelsize=21)
    ax3.tick_params(axis="both", which="major", labelsize=23)
    # add title for both plots
    plt.suptitle("\n\nTraining/Validation Losses/Accuracies/F1-Scores for every epoch in every fold during the Training procedure", fontsize=42)

    # plot training loss
    train_loss_label = "Training Loss"
    ax1.plot(train_losses, color="b", label=train_loss_label)
    # plot validation loss
    val_loss_label = "Validation Loss"
    ax1.plot(val_losses, color="r", label=val_loss_label)
    # add side labels and legend
    ax1.set_xlabel("epochs", fontsize=25)
    ax1.set_ylabel(r"Loss $J(w)$", fontsize=25)
    ax1.legend(prop={"size": 24})

    # plot training accuracy
    train_acc_label = "Training Accuracy"
    ax2.plot(train_accuracies, color="b", label=train_acc_label)
    # plot validation accuracy
    val_acc_label = "Validation Accuracy"
    ax2.plot(val_accuracies, color="r", label=val_acc_label)
    # add side labels and legend
    ax2.set_xlabel("epochs", fontsize=25)
    ax2.set_ylabel("Accuracy", fontsize=25)
    ax2.legend(prop={"size": 24})

    # plot the Training F1 scores
    train_f1_label = "Training F1 score"
    ax3.plot(train_f1s, color="b", label=train_f1_label)
    # plot the Validation F1 scores
    val_f1_label = "Validation F1 score"
    ax3.plot(val_f1s, color="r", label=val_f1_label)
    # add side labels and legend
    ax3.set_xlabel("epochs", fontsize=35)
    ax3.set_ylabel("F1 Score", fontsize=35)
    ax3.legend(prop={"size": 28})

    # show the plots
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