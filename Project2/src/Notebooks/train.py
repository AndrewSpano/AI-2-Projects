import torch

from train_utils import *
from metrics import *
from plots import *


def train(model, X_train, y_train, X_val, y_val, loss, optimizer, hyperparameters,
         restore_best_weights=True, target_metric="val_loss", target_file="./model.pth", verbose=True):
    """ Function used to train the model """

    # get the hyperparameters as variables
    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]

    # split the data in batches
    X_batches = torch.split(X_train, batch_size)
    y_batches = torch.split(y_train, batch_size)

    # create lists for the metrics in a dictionary
    metrics = {
        "train_losses": [],
        "val_losses": [],
        "train_accuracies": [],
        "val_accuracies": [],
        "train_f1": [],
        "val_f1": []
    }

    # execute a specified number of epochs
    for epoch in range(epochs):

        # log information if specified
        if verbose:
            print("\tepoch: {}".format(epoch + 1))

        # get all the batches
        for X_batch, y_batch in zip(X_batches, y_batches):

            # set the gradients back to 0
            optimizer.zero_grad()

            # compute the loss for the current batch in the current epoch
            batch_loss = model.forward_step(X_batch, y_batch, loss)

            # perform backpropagation to compute the gradients
            batch_loss.backward()

            # perform a step with the optimizer
            optimizer.step()

        # update the different metrics
        compute_metrics(model, X_train, y_train, X_val, y_val, loss, metrics)

        # log information if specified
        if verbose:
            log_metrics(metrics)

        # check if we have good weights, and if we should save them
        if restore_best_weights:
            # check if we have an improvement
            if metric_improved(metrics, model.best_metric, target_metric):
                # keep the value and save the weights
                model.best_metric = metrics[target_metric][-1]
                torch.save(model.state_dict(), target_file)

    # return the metrics dictionary
    return metrics



def k_fold_train(model, X_train, y_train, loss, optimizer, hyperparameters, k=10, early_stopping=True,
                 restore_best_weights=True, target_metric="val_loss", target_file="./model.pth", plot_curves=True, verbose=True):
    """ Function used to implement k-fold cross validation for the training procedure """

    # check if best weights should be restored, then initialize the best metric
    if restore_best_weights:
        best_metric_init(model, target_metric)

    # define a list to store the history of each fold
    histories = []
  
    # perform training procedure k times, each with different datasets
    for iteration in range(k):

        # log some information if specified
        if verbose:
            print("\nFold: {}".format(iteration + 1))

        # get the data of the current iteration
        _X_train, _y_train, X_val, y_val = get_kth_fold_data(X_train, y_train, k, iteration)

        # train the model with the assigned data
        history = train(model, _X_train, _y_train, X_val, y_val, loss, optimizer, hyperparameters,
                        restore_best_weights, target_metric, target_file, verbose=verbose)

        # append it to the histories list
        histories.append(history)

        # check for early stopping condition
        if (iteration > k // 3) and (early_stopping is True) and (average_validation_error_has_not_decreased_enough(histories)):
            break

    # if plotting curves is specified, do it
    if plot_curves:
        plot_metrics(histories)

    # if best weights should be restored, restore them
    if restore_best_weights:
        model.load_state_dict(torch.load(target_file))

    # return the histories
    return histories