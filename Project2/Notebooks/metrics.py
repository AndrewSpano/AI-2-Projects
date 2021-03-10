import torch


def accuracy(y_pred, y, threshold=0.5):
    """ Method to compute the accuracy of a prediction of a model """
    # get the actual 0-1 labels
    labels = torch.where(y_pred > threshold, 1, 0)
    # get the number of correct predictions
    correct = (labels == y).sum().item()
    # compute the accuracy and return it
    m = y.shape[0]
    accuracy = correct / m
    return accuracy


def roc_metrics(y_pred, y, threshold=0.5):
    """ Function used to compute useful metrics """
    # get the actual 0-1 labels
    labels = torch.where(y_pred > threshold, 1.0, 0.0)

    # get the number of positives and the number of negatives
    positives = (y == 1.0).sum().item()
    negatives = (y == 0.0).sum().item()

    # get the true positives and true negatives
    true_positives = ((y == 1.0) & (labels == y)).sum().item()
    true_negatives = ((y == 0.0) & (labels == y)).sum().item()

    # get the false positives and false negatives
    false_positives = ((labels == 1.0) & (y == 0.0)).sum().item()
    false_negatives = ((labels == 0.0) & (y == 1.0)).sum().item()

    # compute the ROC metrics
    recall = true_positives / positives if positives != 0 else 0
    fall_out = false_positives / negatives if negatives != 0 else 0

    # return the wanted metrics
    return recall, fall_out


def f1_score(y_pred, y, threshold=0.5):
    """ Function used to compute the F1 score of a prediction of a model """
    # get the actual 0-1 labels
    labels = torch.where(y_pred > threshold, 1.0, 0.0)

    # get the number of positives and the number of negatives
    positives = (y == 1.0).sum().item()
    negatives = (y == 0.0).sum().item()

    # get the true positives and true negatives
    true_positives = ((y == 1.0) & (labels == y)).sum().item()
    true_negatives = ((y == 0.0) & (labels == y)).sum().item()

    # get the false positives and false negatives
    false_positives = ((labels == 1.0) & (y == 0.0)).sum().item()
    false_negatives = ((labels == 0.0) & (y == 1.0)).sum().item()

    # compute the F1 score metrics
    precision = true_positives / (true_positives + false_positives) if true_positives != 0 else 0
    recall = true_positives / positives if positives != 0 else 0

    # compute the F1 score
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0

    # return the score
    return f1


def compute_metrics(model, X_train, y_train, X_val, y_val, loss, metrics):
    """ Function used to compute all the metrics needed for a function """

    # make training and validation predictions
    y_train_pred = model(X_train)
    y_val_pred = model(X_val)

    # compute the losses
    train_loss = loss(y_train_pred, y_train).item()
    val_loss = loss(y_val_pred, y_val).item()

    # compute the accuracies
    train_acc = accuracy(y_train_pred, y_train)
    val_acc = accuracy(y_val_pred, y_val)

    # compute the F1 scores
    train_f1 = f1_score(y_train_pred, y_train)
    val_f1 = f1_score(y_val_pred, y_val)

    # append everything to the lists
    metrics["train_losses"].append(train_loss)
    metrics["val_losses"].append(val_loss)
    metrics["train_accuracies"].append(train_acc)
    metrics["val_accuracies"].append(val_acc)
    metrics["train_f1"].append(train_f1)
    metrics["val_f1"].append(val_f1)


def log_metrics(metrics):
    """ Function to print the metrics of the last iteration so far during the training of a model """
    print("\t\tTraining Loss: {:.2f}, Validation Loss: {:.2f}, Training Accuracy: {:.2f}, Validation Accuracy: "
          "{:.2f}".format(metrics["train_losses"][-1], metrics["val_losses"][-1],
                          metrics["train_accuracies"][-1], metrics["val_accuracies"][-1]))
    print("\t\tTraining F1-score: {:.2f}, Validation F1-score: {:.2f}".format(metrics["train_f1"][-1], metrics["val_f1"][-1]))


def metric_improved(metrics, best_metric, metric_type):
    """ Function that returns True if the current metric is better than the best so far; Else False """
    if metric_type == "val_f1" or metric_type == "val_accuracies":
        return metrics[metric_type][-1] >= best_metric
    else:
        return metrics[metric_type][-1] <= best_metric


def best_metric_init(model, metric):
    """ Function to initialize the best metrics of a model """
    if metric == "val_losses":
        model.best_metric = float("inf")
    else:
        model.best_metric = 0.0
            