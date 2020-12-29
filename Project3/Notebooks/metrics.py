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


def F1_score(y_pred, y, threshold=0.5):
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


def log_metrics(epoch, metrics):
    """ Function to print the metrics of the last batch during the training of a model """
    print(f'Epoch: {epoch}')
    print('\t\tAverage Training Loss: {:.2f}, Average Validation Loss: {:.2f}\n\t\t'
          'Average Training Accuracy: {:.2f}, '
          'Average Validation Accuracy: {:.2f}'.format(metrics['train_losses'][-1],
                                                       metrics['val_losses'][-1],
                                                       metrics['train_accuracies'][-1],
                                                       metrics['val_accuracies'][-1]))
    print('\t\tAverage Training F1-score: {:.2f}, '
          'Average Validation F1-score: {:.2f}\n'.format(metrics['train_f1'][-1],
                                                         metrics['val_f1'][-1]))
