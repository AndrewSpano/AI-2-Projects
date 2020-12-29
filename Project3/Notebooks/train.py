import torch

from metrics import *



def train(model, train_iterator, val_iterator, epochs, optimizer, criterion, scheduler, clip=5.0,
          unfreeze_on_epoch=5, verbose=False):
  """
  :param torch.nn.Module model:                         PyTorch model to train.
  :param torchtext.data.BucketIterator train_iterator:  Train Iterator for the Training Dataset.
  :param torchtext.data.BucketIterator val_iterator:    Val Iterator for the Validation Dataset.
  :param int epochs:                                    Number of epochs to do during training.
  :param torch.optim optimizer:                         Optimizer to use during training.
  :param nn.Loss criterion:                             Loss function used in the model.
  :param torch.optim.lr_scheduler scheduler:            The scheduler used to change the lr.
  :param int unfreeze_on_epoch:                         The epoch on which to unfeeze The
                                                          embedding layer.
  :param double clip:                                   The clipping threshold for gradients.
  :param bool verbose:                                  Bool determining whether we should log
                                                          the metrics of the model.

  :return:  A dictionary containing all the metrics computed during training.
  :rtype:   dict
  """

  # accumulate losses
  metrics = {
    'train_losses': [],
    'val_losses': [],
    'train_accuracies': [],
    'val_accuracies': [],
    'train_f1': [],
    'val_f1': []
  }

  # start training
  for epoch in range(epochs):

    if epoch == unfreeze_on_epoch:
      model.unfreeze_embedding()

    # start train mode
    model.train()
    sum_of_train_losses = 0.0
    sum_of_train_accuracies = 0.0
    sum_of_train_f1 = 0.0

    # train on the training dataset
    for batch in train_iterator:

      # unpack data
      X, lengths = batch.text
      y = batch.label.reshape(-1, 1)

      # reset gradients and compute loss
      optimizer.zero_grad()
      y_pred = model(X, lengths)
      loss = criterion(y_pred, y)

      # keep track of the different metrics
      sum_of_train_losses += loss.item()
      sum_of_train_accuracies += accuracy(y_pred, y)
      sum_of_train_f1 += F1_score(y_pred, y)

      # backpropagate and clip gradients
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

      # apply gradients
      optimizer.step()

    # evaluation time: don't calculate gradients
    with torch.no_grad():

      # faster inference
      model.eval()

      sum_of_val_losses = 0.0
      sum_of_val_accuracies = 0.0
      sum_of_val_f1 = 0.0

      for val_batch in val_iterator:

        X_val, lengths = val_batch.text
        y_val = val_batch.label.reshape(-1, 1)

        y_val_pred = model(X_val, lengths)
        loss = criterion(y_val_pred, y_val)

        # compute metrics
        sum_of_val_losses += loss.item()
        sum_of_val_accuracies += accuracy(y_val_pred, y_val)
        sum_of_val_f1 += F1_score(y_val_pred, y_val)

    # compute the mean train metrics
    mean_train_loss = sum_of_train_losses / len(train_iterator)
    mean_train_acc = sum_of_train_accuracies / len(train_iterator)
    mean_train_f1 = sum_of_train_f1 / len(train_iterator)

    # compute the mean val metrics
    mean_val_loss = sum_of_val_losses / len(val_iterator)
    mean_val_acc = sum_of_val_accuracies / len(val_iterator)
    mean_val_f1 = sum_of_val_f1 / len(val_iterator)

    # perform a step with the scheduler
    scheduler.step(mean_val_loss)

    # update the lists with the metrics
    metrics['train_losses'].append(mean_train_loss)
    metrics['train_accuracies'].append(mean_train_acc)
    metrics['train_f1'].append(mean_train_f1)
    metrics['val_losses'].append(mean_val_loss)
    metrics['val_accuracies'].append(mean_val_acc)
    metrics['val_f1'].append(mean_val_f1)

    # log information for the specific step, if specified
    if verbose:
      log_metrics(epoch, metrics)

  # return the dictionary containing the metrics computed
  return metrics
