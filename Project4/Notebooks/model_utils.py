import torch
import torch.nn as nn
import tqdm.notebook as tq



def determine_checkpoint(idx, epoch):
    """ finds the number of the current batch group update """
    if epoch == 0:
        return idx
    else:
        return idx + checkpoints[-1]



def train_model(model, train_dataloader, dev_dataloader, criterion, optimizer, epochs, losses, log_every=50):
    """
    :param nn.Module model:              The PyTorch model to be trained.
    :param DataLoader train_dataloader:  DataLoader that iterates through the Training Set.
    :param DataLoader dev_dataloader:    DataLoader that iterates through the Development Set.
    :param nn.Module criterion:          Loss function.
    :param torch.optim optimizer:        Optimizer.
    :param int epochs:                   Number of epochs to train the model.
    :param dict losses:                  Dictionary where all the losses will be stored.
    :param int log_every:                Frequency (in batches) of logging and computing of loss.

    :return:  None.

    This function trains a PyTorch model for a specific number of epochs, using the provided
    dataloaders, loss function and optimizer. The results are stored in lists inside the "losses"
    dictionary.
    """

    for epoch in range(epochs):

        print("Epoch {}:".format(epoch + 1))
        epoch_train_loss = 0
        average_train_loss = 0

        # run through the batches of the Training Dataset
        for idx, batch in tq.tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                  position=0, leave=True):

            optimizer.zero_grad()

            X, y, ids = batch

            start_positions_true, end_positions_true = y[0].squeeze(-1), y[1].squeeze(-1)
            start_positions_pred, end_positions_pred = model(X)

            start_positions_loss = criterion(start_positions_pred, start_positions_true)
            end_positions_loss = criterion(end_positions_pred, end_positions_true)

            final_loss = (start_positions_loss + end_positions_loss) / 2
            epoch_train_loss += final_loss.item()
            average_train_loss += final_loss.item()

            if (idx + 1) % log_every == 0 or idx == len(train_dataloader) - 1:
                iterations = log_every if (idx + 1) % log_every == 0 \
                                       else len(train_dataloader) % log_every
                checkpoint = idx if epoch == 0 \
                                 else losses['average_train_checkpoints'][-1] + iterations

                average_train_loss = average_train_loss / iterations
                losses['average_train_losses'].append(average_train_loss)
                losses['average_train_checkpoints'].append(checkpoint)

                print('Average Training Loss between ' \
                      'batches {} - {}: {}'.format(idx+2-iterations, idx+1, average_train_loss))
                average_train_loss = 0

            final_loss.backward()

            optimizer.step()

        # compute the average training loss for the current epoch
        epoch_train_loss = epoch_train_loss / len(train_dataloader)
        losses['average_epoch_train_losses'].append(epoch_train_loss)

        # evaluate the model in the devset after the training has finished for this epoch
        optimizer.zero_grad()
        model.eval()

        # do not compute gradients
        with torch.no_grad():

            epoch_val_loss = 0
            average_val_loss = 0

            for idx, batch in tq.tqdm(enumerate(dev_dataloader), total=len(dev_dataloader),
                                      position=0, leave=True):

                X, y, ids = batch
                start_positions_true, end_positions_true = y[0].squeeze(-1), y[1].squeeze(-1)
                start_positions_pred, end_positions_pred = model(X)

                start_positions_loss = criterion(start_positions_pred, start_positions_true)
                end_positions_loss = criterion(end_positions_pred, end_positions_true)

                final_loss = (start_positions_loss + end_positions_loss) / 2
                epoch_val_loss += final_loss.item()
                average_val_loss += final_loss.item()

                if (idx + 1) % log_every == 0 or idx == len(dev_dataloader) - 1:
                    iterations = log_every if (idx + 1) % log_every == 0 \
                                           else len(dev_dataloader) % log_every
                    checkpoint = idx if epoch == 0 \
                                     else losses['average_val_checkpoints'][-1] + iterations

                    average_val_loss = average_val_loss / iterations
                    losses['average_val_losses'].append(average_val_loss)
                    losses['average_val_checkpoints'].append(checkpoint)

                    print('Average Validation Loss between ' \
                          'batches {} - {}: {}'.format(idx+2-iterations, idx+1, average_val_loss))
                    average_val_loss = 0

            # compute the average validation loss for the current epoch
            epoch_val_loss = epoch_val_loss / len(dev_dataloader)
            losses['average_epoch_val_losses'].append(epoch_val_loss)

        # place again the model in training mode for the next epoch
        model.train()
        print()



def load_pytorch_model(model, path, device):
    """ wrapper function used to load a pytorch model, given it's save path """
    model.load_state_dict(torch.load(path, map_location=device))



def save_pytorch_model(model, path):
    """ wrapper function used to save a pytorch model, given it's save path """
    torch.save(model.state_dict(), path)



def get_answer(input_ids, tokenizer, start_pred, end_pred):
    """ function to extract the answer text from start and end index tokens """
    if start_pred == 0 or end_pred == 0 or end_pred < start_pred:
        return ''
    else:
        token_ids = input_ids.tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        answer = tokens[start_pred]
        for token_index in range(start_pred + 1, end_pred + 1):
            if tokens[token_index][0:2] == '##':
                answer += tokens[token_index][2:]
            else:
                answer += ' ' + tokens[token_index]
    return answer



def map_ids_to_predictions(model, dataloader, tokenizer, is_fine_tuned=False):
    """
    :param torch.nn.Module model:    The model which we will use to make the predictions.
    :param Dataloader dataloader:    The dataloader to a Dataset from which predictions are made.
    :param BertTokenizer tokenizer:  The tokenizer used to convert IDs to their respective tokens.
    :param bool is_fine_tuned:       Whether the model is we are using to make predictions is the
                                        "bert-large-uncased-whole-word-masking-finetuned-squad".

    :return:  A dictionary that maps question ID -> Predicted Answer (in text format).
    :rtype:   Dict
    """
    question_id_to_prediction = {}
    model.eval()

    with torch.no_grad():

        for batch in tq.tqdm(dataloader, total=len(dataloader), position=0, leave=True):

            X, y, ids = batch
            if is_fine_tuned:
                bert_inputs, mask_ids, segment_ids = X
                outputs = model(bert_inputs, attention_mask=mask_ids, token_type_ids=segment_ids,
                                return_dict=True)
                start_positions_pred = outputs.start_logits
                end_positions_pred = outputs.end_logits
            else:
                start_positions_pred, end_positions_pred = model(X)

            actual_start_positions = torch.argmax(start_positions_pred, dim=1)
            actual_end_positions = torch.argmax(end_positions_pred, dim=1)

            bert_inputs, mask_ids, segment_ids = X

            for bert_input, question_id, start_pred, end_pred in zip(bert_inputs, ids,
                                                                     actual_start_positions,
                                                                     actual_end_positions):

                answer = get_answer(bert_input, tokenizer, start_pred, end_pred)
                question_id_to_prediction[question_id] = answer

    return question_id_to_prediction



def predict_answer(model, question, context, tokenizer, maxlen, device):
    """
    :param nn.Module model:          Model used to predict the answer of the question.
    :param str question:             Question string.
    :param str context:              Context string containing the answer to the question.
    :param BertTokenizer tokenizer:  The BERT tokenizer used to encode the input.
    :param int maxlen:               The maximum length that a sentence can (and must) have.
    :param torch.device device:      The device (cuda or cpu) on which the model operates on.

    :return:  A string with the answer (output of the model) to the given question/.
    :rtype:   str
    """

    encoded = tokenizer.encode_plus(question, context)
    input_ids = encoded['input_ids']
    segment_ids = encoded['token_type_ids']
    mask_ids = encoded['attention_mask']
    length_before_padding = len(input_ids)

    def pad_to_maxlen(input, maxlen, pad_token):
        return input + (maxlen - len(input)) * [pad_token]

    input_ids = torch.LongTensor(pad_to_maxlen(input_ids, maxlen, 0)).reshape(1, -1).to(device)
    mask_ids = torch.LongTensor(pad_to_maxlen(mask_ids, maxlen, 0)).reshape(1, -1).to(device)
    segment_ids = torch.LongTensor(pad_to_maxlen(segment_ids, maxlen, 1)).reshape(1, -1).to(device)

    X = (input_ids, mask_ids, segment_ids)
    start_positions_pred, end_positions_pred = model(X)

    actual_start_position = torch.argmax(start_positions_pred, dim=1).item()
    actual_end_position = torch.argmax(end_positions_pred, dim=1).item()

    original_input_ids = input_ids.squeeze(0)
    return get_answer(original_input_ids, tokenizer, actual_start_position, actual_end_position)
