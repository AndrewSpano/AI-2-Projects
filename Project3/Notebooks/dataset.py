import os
import logging
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.data import Field, BucketIterator, TabularDataset

from preprocessing import *


def split_dataset(root_dir, filename, split_dir_name, train_size=0.8):
    """
    :param str root_dir:        The path of the root directory containing the Dataset file.
    :param str filename:        The plain name of the file inside the root directory.
    :param str split_dir_name:  The path of the directory in which the split dataset files
                                    will be place into.
    :param double train_size:   The portion of the whole dataset which will be used for training.
                                    The rest is split equally among validation and testing.

    :return:  The names of the created files if the splitting succeeds; Else None.
    :rtype:   Optional((str, str, str))

    Function that reads the whole dataset, splits it into training-validation-testing sub-datasets,
    and stores each one in a specific file, the path of which gets returned by the function itself.
    If any error is raised, then an appropriate message is logged and None is returned.
    """

    # read the dataset and remove useless features
    dataset_path = os.path.join(root_dir, filename)
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        logging.error('The path {} does not correspond to the dataset, ' \
                      'aborting..'.format(dataset_path))
        return None
    df.drop(['Unnamed: 0', 'id', 'date', 'flag', 'user'], axis=1, inplace=True)
    df.reindex(columns=['text', 'target'])
    df['target'] = df['target'].apply(lambda x: int(x != 0))

    # df = df[:300000]

    # log some information
    labels = df['target'].tolist()
    print('Negative Occurences in full dataset: ', labels.count(0))
    print('Positive Occurences in full dataset: ', labels.count(1))
    print()

    # do the preprocessing here
    df['text'] = df['text'].apply(main_pipeline)

    # remove empty sentences
    indices_of_empty_sentences = df[df['text'] == ''].index
    df.drop(indices_of_empty_sentences, inplace=True)

    # do the splitting
    train, test = train_test_split(df, train_size=train_size, random_state=13, shuffle=True,
                                   stratify=df[['target']])
    val, test = train_test_split(test, train_size=0.5, random_state=13, stratify=test[['target']])


    # log some information
    train_labels = train['target'].tolist()
    print('Negative Occurences in split Training Dataset: ', train_labels.count(0))
    print('Positive Occurences in split Training Dataset: ', train_labels.count(1))
    print()

    val_labels = val['target'].tolist()
    print('Negative Occurences in split Validation Dataset: ', val_labels.count(0))
    print('Positive Occurences in split Validation Dataset: ', val_labels.count(1))
    print()

    test_labels = test['target'].tolist()
    print('Negative Occurences in split Test Dataset: ', test_labels.count(0))
    print('Positive Occurences in split Test Dataset: ', test_labels.count(1))

    # create the directory where the split files will be placed, and place them
    if not os.path.isdir(split_dir_name):
        os.mkdir(split_dir_name)

    train_filename = 'train_dataset.csv'
    train_dataset_name = os.path.join(split_dir_name, train_filename)
    val_filename = 'val_dataset.csv'
    val_dataset_name = os.path.join(split_dir_name, val_filename)
    test_filename = 'test_dataset.csv'
    test_dataset_name = os.path.join(split_dir_name, test_filename)

    train.to_csv(train_dataset_name, index=False)
    val.to_csv(val_dataset_name, index=False)
    test.to_csv(test_dataset_name, index=False)

    # return the names
    return train_filename, val_filename, test_filename


def dataset_length(split_dir_name, filename):
    """
    :param str split_dir_name:  The name of the root directory containing the dataset.
    :param str filename:        The name of the dataset file inside the root directory.

    :return:  The number of training examples contained in the dataset.
    :rtype:   int
    """
    dataset = os.path.join(split_dir_name, filename)
    df = pd.read_csv(dataset)
    return df.shape[0]


def parse_datasets(root_dir, filenames, device, batch_size=32, glove='glove.twitter.27B.25d',
                   max_vocab_size=20_000):
    """
    :param str root_dir:         The path to the root directory containing the split datasets.
    :param tuple filenames:      A triple (3-tuples) containing the filenames to the train-val-test
                                    datasets inside the root directory, respectively.
    :param torch.device device:  The device on which the torch tensors will be converted to,
                                    e.g. cuda
    :param int batch_size:       The batch size that will be used during training.
    :param str glove:            A string denoting which glove word embedding to use,
                                    e.g. glove.twitter.27B.25d
    :param int max_vocab_size:   The max size that the vocabulary of words which will be created
                                    from the training set, can have.

    :return:  The Field objectes created, along with the datasets loaded and the respective
                BucketIterators.
    :rtype:   (torchtext.data.Field, torchtext.data.Field, TabularDataset,
                (BucketIterator, BucketIterator, BucketIterator))

    This function loads the different datasets described in the paths triple, creates a vocabulary
    from the training set, along with BucketIterators for each dataset, and returns them all.
    """
    train_filename, val_filename, test_filename = filenames

    # create the corresponding Field objects for the datasets
    tokenizer = lambda sentence: sentence.split()
    TEXT = Field(sequential=True, use_vocab=True, tokenize=tokenizer, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)
    fields = {'text': ('text', TEXT), 'target': ('label', LABEL)}

    # create the Tabular Datasets and unpack them
    datasets = TabularDataset.splits(path=root_dir, train=train_filename, validation=val_filename,
                                     test=test_filename, format='csv', fields=fields)
    train_dataset, val_dataset, test_dataset = datasets

    # build the vocabulary, while initializing unknown words to random vectors from normal dict
    TEXT.build_vocab(train_dataset, max_size=max_vocab_size, vectors=glove,
                     unk_init=torch.Tensor.normal_)

    # construct the iterators
    test_examples = dataset_length(root_dir, test_filename)

    train_it = BucketIterator(train_dataset, batch_size=batch_size,
                              sort_key=lambda data: len(data.text), device=device, repeat=False,
                              sort_within_batch=False)
    val_it = BucketIterator(val_dataset, batch_size=batch_size,
                            sort_key=lambda data: len(data.text), device=device, repeat=False,
                            sort_within_batch=False)
    test_it = BucketIterator(test_dataset, batch_size=test_examples,
                             sort_key=lambda data: len(data.text), device=device, repeat=False,
                             sort_within_batch=False)
    
    """
    sort_key = lambda data: len(data.text)
    train_it, val_it, test_it = BucketIterator.splits((train_dataset, val_dataset, test_dataset),
                                                      batch_size=(batch_size, batch_size,
                                                                  test_examples),
                                                      sort_within_batch=False, sort_key=sort_key,
                                                      repeat=False, device=device)
    """

    # return the created objects
    iterators = (train_it, val_it, test_it)
    return TEXT, LABEL, datasets, iterators
