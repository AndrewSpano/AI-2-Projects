import torch
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')


def compute_lengths(X):
    """ function used to compute a dictionary that maps:
        size of sentence -> number of sentences with this size in X """
    # the dictionary to be returned
    lengths = {}

    # for each sentence in X
    for sentence in X:
        # get its length
        l = len(sentence.split())
        # if this is the first time we see it, set the value to 1
        if l not in lengths:
            lengths[l] = 1
        # else increment it by one
        else:
            lengths[l] += 1

    # return the result
    return lengths


def pad_up_to_maxlen(split_sentence, maxlen, pad_token="<PAD>"):
    """ function used to pad a sentence up to maxlen """
    # get its current length
    current_len = len(split_sentence)

    # calculate how many pad token should be added and add them
    remaining_pads = maxlen - current_len
    split_sentence.extend([pad_token] * remaining_pads)

    # return new sentence
    return " ".join(split_sentence)


def truncate_down_to_maxlen(split_sentence, maxlen):
    """ function used to truncate a sentence down to maxlen words """
    # truncate it
    truncated_split_sentence = split_sentence[:maxlen]

    # return the rejoined sentence
    return " ".join(truncated_split_sentence)


def transform_sentence(sentence, maxlen, pad_token="<PAD>"):
    """ function used to invoke one of the above functions """
    # split the sentence in order to get its length
    split_sentence = sentence.split()

    # the sentence to be returned
    target_sentence = ""

    # check whether the sentence needs to be transformed
    if len(split_sentence) < maxlen:
        target_sentence = pad_up_to_maxlen(split_sentence, maxlen, pad_token)
    elif len(split_sentence) > maxlen:
        target_sentence = truncate_down_to_maxlen(split_sentence, maxlen)
    else:
        target_sentence = sentence

    # return the sentence
    return target_sentence


def load_word_embeddings(embedding_path, embedding_dimension, pad_token, unknown_token):
    """ function used to load word embeddings """
    # list of words
    words = [pad_token, unknown_token]
    # their indices
    index_of_word = 2
    # dictionary that maps: word -> index
    word_to_index_glove = {pad_token: 0, unknown_token: 1}
    # vectors with embeddings
    pad_vector = np.random.normal(scale=0.6, size=(embedding_dimension, ))
    unknown_vector = np.random.normal(scale=0.6, size=(embedding_dimension, ))
    vectors_glove = [pad_vector, unknown_vector]

    # used to remove stopwords and stem the given words so that they match our vocabulary
    stop_words = stopwords.words("english")
    stemmer = PorterStemmer()

    # finally open the embeddings file
    with open(embedding_path, "rb") as emb_file:

        # for each line the the embeddings file
        for l in emb_file:

            # decode the line and split it in space
            line = l.decode().split()
            # get the word and append it to the word list
            word = line[0]

            # if some special character is read, then probably the word gets skipped and we
            # end up reading the first number of the vector, skip this case
            if "0" in word or "1" in word or "2" in word or "3" in word or "4" in word or \
               "5" in word or "6" in word or "7" in word or "8" in word or "9" in word:
               continue

            # check if the word is a stopword. If it is, go to the next word
            if word in stop_words:
                continue

            # stem the word
            word = stemmer.stem(word)

            # check if this stemmed word has already been added, and if yes proceed
            if word in word_to_index_glove:
                continue

            # now append the word to the words list
            words.append(word)
            # also update the index dict
            word_to_index_glove[word] = index_of_word
            index_of_word += 1

            # initialize a vector with the values of the vector in this line
            vector = np.array(line[1:]).astype(np.float)
            vectors_glove.append(vector)

    # return the result
    return words, word_to_index_glove, vectors_glove


def map_sentences_to_indices_of_vectors(sentences, word_to_index_glove, unknown_token):
    """ map senteces to integers that represent the index of each word in the glove vocabulary """

    # the list to be returned
    mapped_sentences = []

    # get the index of the unknown token
    unknown_token_index = word_to_index_glove[unknown_token]

    # iterate for each sentence
    for sentence in sentences:

        # get the split sentence
        split_sentence = sentence.split()
        # map it to the corresponding indices
        mapped_sentence = [word_to_index_glove.get(word, unknown_token_index) for word in split_sentence]
        # append it to the list
        mapped_sentences.append(mapped_sentence)

    # return the list
    return mapped_sentences


def convert_to_torch_tensors(X_train, y_train, X_test, y_test):
    """ Function to quickly convert datasets to pytorch tensors """
    # convert training data
    _X_train = torch.LongTensor(X_train)
    _y_train = torch.FloatTensor(y_train)
    # convert test data
    _X_test = torch.LongTensor(X_test)
    _y_test = torch.FloatTensor(y_test)

    # return the tensors
    return _X_train, _y_train, _X_test, _y_test



def embed_and_flatten(X_train, X_test, maxlen, embedding_dimension, index_to_vector):
    """ Function to embed the training/test sets and flatten them """
    # number of training examples
    m_train = X_train.shape[0]

    # create an empty array
    _X_train = torch.zeros((m_train, maxlen, embedding_dimension))

    # iterate to gradually start filling this array
    for example in range(m_train):
        # get all indices to get the embedding of each one
        for index in range(maxlen):
            # get the value
            actual_index = X_train[example][index]
            # set the embedding
            _X_train[example, index, :] = index_to_vector[actual_index]

    # flatten it
    _X_train = torch.flatten(_X_train, start_dim=1)

    # check if only one dataset is needed
    if X_test is None:
        return _X_train

    # number of testing examples
    m_test = X_test.shape[0]

    # create an empty array
    _X_test = torch.zeros((m_test, maxlen, embedding_dimension))

    # iterate to gradually start filling this array
    for example in range(m_test):
        # get all indices to get the embedding of each one
        for index in range(maxlen):
            # get the value
            actual_index = X_test[example][index]
            # set the embedding
            _X_test[example, index, :] = index_to_vector[actual_index]

    # flatten it
    _X_test = torch.flatten(_X_test, start_dim=1)

    # return the 2 target vectors
    return _X_train, _X_test
