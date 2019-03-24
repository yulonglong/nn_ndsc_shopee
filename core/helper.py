import argparse
import logging
import numpy as np
from time import time
import sys
from core import utils as U
import pickle as pk
import copy
from core import reader as reader

logger = logging.getLogger(__name__)

"""
Helper.py

Contains helper functions needed by other modules
"""



def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    '''This function is copied from from KERAS version 1.1.1

    Pads each sequence to the same length:
    the length of the longest sequence.

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def sort_data(x, y, filename_y, img_x=None):
    """Sort data based on the length of x"""

    test_img_x = img_x
    test_xy = zip(x, y, filename_y)
    if not (img_x is None):
        test_xy = zip(x, y, filename_y, img_x)

    # Sort tuple based on the length of the first entry in the tuple
    test_xy = sorted(test_xy, key=lambda t: len(t[0]))

    if not (img_x is None):
        test_x, test_y, test_filename_y, test_img_x = zip(*test_xy)
    else:
        test_x, test_y, test_filename_y = zip(*test_xy)

    return test_x, test_y, test_filename_y, test_img_x

def sort_data_given_index(x, y, perm_index, img_x=None):
    """
    Arrange data sequence given permutation index.
    The index was randomly shuffled before hand (usually for training)
    """
    assert len(x) == len(y)
    train_x = [None]* len(x)
    train_y = [None]* len(y)
    train_img_x = None
    if not (img_x is None): train_img_x = [None] * len(img_x)

    counter = 0
    for idx in perm_index:
        train_x[idx] = x[counter]
        train_y[idx] = y[counter]
        if not (img_x is None): train_img_x[idx] = img_x[counter]
        counter += 1

    return train_x, train_y, train_img_x

def split_data_into_chunks(x, y, batch_size, filename_y=None, img_x=None):
    """
    Split data into chunks/batches (with the specified batch size) for mini-batch processing in neural network.
    """

    test_x_chunks = list(chunks(x, batch_size))
    test_x = []
    test_x_len = 0

    test_y_chunks = list(chunks(y, batch_size))
    test_y = []
    test_y_len = 0

    test_filename_y_chunks = None
    test_filename_y = []
    test_filename_y_len = 0
    if not (filename_y is None): test_filename_y_chunks = list(chunks(filename_y, batch_size))

    test_img_x_chunks = None
    test_img_x = []
    test_img_x_len = 0
    if not (img_x is None): test_img_x_chunks = list(chunks(img_x, batch_size))

    assert len(test_x_chunks) == len(test_y_chunks)

    global_max_sentence_len = 0
    global_max_words_len = 0
    for i in range(len(test_x_chunks)):
        # Current mini batch
        curr_test_x = test_x_chunks[i]

        # The original padding, an ED note is a sequence of very long words
        curr_test_x = pad_sequences(curr_test_x)

        test_x.append(curr_test_x)
        test_x_len += len(curr_test_x)

        curr_test_y = test_y_chunks[i]
        curr_test_y = np.array(curr_test_y, dtype='float32')
        test_y.append(curr_test_y)
        test_y_len += len(curr_test_y)

        if not (filename_y is None):
            curr_test_filename_y = test_filename_y_chunks[i]
            test_filename_y.append(curr_test_filename_y)
            test_filename_y_len += len(curr_test_filename_y)

        if not (img_x is None):
            curr_test_img_x = test_img_x_chunks[i]
            test_img_x.append(curr_test_img_x)
            test_img_x_len += len(curr_test_img_x)
        

    assert test_x_len == test_y_len
    assert test_x_len == len(y)

    if not (filename_y is None):
        assert test_x_len == test_filename_y_len
    if not (img_x is None):
        assert test_x_len == test_img_x_len

    test_combined_y = np.array(y, dtype='float32')

    # if (global_max_sentence_len > 0 and global_max_words_len > 0):
    #     logger.info("Global max number of sentences in an ED note: " + str(global_max_sentence_len))
    #     logger.info("Global max number of words in a sentence    : " + str(global_max_words_len))

    return test_x, test_y, test_combined_y, test_filename_y, test_img_x

def sort_and_split_data_into_chunks(x, y, filename_y, batch_size, img_x=None):
    """
    Sort based on length of x
    Split test data into chunks of N (batch size) and pad them per chunk/batch
    Faster processing because of localized padding
    Usually used for validation and testing where the sequence of the dataset does not matter.
    """
    test_img_x = None
    test_x, test_y, test_filename_y, test_img_x = sort_data(x, y, filename_y, img_x=img_x)

    test_x, test_y, test_combined_y, test_filename_y, test_img_x = split_data_into_chunks(test_x, test_y, batch_size, filename_y=test_filename_y, img_x=test_img_x)
    return test_x, test_y, test_combined_y, test_filename_y, test_img_x


def get_permutation_list(args, train_y, save_to_file=False):
    """
    Produce a fixed list of permutation indices for training
    So that we dont depend on third party shuffling
    """
    
    shuffle_list_filename = "shuffle_permutation_list_len" + str(len(train_y)) + "_seed" + str(args.shuffle_seed) + ".txt"

    logger.info('Creating and saving shuffle permutation list to %s' % shuffle_list_filename)
    if args.shuffle_seed > 0:
        np.random.seed(args.shuffle_seed)

    # Create permutation list with the number of 2 x epochs
    # Actually 1x number of epoch is enough but just to be safe if need for other purposes
    permutation_list = []
    for ii in range(args.epochs*2):
        p = np.random.permutation(len(train_y))
        permutation_list.append(p)

    permutation_list = np.asarray(permutation_list, dtype=int)
    if save_to_file:
        np.savetxt(args.out_dir_path + "/" + shuffle_list_filename, permutation_list, fmt='%d')

    logger.info('Creating and saving shuffle permutation list completed!')
    return permutation_list



def calculate_mean_average_precision(y_gold, y_pred, n=3):
    """
    Calculate the mean average precision @ n
    Results have been validated and verified against sklearn.average_precision_score
    """

    y_actual = copy.deepcopy(y_gold)
    y_hat = copy.deepcopy(y_pred)

    
    y_actual = y_actual.tolist()
    y_hat = y_hat.tolist()

    y_hat_len = len(y_hat)

    assert (len(y_actual) == len(y_hat))

    total_ave_precision = 0.0
    num_classes = len(y_hat[0])

    pos_y_hat_len = 0
    for i in range(y_hat_len):

        relevant_answers = 1
        pos_y_hat_len += 1

        ave_precision = 0
        predicted_answers = 0
        correct_answers = 0
        for j in range (n):
            predicted_answers += 1
            if (y_actual[i] == y_hat[i][j]):
                correct_answers += 1
                ave_precision += float(correct_answers) / float(predicted_answers)

        ave_precision = ave_precision / float(relevant_answers)
        total_ave_precision += ave_precision

    mean_average_precision = float(total_ave_precision) / float(pos_y_hat_len)
    return mean_average_precision
