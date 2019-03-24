#!/usr/bin/python3

import argparse
import logging
import copy
from time import time
from core import utils as U
from core import reader as R
from core import helper as H
from core.evaluator import Evaluator

logger = logging.getLogger(__name__)

###################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-train-csv", "--train-csv-path", dest="train_csv_path", type=str, metavar='<str>', required=True, help="The path to the CSV file")
parser.add_argument("-final-csv", "--final-test-csv-path", dest="final_test_csv_path", type=str, metavar='<str>', required=True, help="The path to the CSV file")
parser.add_argument("-img", "--img-path", dest="img_path", type=str, metavar='<str>', required=True, help="The path to the folder containing the images")
parser.add_argument("-imd", "--img-dim", dest="img_dim", type=int, metavar='<int>', default=640, help="Default image dimension to resize (default=640)")
parser.add_argument("-opt", "--optimizer", dest="optimizer_algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adam) (default=adam)")
parser.add_argument("-mt", "--model-type", dest="model_type", type=str, metavar='<str>', default='cnn', help="Model type (cnn|rescnn) (default=cnn)")

parser.add_argument("-sci", "--start-class-index", dest="start_class_index", type=int, metavar='<int>', default=0, help="Start class index to train (default=0)")
parser.add_argument("-eci", "--end-class-index", dest="end_class_index", type=int, metavar='<int>', default=20, help="End class index to train (default=20)")

parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=5, help="Number of epochs (default=5)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=16, help="Batch size for training (default=16)")
parser.add_argument("-be", "--batch-size-eval", dest="batch_size_eval", type=int, metavar='<int>', default=32, help="Batch size for evaluation (default=32)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")
parser.add_argument("--shuffle-seed", dest="shuffle_seed", type=int, metavar='<int>', default=1337, help="Random shuffle seed (default=1337)")

args = parser.parse_args()
U.mkdir_p(args.out_dir_path)
U.mkdir_p(args.out_dir_path + '/preds')
U.mkdir_p(args.out_dir_path + '/models')
U.set_logger(args.out_dir_path)
U.print_args(args)

itemid, title, image_path, y_multi, class_titles, class_max_number, num_classes = R.read_csv_train(args.train_csv_path)
final_itemid, final_title, final_image_path = R.read_csv_final_test(args.final_test_csv_path)

vocab, x_title = R.create_vocab(title)
final_x_title = R.convert_word_to_idx_using_vocab(vocab, final_title)


##############################################################
## Process data and train model according to class category
##
#

import numpy as np
if args.seed > 0:
    logger.info('Setting np.random.seed(%d) before importing torch' % args.seed)
    np.random.seed(args.seed)

import random
random.seed(args.seed)

start_idx = args.start_class_index
end_idx = min(args.end_class_index, num_classes)

for class_idx in range(start_idx, end_idx):
    # Start serious work here
    train_x_title = []
    train_x_image_path = []
    train_y = []
    train_filename_y = []
    train_y_multi = y_multi[class_idx]

    valid_x_title = []
    valid_x_image_path = []
    valid_y = []
    valid_filename_y = []

    assert (len(image_path) == len(x_title))
    assert (len(image_path) == len(train_y_multi))
    for i in range(len(train_y_multi)):
        if train_y_multi[i] != None:
            if (random.randint(1,10) <= 2):
                valid_x_title.append(x_title[i])
                valid_x_image_path.append(image_path[i])
                valid_y.append(train_y_multi[i])
                valid_filename_y.append(itemid[i])
            train_x_title.append(x_title[i])
            train_x_image_path.append(image_path[i])
            train_y.append(train_y_multi[i])
            train_filename_y.append(itemid[i])

    logger.info("=======================================")
    logger.info("Class %d (%s)" %(class_idx, class_titles[class_idx]))
    logger.info("Training size   : %d" % len(train_y))
    logger.info("Validation size : %d" % len(valid_y))

    # Padding for words
    # train_x_title = H.pad_sequences(train_x_title)
    # valid_x_title = H.pad_sequences(valid_x_title)

    # Get permutation list
    permutation_list = H.get_permutation_list(args, train_y)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.autograd import Variable
    import multiprocessing

    if ("rescnn" in args.model_type):
        from core.models import ResCNN as Net
    else:
        from core.models import SimpleCNN as Net
    model = Net(args, len(vocab), num_classes=class_max_number[class_idx]+1)

    if torch.cuda.is_available():
        model.cuda()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = None
    if args.optimizer_algorithm == "rmsprop":
        optimizer = optim.RMSprop(model.parameters())
    elif args.optimizer_algorithm == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = optim.Adam(model.parameters())

    total_train_time = 0
    total_eval_time = 0

    ############################################################################################
    ## Initialize Evaluator
    ## WARNING! Must be initialize Evaluator before padding to do dynamic padding
    #

    evl = Evaluator(
        args,
        (train_x_title, train_x_image_path, train_y, train_filename_y),
        (valid_x_title, valid_x_image_path, valid_y, valid_filename_y),
        (final_x_title, final_image_path, final_itemid),
        criterion, batch_size_eval=args.batch_size_eval, class_name=class_titles[class_idx])

    # logger.info('------------------------------------------------------')
    # logger.info('Initial Evaluation:')
    # evl.evaluate(model, -1)
    # evl.print_info()

    for ii in range(args.epochs):

        # Get current set of permutation indexes
        curr_perm = permutation_list[ii]
        # Split data and dynamically pad based on the batch size
        perm_train_x_title, perm_train_y, perm_train_img_path = H.sort_data_given_index(train_x_title, train_y, curr_perm, img_x=train_x_image_path)
        train_x_title_list, train_y_list, _, _, train_img_path_list  = H.split_data_into_chunks(perm_train_x_title, perm_train_y, args.batch_size, img_x=perm_train_img_path)

        t0 = time()
        # Train in chunks of batch size and dynamically padded
        model.train() # Set model to training mode, with dropout
        train_loss_sum = 0
        correct = 0

        # Go through the mini-batches
        for idx, _ in enumerate(train_x_title_list):
            # if (idx % 1000 == 0):
            #     logger.info("Processing %d out of %d" % (idx, len(train_x_title_list)))

            train_x_img_path = copy.deepcopy(train_img_path_list[idx])

            # Add the full path to image path
            for i in range(len(train_x_img_path)):
                train_x_img_path[i] = args.img_path + "/" + train_x_img_path[i]

            train_x_img = R.process_image_multiprocess(args.img_dim, train_x_img_path)
            train_x_img = np.array(train_x_img, dtype='float32')

            # Convert data to pytorch Variable
            pytorch_train_x_title = torch.from_numpy(train_x_title_list[idx].astype('int64'))
            pytorch_train_x_img = torch.from_numpy(train_x_img)
            pytorch_train_y = torch.from_numpy(train_y_list[idx].astype('int64'))
            if torch.cuda.is_available():
                pytorch_train_x_title = pytorch_train_x_title.cuda()
                pytorch_train_x_img = pytorch_train_x_img.cuda()
                pytorch_train_y = pytorch_train_y.cuda()
            pytorch_train_x_title = Variable(pytorch_train_x_title)
            pytorch_train_x_img = Variable(pytorch_train_x_img)
            pytorch_train_y = Variable(pytorch_train_y)
        
            # Forward pass and get loss
            output = model(pytorch_train_x_title, pytorch_train_x_img)
            loss = criterion(output, pytorch_train_y)

            # Compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.data.item() * len(train_y_list[idx])

        train_loss = train_loss_sum / (len(train_y))
        # logger.info("Loss : %.5f" % (train_loss))

        tr_time = time() - t0
        total_train_time += tr_time

        # Evaluate
        t0 = time()
        model.eval()
        evl.evaluate(model, ii)
        evl_time = time() - t0
        total_eval_time += evl_time

        logger.info('\r\n'+
        '============ Completed Epoch %d, train: %is (%.1fm), evaluation: %is (%.1fm), loss: %.4f =========' % (
            ii, tr_time, tr_time/60.0, evl_time, evl_time/60.0, train_loss)
        )
