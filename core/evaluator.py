"""Evaluator.py - A python class/module to calculate neural network performance."""

import numpy as np
import pickle as pk
import copy
import logging
import math
from core import utils as U
from core import reader as R
from core import helper as H
from time import time

from torch.autograd import Variable
import torch.nn.functional as F
import torch

logger = logging.getLogger(__name__)

####################################################################################################
## Evaluator class
#

class Evaluator(object):
    """
    Evaluator class

    The class which handles evaluation
    after the neural network model is trained.
    It is also responsible to save the model and weights for future use.
    """

    def __init__(self, args, train, dev, test, loss_function, batch_size_eval=256, class_name=""):
        """
        Constructor to initialize the Evaluator class with the necessary attributes.
        """
        self.class_name = class_name
        self.img_dim = args.img_dim
        self.img_path = args.img_path
        self.out_dir = args.out_dir_path
        self.loss_function = loss_function
        self.batch_size_eval = batch_size_eval
        self.map_at_n = 2

        self.has_predicted_final_test = False

        self.train_x_title, self.train_x_img_path, self.train_y, self.train_filename_y = (
            train[0], train[1], train[2], train[3])
        self.dev_x_title, self.dev_x_img_path, self.dev_y, self.dev_filename_y = (
            dev[0], dev[1], dev[2], dev[3])
        self.test_x_title, self.test_x_img_path, self.test_filename_y = (
            test[0], test[1], test[2])

        # Sort data based on their length and pad them only per batch_size
        self.train_x_title, self.train_y, self.train_y_org, self.train_filename_y, self.train_x_img_path = (
            H.split_data_into_chunks(
                self.train_x_title, self.train_y, batch_size_eval, 
                filename_y=self.train_filename_y,img_x=self.train_x_img_path)
        )

        self.dev_x_title, self.dev_y, self.dev_y_org, self.dev_filename_y, self.dev_x_img_path = (
            H.split_data_into_chunks(
                self.dev_x_title, self.dev_y, batch_size_eval, 
                filename_y=self.dev_filename_y,img_x=self.dev_x_img_path)
        )

        # Dummy test_y, just for splitting
        self.test_y = list(range(len(self.test_filename_y)))
        self.test_x_title, self.test_y, self.test_y_org, self.test_filename_y, self.test_x_img_path = (
            H.split_data_into_chunks(
                self.test_x_title, self.test_y, batch_size_eval, 
                filename_y=self.test_filename_y,img_x=self.test_x_img_path)
        )

        self.train_y_org = self.train_y_org.astype('int64')
        self.dev_y_org = self.dev_y_org.astype('int64')

        self.best_dev = 0.0
        self.best_dev_loss = 0.0
        self.best_dev_epoch = -1

        self.dev_loss, self.dev_metric = 0.0, 0.0
        
        self.train_accuracy = 0.0
        self.train_mean_average_precision = 0.0
        self.train_loss = 0.0

        self.dev_accuracy = 0.0
        self.dev_mean_average_precision = 0.0
        self.dev_loss = 0.0

        self.train_pred = np.array([])
        self.dev_pred = np.array([])

        self.test_pred = []

    def dump_ref_filenames(self):
        """Dump/print the reference (ground truth) filenames to a file"""
        dev_ref_filename_file = open(self.out_dir + '/preds/dev_ref_filenames.txt', "w")
        for idx, _ in enumerate(self.dev_filename_y):
            for dev_filename in self.dev_filename_y[idx]:
                dev_ref_filename_file.write(dev_filename + '\n')
        dev_ref_filename_file.close()

        test_ref_filename_file = open(self.out_dir + '/preds/test_ref_filenames.txt', "w")
        for idx, _ in enumerate(self.test_filename_y):
            for test_filename in self.test_filename_y[idx]:
                test_ref_filename_file.write(test_filename + '\n')
        test_ref_filename_file.close()

    def dump_predictions(self, test_pred, epoch):
        """Dump predictions of the model on the final test set"""
        with open(self.out_dir + '/preds/test_pred_' + self.class_name + '_' + str(epoch) + '.txt', "w") as outfile:
            for idx, _ in enumerate(self.test_filename_y):
                for i in range(len(self.test_filename_y[idx])):
                    test_filename = self.test_filename_y[idx][i]
                    outfile.write(test_filename + '_' + self.class_name + ',' + str(test_pred[idx][i][0]) + ' ' + str(test_pred[idx][i][1]) + '\n')

    def save_model(self, model):
        """Save current best model"""
        torch.save(model, self.out_dir + '/models/best_model_weights_' + self.class_name + '.h5')

    def evaluate(self, model, epoch):
        """
        The main (most important) function in this class

        Evaluate a trained model at a given epoch on the development and test set.
        Handles all model saving, prediction score printing, dynamic padding by length sorting, etc.
        """

        # Reset train_pred, dev_pred and test_pred
        self.train_pred = np.array([])
        self.dev_pred = np.array([])
        self.test_pred = []
        self.train_loss = 0.0
        self.dev_loss = 0.0

        # Set model to eval mode, disabling dropout
        model.eval()

        # for idx, _ in enumerate(self.train_x_title):
        #     # if (idx % 1000 == 0):
        #     #     logger.info("Processing %d out of %d" % (idx, len(self.train_x_title)))

        #     curr_train_x_img_path = copy.deepcopy(self.train_x_img_path[idx])

        #     # Add the full path to image path
        #     for i in range(len(curr_train_x_img_path)):
        #         curr_train_x_img_path[i] = self.img_path + "/" + curr_train_x_img_path[i]
        
        #     train_x_img = R.process_image_multiprocess(self.img_dim, curr_train_x_img_path)
        #     train_x_img = np.array(train_x_img, dtype='float32')

        #     # Convert data to pytorch Variable
        #     pytorch_train_x_title = torch.from_numpy(self.train_x_title[idx].astype('int64'))
        #     pytorch_train_y =  torch.from_numpy(self.train_y[idx].astype('int64'))
        #     pytorch_train_x_img = torch.from_numpy(train_x_img)
        #     if torch.cuda.is_available():
        #         pytorch_train_x_title = pytorch_train_x_title.cuda()
        #         pytorch_train_x_img = pytorch_train_x_img.cuda()
        #         pytorch_train_y = pytorch_train_y.cuda()
        #     pytorch_train_x_title = Variable(pytorch_train_x_title)
        #     pytorch_train_x_img = Variable(pytorch_train_x_img)
        #     pytorch_train_y = Variable(pytorch_train_y)
            
        #     output = model(pytorch_train_x_title, pytorch_train_x_img)
            
        #     # Compute loss
        #     train_loss = self.loss_function(output, pytorch_train_y)
        #     self.train_loss += train_loss.data.item() * len(self.train_y[idx])
            
        #     # Compute real probability output
        #     softmax_outputs = F.softmax(output,dim=1)
        #     _, curr_train_pred = torch.topk(softmax_outputs, self.map_at_n, dim=1)
        #     curr_train_pred = curr_train_pred.cpu().data.numpy()

        #     if (self.train_pred.size > 0):
        #         self.train_pred = np.append(self.train_pred, curr_train_pred, axis=0)
        #     else:
        #         self.train_pred = curr_train_pred

        # self.train_loss = self.train_loss / len(self.train_y_org)
        

        for idx, _ in enumerate(self.dev_x_title):
            # if (idx % 1000 == 0):
            #     logger.info("Processing %d out of %d" % (idx, len(self.dev_x_title)))

            curr_dev_x_img_path = copy.deepcopy(self.dev_x_img_path[idx])

            # Add the full path to image path
            for i in range(len(curr_dev_x_img_path)):
                curr_dev_x_img_path[i] = self.img_path + "/" + curr_dev_x_img_path[i]
        
            dev_x_img = R.process_image_multiprocess(self.img_dim, curr_dev_x_img_path)
            dev_x_img = np.array(dev_x_img, dtype='float32')

            # Convert data to pytorch Variable
            pytorch_dev_x_title = torch.from_numpy(self.dev_x_title[idx].astype('int64'))
            pytorch_dev_y =  torch.from_numpy(self.dev_y[idx].astype('int64'))
            pytorch_dev_x_img = torch.from_numpy(dev_x_img)
            if torch.cuda.is_available():
                pytorch_dev_x_title = pytorch_dev_x_title.cuda()
                pytorch_dev_x_img = pytorch_dev_x_img.cuda()
                pytorch_dev_y = pytorch_dev_y.cuda()
            pytorch_dev_x_title = Variable(pytorch_dev_x_title)
            pytorch_dev_x_img = Variable(pytorch_dev_x_img)
            pytorch_dev_y = Variable(pytorch_dev_y)
            
            output = model(pytorch_dev_x_title, pytorch_dev_x_img)
            
            # Compute loss
            dev_loss = self.loss_function(output, pytorch_dev_y)
            self.dev_loss += dev_loss.data.item() * len(self.dev_y[idx])
            
            # Compute real probability output
            softmax_outputs = F.softmax(output,dim=1)
            _, curr_dev_pred = torch.topk(softmax_outputs, self.map_at_n, dim=1)
            curr_dev_pred = curr_dev_pred.cpu().data.numpy()

            if (self.dev_pred.size > 0):
                self.dev_pred = np.append(self.dev_pred, curr_dev_pred, axis=0)
            else:
                self.dev_pred = curr_dev_pred

        self.dev_loss = self.dev_loss / len(self.dev_y_org)

        # self.train_mean_average_precision = H.calculate_mean_average_precision(self.train_y_org, self.train_pred, n=self.map_at_n)
        self.dev_mean_average_precision = H.calculate_mean_average_precision(self.dev_y_org, self.dev_pred, n=self.map_at_n)

        best_updated = False
        if (self.dev_mean_average_precision > self.best_dev or epoch <= 0):
            self.best_dev_loss = self.dev_loss
            self.best_dev_epoch = epoch
            self.best_dev = self.dev_mean_average_precision
            
            # Save all necessary things, best epoch
            self.save_model(model)

            best_updated = True

        self.print_info()

        if (best_updated and epoch >= 7) or (epoch >= 7 and not self.has_predicted_final_test) or (epoch == 0):
            self.has_predicted_final_test = True

            logger.info("Starting prediction on final test set....")
            start = time()
            #############################################
            ## Run prediction on this
            ##
            for idx, _ in enumerate(self.test_x_title):
                # if (idx % 1000 == 0):
                #     logger.info("Processing %d out of %d" % (idx, len(self.test_x_title)))

                curr_test_x_img_path = copy.deepcopy(self.test_x_img_path[idx])

                # Add the full path to image path
                for i in range(len(curr_test_x_img_path)):
                    curr_test_x_img_path[i] = self.img_path + "/" + curr_test_x_img_path[i]
            
                test_x_img = R.process_image_multiprocess(self.img_dim, curr_test_x_img_path)
                test_x_img = np.array(test_x_img, dtype='float32')

                # Convert data to pytorch Variable
                pytorch_test_x_title = torch.from_numpy(self.test_x_title[idx].astype('int64'))
                pytorch_test_x_img = torch.from_numpy(test_x_img)
                if torch.cuda.is_available():
                    pytorch_test_x_title = pytorch_test_x_title.cuda()
                    pytorch_test_x_img = pytorch_test_x_img.cuda()
                pytorch_test_x_title = Variable(pytorch_test_x_title)
                pytorch_test_x_img = Variable(pytorch_test_x_img)
                
                output = model(pytorch_test_x_title, pytorch_test_x_img)
                
                # Compute real probability output
                softmax_outputs = F.softmax(output,dim=1)
                _, curr_test_pred = torch.topk(softmax_outputs, self.map_at_n, dim=1)
                curr_test_pred = curr_test_pred.cpu().data.numpy().tolist()

                self.test_pred.append(curr_test_pred)

            self.dump_predictions(self.test_pred, 9999)

            test_time = time() - start
            logger.info('Time taken : %.1f mins' % (test_time/60.0))

    @staticmethod
    def get_string_map(map_at_n, mean_ave_prec):
        return ( "MAP@%d: %.3f" % (map_at_n, mean_ave_prec))

    def print_info(self):
        """
        Print and return the current performance of the model.
        """

        logger.info(
            #"[TRAIN] Loss: %.5f  " % (self.train_loss) + Evaluator.get_string_map(self.map_at_n, self.train_mean_average_precision) + "\r\n" +
            "[DEV]   Loss: %.5f  " % (self.dev_loss) + Evaluator.get_string_map(self.map_at_n, self.dev_mean_average_precision)
        )
        self.print_final_info()
    
    def print_final_info(self):
        """Print and return the final performance of the model"""
        logger.info("[BEST-DEV @%2d]" % (self.best_dev_epoch) + " Loss: %.5f   " % (self.best_dev_loss) +
            Evaluator.get_string_map(self.map_at_n, self.best_dev)
        )
