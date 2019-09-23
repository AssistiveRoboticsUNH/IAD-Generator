# ensemble.py
#
# This program takes as input a set of classified IADS...
#
# Usage:
# ensemble.py <train|test> <model name> <dataset name> <num_classes> <dataset size>
#   if train is specified, the model name will be the name of the saved model
#   if test is specified, the model name will be the name of the loaded model

import argparse
import csv
import numpy as np
import os
import sys
import tensorflow as tf

import time
import random

TRAIN_PREFIX = "train"
TEST_PREFIX = "test"

parser = argparse.ArgumentParser(description="Ensemble model processor")
parser.add_argument('model', help='model to save (when training) or to load (when testing)')
parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')
parser.add_argument('iad_dir', help='location of the generated IADs')
parser.add_argument('prefix', help='"train" or "test"')

#parser.add_argument('--train', default='', help='.list file containing the train files')
#parser.add_argument('--test', default='', help='.list file containing the test files')
parser.add_argument('--window_length', type=int, default=-1, help='the size of the window. If left unset then the entire IAD is fed in at once. \
                                                                    If the window is longer than the video then we pad to the IADs to that length')

parser.add_argument('--gpu', default="0", help='gpu to run on')
parser.add_argument('--v', default=False, help='verbose')

args = parser.parse_args()

input_shape_c3d_custom = [(64, args.window_length), (128, args.window_length), (256, args.window_length/2), (256, args.window_length/4), (256, args.window_length/8)]
input_shape_i3d_custom = [(64, args.window_length/2), (192, args.window_length/2), (480, args.window_length/2), (832, args.window_length/4), (1024, args.window_length/8)]

input_shape_c3d_full = [(64, 1024), (128, 1024), (256, 512), (256, 256), (256, 128)]
input_shape_c3d_frame = [(64, 64), (128, 64), (256, 32), (256, 16), (256, 8)]
input_shape_i3d = [(64, 32), (192, 32), (480, 32), (832, 16), (1024, 8)]
input_shape = input_shape_c3d_custom

# optional - specify the CUDA device to use for GPU computation
# comment this line out if you wish to use all CUDA-capable devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

##############################################
# Parameters
##############################################

# trial-specific parameters
# EPOCHS is the number of training epochs to complete
# ALPHA is the learning rate

EPOCHS = 30
ALPHA = 1e-4

##############################################
# File IO
##############################################

def parse_iadlist(iad_dir, prefix):

    iadlist_filename = os.path.join(iad_dir, prefix+".iadlist")

    try:
        ifile = open (iadlist_filename, 'r')
    except:
        print("File doesn't exist: "+ iadlist_filename)
        sys.exit(1)
    
    iad_groups = []

    line = ifile.readline()
    while(len(line) > 0):
        filename_group = [os.path.join(iad_dir, f) for f in line.split()]
        iad_groups.append(filename_group)
        line = ifile.readline()
    return iad_groups

def open_and_org_file(filename_group):
    file_data = []

    #join the separate IAD layers
    for layer, filename in enumerate(filename_group):
        
        f = np.load(filename)
        d, label, z = f["data"], f["label"], f["length"]

        #break d in to chuncks of window size
        window_size = input_shape[layer][1]
        pad_length = window_size - (z%window_size)
        d = np.pad(d, [[0,0],[0,pad_length]], 'constant', constant_values=0)
        d = np.split(d, d.shape[1]/window_size, axis=1)
        d = np.stack(d)
        file_data.append(d)

    #append the flattened and merged IAD
    flat_data = np.concatenate([x.reshape(x.shape[0], -1, 1) for x in file_data], axis = 1)
    file_data.append(flat_data)

    return file_data, np.array([int(label)])

def get_data_train(iad_list):
    
    batch_data = []
    for i in range(6):
        batch_data.append([])
    batch_label = []

    #select files randomly
    batch_indexs = np.random.randint(0, len(iad_list), size=BATCH_SIZE)

    for index in batch_indexs:
        file_data, label = open_and_org_file(iad_list[index])
        
        #randomly select a window from the example
        win_index = random.randint(0, file_data[0].shape[0]-1)
        for layer in range(len(file_data)):
            batch_data[layer].append(file_data[layer][win_index])

        batch_label.append(label)

    for i in range(6):
        batch_data[i] = np.array(batch_data[i])

    return batch_data, np.array(batch_label).reshape(-1)

def get_data_test(iad_list, index):
    return open_and_org_file(iad_list[index])

def locate_iads(file, iad_dict):
    iads = []
    ifile = open(file, 'r')

    line = ifile.readline()
    while len(line) != 0:
        filename = line.split()[0].split('/')[-1]
        iads.append(iad_dict[filename])

        line = ifile.readline()

    return np.array(iads)

##############################################
# Model Structure
##############################################

def model(features, c3d_depth, num_classes, data_shapes):
    """Return a single layer softmax model."""
    # input layers
    input_layer = tf.reshape(features["x_" + str(c3d_depth)], [-1, data_shapes[c3d_depth][0], data_shapes[c3d_depth][1], 1])  # batch_size, h, w, num_channels

    # hidden layers
    flatten = tf.reshape(input_layer, [-1, data_shapes[c3d_depth][0]* data_shapes[c3d_depth][1]])
    dense = tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.leaky_relu)
    dropout = tf.layers.dropout(dense, rate=0.5, training=features["train"])

    # output layers
    return tf.layers.dense(inputs=dropout, units=num_classes)


def conv_model(features, c3d_depth, num_classes, data_shapes):
    """Return a convolutional model."""
    # input layers
    input_layer = tf.reshape(features["x_" + str(c3d_depth)], [-1, data_shapes[c3d_depth][0], data_shapes[c3d_depth][1], 1])  # batch_size, h, w, num_channels

    # hidden layers
    num_filters = 32
    filter_width = 4
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=num_filters,
        kernel_size=[1, filter_width],
        padding="valid",  # don't want to add padding because that changes the IAD
        activation=tf.nn.leaky_relu)
    flatten = tf.reshape(input_layer, [-1, data_shapes[c3d_depth][0]* data_shapes[c3d_depth][1]])
    dense = tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.leaky_relu)
    dropout = tf.layers.dropout(dense, rate=0.5, training=features["train"])

    # output layers
    return tf.layers.dense(inputs=dropout, units=num_classes)

def model_consensus(confidences):
    """Generate a weighted average over the composite models"""
    confidence_discount_layer = [0.5, 0.7, 0.9, 0.9, 0.9, 1.0]

    confidences = confidences * confidence_discount_layer
    confidences = np.sum(confidences, axis=2)
    return np.argmax(confidences)

def tensor_operations(num_classes, data_shapes):
    """Create the tensor operations to be used in training and testing, stored in a dictionary."""
    # Placeholders
    ph = {
        "y": tf.placeholder(tf.int32, shape=(None)),
        "train": tf.placeholder(tf.bool)
    }

    for c3d_depth in range(6):
        ph["x_" + str(c3d_depth)] = tf.placeholder(
            tf.float32, shape=(None, data_shapes[c3d_depth][0], data_shapes[c3d_depth][1])
        )

    # Tensor operations
    loss_arr = []
    train_op_arr = []
    predictions_arr = []
    accuracy_arr = []
    weights = {}

    # for each model generate the tensor ops
    for c3d_depth in range(6):
        # logits
        if(c3d_depth < 3):
            logits = conv_model(ph, c3d_depth, num_classes, data_shapes)
        else:
            logits = model(ph, c3d_depth, num_classes, data_shapes)

        # probabilities and associated weights
        probabilities = tf.nn.softmax(logits, name="softmax_tensor")
        

        # functions for predicting class
        predictions = {
            "classes": tf.argmax(input=logits, axis=1, output_type=tf.int32),
            "probabilities": probabilities
        }
        predictions_arr.append(predictions)

        # functions for training/optimizing the network
        loss = tf.losses.sparse_softmax_cross_entropy(labels=ph["y"], logits=logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        loss_arr.append(loss)
        train_op_arr.append(train_op)

        # functions for evaluating the network
        correct_pred = tf.equal(predictions["classes"], ph["y"])
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        accuracy_arr.append(accuracy)

    # combine all of the models together for the ensemble
    all_preds = tf.stack([x["probabilities"] for x in predictions_arr])
    all_preds = tf.transpose(all_preds, [1, 2, 0])

    model_preds = tf.transpose(all_preds, [0, 2, 1])
    model_top_10_values, model_top_10_indices = tf.nn.top_k(model_preds, k=10)
    model_preds = tf.argmax(model_preds, axis=2, output_type=tf.int32)
    model_preds = tf.squeeze(model_preds)

    # average over softmaxes
    test_prob = tf.reduce_mean(all_preds, axis=2)
    test_class = tf.argmax(test_prob, axis=1, output_type=tf.int32)

    # verify if prediction is correct
    test_correct_pred = tf.equal(test_class, ph["y"])
    ops = dict()
    placeholders = ph
    ops['loss_arr'] = loss_arr
    ops['train_op_arr'] = train_op_arr
    ops['predictions_arr'] = predictions_arr#
    ops['accuracy_arr'] = accuracy_arr
    ops['weights'] = weights#
    ops['logits'] = logits#
    ops['all_preds'] = all_preds#
    ops['model_preds'] = model_preds#
    ops['model_top_10_values'] = model_top_10_values#
    ops['model_top_10_indices'] = model_top_10_indices#
    ops['test_prob'] = test_prob#
    ops['test_class'] = test_class#

    ops['test_correct_pred'] = test_correct_pred
    ops["train"] = ops['train_op_arr'] + ops['loss_arr'] + ops['accuracy_arr']

    return ph, ops

def train_model(model_name, num_classes, train_data, test_data):

    # get the shape of the flattened and merged IAD and append
    data_shape = input_shape + [(np.sum([shape[0]*shape[1] for shape in input_shape]), 1)]

    #define network
    ph, ops = tensor_operations(num_classes, data_shape)
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # train the network
    
        num_iter = EPOCHS * len(train_data) / BATCH_SIZE
        for i in range(num_iter):
        # setup training batch

            data, label = get_data_train(train_data)
        
            batch_data = {}
            for d in range(6):
                batch_data[ph["x_" + str(d)]] = data[d]

            batch_data[ph["y"]] = label
            batch_data[ph["train"]] = True

            # combine training operations into one variable
            start = time.time()

            out = sess.run(ops["train"], feed_dict=batch_data)

            if(args.v):
                print("execution time: {:6.3f}".format(time.time() - start))

            # print out every 2K iterations
            if i % 2000 == 0:
                print("step: ", str(i) + '/' + str(num_iter))
                for x in range(6):
                    print("depth: ", str(x), "loss: ", out[6 + x], "train_accuracy: ", out[12 + x])

                # evaluate test network
                data, label = get_data_train(test_data)
            
                batch_data = {}
                for d in range(6):
                    batch_data[ph["x_" + str(d)]] = data[d]

                batch_data[ph["y"]] = label
                batch_data[ph["train"]] = False

                correct_prediction = sess.run([ops['test_correct_pred']], feed_dict=batch_data)
                correct, total = np.sum(correct_prediction), len(correct_prediction[0])

                print("test_accuracy: {0}, correct: {1}, total: {2}".format(correct / float(total), correct, total))

        # save the model
        print("Final model saved in %s" % saver.save(sess, model_name))

def test_model(model_name, num_classes, test_data):

    # get the shape of the flattened and merged IAD and append
    test_batch_size = 1
    data_shape = input_shape + [(np.sum([shape[0]*shape[1] for shape in input_shape]), 1)]

    #define network
    ph, ops = tensor_operations(num_classes, data_shape)
    saver = tf.train.Saver()

    correct, total = 0, 0
    model_correct, model_total = [0]*6, [0]*6

    correct_class = np.zeros(num_classes, dtype=np.float32)
    total_class = np.zeros(num_classes, dtype=np.float32)

    with tf.Session() as sess:
        # restore the model
        try:
            saver.restore(sess, model_name)
            print("Model restored from %s" % model_name)
        except:
            print("Failed to load model")

        num_iter = len(test_data)
        for i in range(num_iter):
            data, label = get_data_test(test_data, i)
            label = int(label[0])
            
            aggregated_confidences = []

            batch_data = {}
            batch_data[ph["y"]] = label
            batch_data[ph["train"]] = False

            for j in range(len(data[0])):
                for d in range(6):
                    batch_data[ph["x_" + str(d)]] = np.expand_dims(data[d][j], axis = 0)

                confidences, predictions = sess.run([
                    ops['all_preds'], # confidences
                    ops['model_preds'], # predictions
                ], feed_dict=batch_data)

                aggregated_confidences.append(confidences)
                print("predictions:", predictions.shape)

                for d in range(6):
                    if(predictions[d] == label):
                        model_correct[d] += 1
                    model_total[d] += 1

            aggregated_confidences = np.mean(aggregated_confidences, axis=0)
            ensemble_prediction = model_consensus(aggregated_confidences)

            

            #check if ensemble is correct
            if(ensemble_prediction == label):
                correct_class[label] += 1
            total_class[label] += 1
            


    print("Model accuracy: ")
    for i in range(6):
        print("%s: %s" % (i, model_correct[i] / float(model_total[i])))
           
    print("sum:",  np.sum(correct_class),  np.sum(total_class))
    print("FINAL - accuracy:", np.sum(correct_class) / np.sum(total_class))
    np.save("classes.npy",  correct_class / total_class)



if __name__ == "__main__":
    """Determine if the user has specified training or testing and run the appropriate function."""
    '''
    iad_dict = {}
    for iad in os.listdir(args.iad_dir):
        iad_filename = iad[:-6]
        if(iad_filename not in iad_dict):
            iad_dict[iad_filename] = []
        iad_dict[iad_filename].append(os.path.join(args.iad_dir, iad) )

    for k in iad_dict.keys():
        iad_dict[k].sort()
    '''
  
    # define the dataset file names    
    #eval_dataset = parse_iadlist(args.iad_dir, TEST_PREFIX)#locate_iads(args.test, iad_dict) 

    if args.prefix == TRAIN_PREFIX:
        print("----> TRAINING")
        BATCH_SIZE = 15
        train_dataset = parse_iadlist(args.iad_dir, TRAIN_PREFIX)
        eval_dataset = parse_iadlist(args.iad_dir, TEST_PREFIX)
        train_model(args.model, args.num_classes, train_dataset, eval_dataset)
    elif args.prefix == TEST_PREFIX:
        print("----> TESTING")
        BATCH_SIZE = 1
        eval_dataset = parse_iadlist(args.iad_dir, TEST_PREFIX)
        test_model(args.model, args.num_classes, eval_dataset)
    else:
        print('"prefix must be either "train" or "test"')
        sys.exit(1)