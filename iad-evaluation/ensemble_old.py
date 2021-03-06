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

parser = argparse.ArgumentParser(description="Ensemble model processor")
parser.add_argument('model', help='model to save (when training) or to load (when testing)')
parser.add_argument('num_classes', help='the number of classes in the dataset')
parser.add_argument('iad_dir', help='location of the generated IADs')

parser.add_argument('--train', default='', help='.list file containing the train files')
parser.add_argument('--test', default='', help='.list file containing the test files')

parser.add_argument('--gpu', default="0", help='gpu to run on')
parser.add_argument('--v', default=False, help='verbose')

args = parser.parse_args()

input_shape_c3d_full = [(64, 1024), (128, 1024), (256, 512), (256, 256), (256, 128)]
input_shape_c3d_frame = [(64, 64), (128, 64), (256, 32), (256, 16), (256, 8)]
input_shape_i3d = [(64, 32), (192, 32), (480, 32), (832, 16), (1024, 8)]
input_shape = input_shape_i3d

# optional - specify the CUDA device to use for GPU computation
# comment this line out if you wish to use all CUDA-capable devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

##############################################
# Parameters
##############################################

# trial-specific parameters
# BATCH_SIZE is the number of samples per training batch
# EPOCHS is the number of training epochs to complete
# ALPHA is the learning rate
# MODEL_NAME is the model to use to test the trained network
# USE_WEIGHTS
BATCH_SIZE = 15
EPOCHS = 30
ALPHA = 1e-4
MODEL_NAME = "ensemble"
USE_WEIGHTS = False


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



def model_consensus(confidences, csv_writer, true_class):
    """Generate a weighted average over the composite models"""
    confidence_discount_layer = [0.5, 0.7, 0.9, 0.9, 0.9, 1.0]

    confidences = confidences * confidence_discount_layer
    confidences = np.sum(confidences, axis=2)
    return np.argmax(confidences)


def read_file(filename_list):
    """Read the data from a given numpy data file."""
    all_data, all_labels = [], []
    for file in filename_list:
        infile = np.load(file)
        data, labels = infile["data"], infile["label"]
        data[data > 1] = 1.0
        data = data - 0.5
        all_data.append(data)
        all_labels = labels

    grouped_data = np.reshape(all_data[0], (all_data[0].shape[0], -1, 1))
    for i in range(1, len(all_data)):
        flat_data = np.reshape(all_data[i], (all_data[i].shape[0], -1, 1))
        grouped_data = np.concatenate((grouped_data, flat_data), axis=1)
    all_data.append(grouped_data)

    all_labels = all_labels.astype(np.uint8)

    return all_data, all_labels

def get_data_train(iad_list):
    
    batch_data = []
    for i in range(6):
        batch_data.append([])
    batch_label = []

    #select files randomly
    batch_indexs = np.random.randint(0, len(iad_list), size=BATCH_SIZE)

    for index in batch_indexs:

        file_data = []
        for layer in range(5):
            filename = iad_list[index][layer]
            f = np.load(filename)
            d, l, z = f["data"], f["label"], f["length"]

            #break d in to chuncks of window size
            window_size = input_shape[layer][1]
            pad_length = window_size - (z%window_size)
            d = np.pad(d, [[0,0],[0,pad_length]], 'constant', constant_values=0)
            d = np.split(d, d.shape[1]/window_size, axis=1)
            d = np.stack(d)
            file_data.append(d)

        flat_data = np.concatenate([x.reshape(x.shape[0], -1, 1) for x in file_data], axis = 1)
        file_data.append(flat_data)

        win_index = random.randint(0, file_data[0].shape[0]-1)
        for layer in range(len(file_data)):
            batch_data[layer].append(file_data[layer][win_index])
        batch_label.append(int(l))

    for i in range(6):
        batch_data[i] = np.array(batch_data[i])

    return batch_data, np.array(batch_label)

def get_data_test(iad_list, index):
   

    file_data = []
    for layer in range(5):
        filename = iad_list[index][layer]
        print("filename:"), filename
        f = np.load(filename)
        d, l, z = f["data"], f["label"], f["length"]

        #break d in to chuncks of window size
        window_size = input_shape[layer][1]
        pad_length = window_size - (z%window_size)
        d = np.pad(d, [[0,0],[0,pad_length]], 'constant', constant_values=0)
        d = np.split(d, d.shape[1]/window_size, axis=1)
        d = np.stack(d)
        file_data.append(d)

    #add flattened data segment
    flat_data = np.concatenate([x.reshape(x.shape[0], -1, 1) for x in file_data], axis = 1)
    file_data.append(flat_data)

    return file_data, np.array([l])


def tensor_operations(num_classes, data_shapes):
    """Create the tensor operations to be used in training and testing, stored in a dictionary."""
    # Placeholders
    ph = {
        "y": tf.placeholder(tf.int32, shape=(None)),
        "train": tf.placeholder(tf.bool)
    }

    for c3d_depth in range(6):
        print("train_data_shapes[c3d_depth]:", data_shapes[c3d_depth])

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
        if USE_WEIGHTS:
            weights[c3d_depth] = tf.get_variable("weight_%s" % c3d_depth, shape=[1], initializer=tf.ones_initializer())
            probabilities = tf.multiply(probabilities, weights[c3d_depth], "probability_weight")

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

    # average over softmaxes
    test_prob = tf.reduce_mean(all_preds, axis=2)
    test_class = tf.argmax(test_prob, axis=1, output_type=tf.int32)

    # verify if prediction is correct
    test_correct_pred = tf.equal(test_class, ph["y"])
    operations = dict()
    operations['ph'] = ph
    operations['loss_arr'] = loss_arr
    operations['train_op_arr'] = train_op_arr
    operations['predictions_arr'] = predictions_arr
    operations['accuracy_arr'] = accuracy_arr
    operations['weights'] = weights
    operations['logits'] = logits
    operations['all_preds'] = all_preds
    operations['model_preds'] = model_preds
    operations['model_top_10_values'] = model_top_10_values
    operations['model_top_10_indices'] = model_top_10_indices
    operations['test_prob'] = test_prob
    operations['test_class'] = test_class
    operations['test_correct_pred'] = test_correct_pred

    return operations


def train_model(model, train, test, num_classes):
    """Train a model given the dataset, dataset parameters, and a model name."""
    #Get Data Shape
    unified_shape = 0
    for shape in input_shape:
        unified_shape += shape[0]*shape[1]
    data_shape = input_shape + [(unified_shape, 1)]

    #define network
    ops = tensor_operations(num_classes, data_shape)
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # train the network
    
        num_iter = EPOCHS * len(train) / BATCH_SIZE
        for i in range(num_iter):
        # setup training batch

            data, label = get_data_train(train)
        
            batch_data = {}
            for d in range(6):
                batch_data[ops['ph']["x_" + str(d)]] = data[d]
            batch_data[ops['ph']["y"]] = label

            batch_data[ops['ph']["train"]] = True


            # combine training operations into one variable
            training_operations = ops['train_op_arr'] + ops['loss_arr'] + ops['accuracy_arr']
            
            start = time.time()
            out = sess.run(training_operations, feed_dict=batch_data)
            if(args.v):
                print("execution time: {:6.3f}".format(time.time() - start))

            # print out every 2K iterations
            if i % 2000 == 0:
                print("step: ", str(i) + '/' + str(num_iter))
                for x in range(6):
                    print("depth: ", str(x), "loss: ", out[6 + x], "train_accuracy: ", out[12 + x])

                # evaluate test network
                data, label = get_data_train(test)
            
                batch_data = {}
                for d in range(6):
                    batch_data[ops['ph']["x_" + str(d)]] = data[d]
                batch_data[ops['ph']["y"]] = label

                batch_data[ops['ph']["train"]] = False

                cp = sess.run([ops['test_correct_pred']], feed_dict=batch_data)
                # print("test_prob_max [shape = %s] = %s" % (tpm.shape, tpm))
                # print("test_class = %s" % tc)
                correct = np.sum(cp)
                total = len(cp[0])
                print("test: ", correct / float(total), "components: ", correct, total)

        # save the model
        save_path = saver.save(sess, model)
        print("Final model saved in %s" % save_path)

def test_model(model, test, num_classes):
    """Test the model."""

    #Get Data Shape
    unified_shape = 0
    for shape in input_shape:
        unified_shape += shape[0]*shape[1]
    data_shape = input_shape + [(unified_shape, 1)]

    #define network
    ops = tensor_operations(num_classes, data_shape)
    saver = tf.train.Saver()

    test_batch_size = 1
    correct, total = 0, 0
    model_correct = [0, 0, 0, 0, 0, 0]
    model_data_fd = open("%s_test_data.csv" % model, 'wb')
    model_csv = csv.writer(model_data_fd, dialect='excel')
    model_csv.writerow(["true_class", "model", "place", "class", "confidence"])
    confidences = [0.] * 6

    correct_class = np.zeros(int(args.num_classes), dtype=np.float32)
    total_class = np.zeros(int(args.num_classes), dtype=np.float32)

    with tf.Session() as sess:
        # restore the model
        try:
            saver.restore(sess, model)
            print("Model restored from %s" % model)
        except:
            print("Failed to load model")


        num_iter = len(test)
        for i in range(num_iter):
            data, label = get_data_test(test, i)

            batch_data = {}
            batch_data[ops['ph']["y"]] = int(label[0])
            batch_data[ops['ph']["train"]] = False

            aggregated_results = []
            for r in range(6):
                aggregated_results.append([])

            for j in range(len(data[0])):
                for d in range(6):
                    #print(d, j, len(data[0]))
                    input_data = data[d][j]
                    #print("------>input_data.shape:", np.expand_dims(input_data, axis = 0).shape)

                    batch_data[ops['ph']["x_" + str(d)]] = np.expand_dims(input_data, axis = 0)

                result = sess.run([
                    ops['test_correct_pred'],
                    ops['test_prob'],
                    ops['all_preds'],
                    ops['model_preds'],
                    ops['model_top_10_values'],
                    ops['model_top_10_indices']
                ], feed_dict=batch_data)

                for r in range(6):
                    aggregated_results[r].append(result[r])


            #print(aggregated_results[2][0])
            print(aggregated_results[5])
            for r in range(6):
                aggregated_results[r] = np.mean(np.array(aggregated_results[r]), axis=0)
            


            #print("classes:", batch_data[ops['ph']["y"]])
            ensemble_prediction = model_consensus(aggregated_results[2], model_csv, batch_data[ops['ph']["y"]])
            print(ensemble_prediction, int(label[0]), ensemble_prediction == int(label[0]))

            if(ensemble_prediction == int(label[0])):
                correct_class[int(label[0])] += 1

            total_class[int(label[0])] += 1

            # check if model output is correct
            for j, m in enumerate(result[3][0]):
                if m == batch_data[ops['ph']["y"]]:
                    model_correct[j] += 1
            if ensemble_prediction == batch_data[ops['ph']["y"]]:
                correct += 1

            # confidence collection
            for j, row in enumerate(result[4][0]):
                confidences[j] += np.max(row)

            total += len(result[0])

            if(i % 1000 == 0):
                print("step: ", str(i) + '/' + str(num_iter), "cummulative_accuracy:", correct / float(total))

    model_data_fd.close()
    print("FINAL - accuracy:", correct / float(total))
    print("Model avg. confidence:")
    for i, v in enumerate(confidences):
        print("%s: %s" % (i, v / float(total)))

    print("Model accuracy: ")
    for i, c in enumerate(model_correct):
        print("%s: %s" % (i, c / float(total)))

    np.save("classes.npy",  correct_class / total_class)
    

def locate_iads(file, iad_dict):
    iads = []
    ifile = open(file, 'r')

    line = ifile.readline()
    while len(line) != 0:
        filename = line.split()[0].split('/')[-1]
        iads.append(iad_dict[filename])

        line = ifile.readline()

    return np.array(iads)

def main():
    """Determine if the user has specified training or testing and run the appropriate function."""
    print("t0")

    iad_dict = {}
    for iad in os.listdir(args.iad_dir):
        iad_filename = iad[:-6]
        if(iad_filename not in iad_dict):
            iad_dict[iad_filename] = []
        iad_dict[iad_filename].append(os.path.join(args.iad_dir, iad) )

    for k in iad_dict.keys():
        iad_dict[k].sort()
    #print("t1")
    # define the dataset file names    
    eval_dataset = locate_iads(args.test, iad_dict) 
    #print("t2")
    if args.train != '':
        print("----> TRAINING")
        BATCH_SIZE = 15
        train_dataset = locate_iads(args.train, iad_dict)
        train_model(args.model, train_dataset, eval_dataset, args.num_classes)
    elif args.test != '':
        print("----> TESTING")
        BATCH_SIZE = 1

        test_model(args.model, eval_dataset, args.num_classes)
    else:
        print("Must provide either train or test file")
        sys.exit(1)

if __name__ == "__main__":
    main()
