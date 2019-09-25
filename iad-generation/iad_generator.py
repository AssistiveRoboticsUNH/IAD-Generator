# Madison Clark-Turner
# iad_generator.py
# 8/29/2019

#import c3d as model
import c3d_large as model
#import i3d_wrapper as model

import os

import tensorflow as tf
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Generate IADs from input files')
#required command line args
parser.add_argument('model_file', help='the tensorflow ckpt file used to generate the IADs')
parser.add_argument('prefix', help='"train" or "test"')
parser.add_argument('dataset_file', help='the *.list file than contains the ')
#optional command line args

parser.add_argument('--min_max_file', nargs='?', default=None, help='max and minimum values')
parser.add_argument('--features_file', nargs='?', default=None, help='which features to keep')
parser.add_argument('--dst_directory', nargs='?', default='generated_iads/', help='where the IADs should be stored')
parser.add_argument('--pad_length', type=int, nargs='?', default=-1, help='length to pad/prune the videos to, default is padd to the longest file in the dataset')

parser.add_argument('--gpu', default="1", help='gpu to run on')
parser.add_argument('--c', type=bool, default=False, help='combine files')
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

batch_size = 1

def output_filename(file, layer, dirname=FLAGS.dst_directory):
	if(dirname == ''):
		return file.split(os.path.sep)[-1]+"_"+str(layer)+".npz"
	return os.path.join(FLAGS.dst_directory, file.split(os.path.sep)[-1]+"_"+str(layer)+".npz")

def convert_to_iad(data, label, file, min_max_vals, update_min_maxes, length_ratio):
	#converts file to iad and extracts the max and min values for the given IAD

	#update max and min values
	if(min_max_vals):
		for layer in range(len(data)):
			local_max_values = np.max(data[layer], axis=1)
			local_min_values = np.min(data[layer], axis=1)

			for i in range(len(local_max_values)):
				if(local_max_values[i] > min_max_vals["max"][layer][i]):
					min_max_vals["max"][layer][i] = local_max_values[i]

				if(local_min_values[i] < min_max_vals["min"][layer][i]):
					min_max_vals["min"][layer][i] = local_min_values[i]

	#save to disk
	for i in range(len(data)):
		filename = output_filename(file, i)
		data[i] = data[i][:, :int(data[i].shape[1]*length_ratio)]

		np.savez(filename, data=data[i], label=label, length=data[i].shape[1])

def convert_dataset_to_iad(list_of_files, min_max_vals, update_min_maxes):
	
	# define placeholder
	input_placeholder = model.get_input_placeholder(batch_size, num_frames=FLAGS.pad_length )
	print("input_placeholder.get_shape():", input_placeholder.get_shape())
	
	# define model
	activation_map, saver = model.load_model(input_placeholder)
	
	#collapse the spatial dimensions of the activation map
	for layer in range(len(activation_map)):
		activation_map[layer] = tf.reduce_max(activation_map[layer], axis = (2,3))
		activation_map[layer] = tf.squeeze(activation_map[layer])
		activation_map[layer] = tf.transpose(activation_map[layer])

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9)#.25
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		# initialize model variables to those in the described checkpoint file
		ckpt = tf.train.get_checkpoint_state(FLAGS.model_file)
		if ckpt and ckpt.model_checkpoint_path:
			print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("load complete!")

		# prevent further modification to the graph
		sess.graph.finalize()

		# process files
		for i in range(len(list_of_files)):
			file, label = list_of_files[i]

			print("converting video to IAD: {:6d}/{:6d}".format(i, len(list_of_files)))

			# read data into placeholders
			raw_data, length_ratio = model.read_file(file, input_placeholder)

			# generate activation map from model
			iad_data = sess.run(activation_map, feed_dict={input_placeholder: raw_data})

			# write the am_layers to file and get the minimum and maximum values for each feature row
			convert_to_iad(iad_data, label, file, min_max_vals, update_min_maxes, length_ratio)

	#save min_max_vals
	if(update_min_maxes):
		np.savez(os.path.join(FLAGS.dst_directory, "min_maxes.npz"), min=np.array(min_max_vals["min"]), max=np.array(min_max_vals["max"]))
	
def normalize_dataset(list_of_files, min_max_vals):
	for i in range(len(list_of_files)):
		file, _ = list_of_files[i]

		print("normalizing IAD: {:6d}/{:6d}".format(i, len(list_of_files)))

		for layer in range(len(model.CNN_FEATURE_COUNT)):

			filename = output_filename(file, layer)
			f = np.load(filename)
			data, label, length = f["data"], f["label"], f["length"]

			for row in range(data.shape[0]):
				if(min_max_vals["max"][layer][row] - min_max_vals["min"][layer][row] == 0):
					data[row] = np.zeros_like(data[row])
				else:
					data[row] = (data[row] - min_max_vals["min"][layer][row]) / (min_max_vals["max"][layer][row] - min_max_vals["min"][layer][row])

			np.savez(filename, data=data, label=label, length=length)

def combine_npy_files(list_of_files):
	# combine all of the IADs from a specific depth together
	for layer in range(len(model.CNN_FEATURE_COUNT)):
		data_all, label_all, length_all = [],[],[]
		for i in range(len(list_of_files)):
			file, _ = list_of_files[i]

			# open data
			filename = output_filename(file, layer)
			f = np.load(filename)
			data, label, length = f["data"], f["label"], f["length"]

			#pad data to common length
			data = np.pad(data, [[0,0],[0,FLAGS.pad_length-length]], 'constant', constant_values=0)

			data_all.append(data)
			label_all.append(label)
			length_all.append(length)

		np.savez(os.path.join(FLAGS.dst_directory, FLAGS.prefix+"_"+str(layer)+".npz"), 
				data=np.array(data_all), 
				label=np.array(label_all), 
				length=np.array(length_all))

def clean_up_npy_files(list_of_files):
	for i in range(len(list_of_files)):
		file, _ = list_of_files[i]
		for layer in range(len(model.CNN_FEATURE_COUNT)):
			os.remove(output_filename(file, layer))

def make_iadlist_file(list_of_files):
	ofile = open(os.path.join(FLAGS.dst_directory, FLAGS.prefix+".iadlist"), 'w')
	print("writing iadlist file: "+os.path.join(FLAGS.dst_directory, FLAGS.prefix+".iadlist"))

	for i in range(len(list_of_files)):
		file, _ = list_of_files[i]

		entry = output_filename(file, 0, dirname='')+' '
		for layer in range(1, len(model.CNN_FEATURE_COUNT)):
			entry += output_filename(file, layer, dirname='')+' '

		ofile.write(entry+'\n')
	ofile.close()



if __name__ == '__main__':
	
	list_of_files_and_labels, max_frame_length = model.obtain_files(FLAGS.dataset_file)
	#list_of_files_and_labels = list_of_files_and_labels[:3]

	print("list_of_files_and_labels:", len(list_of_files_and_labels))
	print("max_frame_length:", max_frame_length)

	if(not os.path.exists(FLAGS.dst_directory)):
		os.makedirs(FLAGS.dst_directory)

	if (FLAGS.pad_length < 0):
		FLAGS.pad_length = max_frame_length
		print("padding iads to a length of {0} frames".format(max_frame_length))

	# generate arrays to store the min and max values of each feature
	update_min_maxes = (FLAGS.min_max_file == None)
	if(update_min_maxes):
		min_max_vals = {"max": [],"min": []}
		for layer in range(len(model.CNN_FEATURE_COUNT)):
			min_max_vals["max"].append([float("-inf")] * model.CNN_FEATURE_COUNT[layer])
			min_max_vals["min"].append([float("inf")] * model.CNN_FEATURE_COUNT[layer])
	else:
		f = np.load(FLAGS.min_max_file, allow_pickle=True)
		min_max_vals = {"max": f["max"],"min": f["min"]}

	
	convert_dataset_to_iad(list_of_files_and_labels, min_max_vals, update_min_maxes)
	normalize_dataset(list_of_files_and_labels, min_max_vals)
	if(FLAGS.c):
		combine_npy_files(list_of_files_and_labels)
		clean_up_npy_files(list_of_files_and_labels)
	else:
		make_iadlist_file(list_of_files_and_labels)

	#summarize operations
	print("Summary")
	print("--------------")
	print("Number of videos into IADs: {0}".format(len(list_of_files_and_labels)))
	print("IADs are padded/pruned to a length of: {0}".format(FLAGS.pad_length))
	print("Longest video sequence in file list: {0}".format(max_frame_length))
	print("Files place in: {0}".format(FLAGS.dst_directory))
	print("Min/Max File was Saved: {0}".format(update_min_maxes))
