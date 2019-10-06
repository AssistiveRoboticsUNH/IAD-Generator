
#https://jacobgil.github.io/deeplearning/pruning-deep-learning

import tensorflow as tf 
import numpy as np 

import i3d_wrapper as model

import argparse
parser = argparse.ArgumentParser(description='Generate IADs from input files')
#required command line args
parser.add_argument('model_file', help='the tensorflow ckpt file used to generate the IADs')
parser.add_argument('dataset_file', help='the *.list file than contains the ')

parser.add_argument('--pad_length', nargs='?', type=int, default=-1, help='length to pad/prune the videos to, default is padd to the longest file in the dataset')
parser.add_argument('--output_file', nargs='?', default="feature_ranks.npz", help='the output file')

FLAGS = parser.parse_args()

#set up input files
batch_size=1
list_of_files_and_labels, max_frame_length = model.obtain_files(FLAGS.dataset_file)
if(FLAGS.pad_length < 0):
	FLAGS.pad_length = max_frame_length
input_placeholder = model.get_input_placeholder(batch_size, num_frames=FLAGS.pad_length)
	
@tf.custom_gradient
def rank_layer(x):
	def grad(dy):

		#calculate the rank
		ranks = tf.math.multiply(x, dy)
		ranks = tf.reduce_sum(ranks, axis=(0, 1, 2, 3)) #combine spatial and temporal points together

		#normalize the rank by the input size
		norm_term = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), tf.float32)
		ranks = tf.math.divide(ranks, norm_term) 

		ranks_out.append(ranks)

		return dy
	return tf.identity(x), grad

# add ranking layer to model
ranks_out = []
def generate_full_model(input_ph):
	target_layers = model.generate_activation_map(input_ph)

	for l in range(len(target_layers)):
		target_layers[l] = rank_layer(target_layers[l])

	return target_layers

# generate the ranking tensors. These tensors are only generated when the 
# gradients function is called and they are added in reverse, so I rotate
# the order that the rankings are presented.
_ = generate_full_model(input_placeholder)
ranks_out = ranks_out[::-1]

# define restore variables
variable_name_list = model.get_variables()
print("variable_name_list: ", variable_name_list.keys())
saver = tf.train.Saver(variable_name_list.values())

total_ranks = None

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	# Restore from file checkpoint
	ckpt = tf.train.get_checkpoint_state(FLAGS.model_file)
	if ckpt and ckpt.model_checkpoint_path:
		print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("load complete!")
	elif os.path.exists(FLAGS.model_file):
		print("loading checkpoint file: "+FLAGS.model_file)
		saver.restore(sess, FLAGS.model_file)	
	else:
		print("Failed to Load model: "+FLAGS.model_file)
		sys.exit(1)

	sess.graph.finalize()

	# parse each file in the input directory through the network to get the node ranks
	for i in range(len(list_of_files_and_labels)):
		print("file: {:6d}/{:6d}".format(i, len(list_of_files_and_labels)))

		file, label = list_of_files_and_labels[i]

		raw_data, length_ratio = model.read_file(file, input_placeholder)

		r = sess.run([ranks_out], feed_dict={input_placeholder: raw_data})
		if(total_ranks == None):
			total_ranks = r
		else:
			total_ranks = np.add(total_ranks, r)

# store rankings in a npy array
total_ranks = total_ranks[0]
depth, index, rank = [],[],[] 
for i in range(len(total_ranks)):
	depth.append(np.full(len(total_ranks[i]), i))
	index.append(np.arange(len(total_ranks[i])))
	rank.append(total_ranks[i])
depth = np.concatenate(depth)
index = np.concatenate(index)
rank = np.concatenate(rank)

np.savez(FLAGS.output_file, depth=depth, index=index, rank=rank)

from feature_rank_utils import view_feature_rankings

view_feature_rankings(FLAGS.output_file)

