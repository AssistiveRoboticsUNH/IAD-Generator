import numpy as np
import fastdtw
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt

def view_iad(file):
	pass

def pre_processing(iads, length):
	unpadded_data = []
	for i in range(len(iads)):
		unpadded_data.append(iads[i, :, :length[i]])

	return iads#unpadded_data

def similarity_metric(iad, other_iad):
	dist, path = fastdtw.fastdtw(iad, other_iad, dist=euclidean)
	return dist

def assess_similarity(file, pre_processing_func, similarity_metric):
	# generate a similarity heatmap for IADs from a specific depth

	f = np.load(file)
	data, label, length = f["data"], f["label"], f["length"]

	'''
	print("COMPARISSON IS LIMITED")
	n=30
	data = data[:n]
	label = label[:n]
	length = length[:n]
	'''

	num_classes = np.max(label)+1

	#apply any pre-processing steps
	data = pre_processing_func(data, length)
	
	similarity_score = np.zeros((num_classes, num_classes), dtype=np.float64)
	counts = np.zeros((num_classes, num_classes))

	#determine the similarity between all of the IADs in teh training dataset
	for i in range(len(data)):
		print(i)
		cur_data, cur_label = data[i], label[i]

		for j in range(i+1, len(data)):

			other_data, other_label = data[j], label[j]

			similarity_score[other_label, cur_label] += similarity_metric(cur_data, other_data)
			counts[other_label, cur_label] += 1

	#normalize by the number of comparisons that were made
	#counts[np.argwhere(counts == 0)] = 1
	similarity_score /= counts
	np.nan_to_num(similarity_score, 0)

	#fill in rest of matrix
	similarity_score += np.flipud(np.rot90(np.tril(similarity_score, -1)))
	
	return similarity_score

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	parser.add_argument('input_file', help='npz with feature ranks')

	FLAGS = parser.parse_args()

	#view_iad(FLAGS.input_file)

	matrix = assess_similarity(FLAGS.input_file, pre_processing, similarity_metric)	
	print(matrix)

	fig, ax = plt.subplots()

	
	
	im = ax.imshow(matrix, cmap='pink')
	ax.set_xticks(np.arange(len(matrix)))
	ax.set_yticks(np.arange(len(matrix)))

	ax.set_xticklabels(np.arange(len(matrix)))
	ax.set_yticklabels(np.arange(len(matrix)))

	ax.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)

	ax.figure.colorbar(im, ax=ax, cmap='pink')

	normalized_vals = 1-(matrix/np.max(matrix))

	for i in range(len(matrix)):
		for j in range(len(matrix)):
			#color = 'w'
			#if(round(normalized_vals[i, j] < .5)):
			color ='k'
			text = ax.text(j, i, round(normalized_vals[i, j], 2),ha="center", va="center", color=color)

	plt.show()