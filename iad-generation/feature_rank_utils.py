import numpy as np 

def order_feature_ranks(file):
	# open file
	f = np.load(file, allow_pickle=True)
	depth, index, rank = f["depth"], f["index"], f["rank"]

	#sort data by rank
	order = rank.argsort()

	return depth[order], index[order], rank[order]

def view_feature_rankings(file):
	depth, index, rank = order_feature_ranks(file)

	#define color on layer
	colors = ['r', 'g', 'b', 'y', 'k']
	c = []
	for i in depth:
		c.append(colors[int(i)])

	import matplotlib.pyplot as plt 

	plt.scatter(np.arange(len(rank)), rank, facecolors='none', edgecolors=c)
	plt.show()

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	parser.add_argument('input_file', help='npz with feature ranks')
	FLAGS = parser.parse_args()

	view_feature_rankings(FLAGS.input_file)