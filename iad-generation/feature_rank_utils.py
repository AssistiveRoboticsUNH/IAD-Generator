import numpy as np 

def order_feature_ranks(file):
	# open file
	f = np.load(file, allow_pickle=True)
	depth, index, rank = f["depth"], f["index"], f["rank"]

	#sort data by rank
	order = rank.argsort()

	return depth[order], index[order], rank[order]

def open_feature_files(file):
	f = np.load(file, allow_pickle=True)
	return f["depth"], f["index"], f["rank"]

def get_top_n_feature_indexes(file, n):
	# open file
	depth, index, rank = open_feature_files(file)

	keep_indexes = []
	for d in np.unique(depth):

		locs = np.argwhere(depth == d)
		d_sub, i_sub, r_sub = depth[locs], index[locs], rank[locs]



		order = r_sub.reshape(-1).argsort()#[::-1]

		print(r_sub.reshape(-1))
		print(order)
		print(r_sub[order].reshape(-1))
		print('')


		d_sub, i_sub, r_sub = d_sub[order], i_sub[order], r_sub[order]

		keep_indexes.append(i_sub[:n].reshape(-1))

	return keep_indexes

def get_top_n_feature_indexes_combined(frames_file, flow_file, n):
	# open files
	depth_rgb, index_rgb, rank_rgb = open_feature_files(frames_file)
	depth_flo, index_flo, rank_flo = open_feature_files(flow_file)

	# combine frame and flow data together
	source = np.concatenate((np.zeros_like(depth_rgb), np.ones_like(depth_flo)))
	depth = np.concatenate((depth_rgb, depth_flo))
	index = np.concatenate((index_rgb, index_flo))
	rank = np.concatenate((rank_rgb, rank_flo))

	pruning_indexes = {"frames": [], "flow":[]}
	for d in np.unique(depth):

		# get only those ranks for the given depth
		locs = np.argwhere(depth == d)
		s_sub, d_sub, i_sub, r_sub = source[locs], depth[locs], index[locs], rank[locs]

		# order the ranks according to descending order from highest rank to lowest
		order = r_sub.reshape(-1).argsort()#[::-1]
		s_sub, d_sub, i_sub, r_sub = s_sub[order], d_sub[order], i_sub[order], r_sub[order]

		# get the indexes of the top ranked features
		idx = i_sub[:n].reshape(-1)

		# organize the ranks depending on whether they came from the frames dataset or 
		# the flow dataset
		pruning_indexes["frames"].append(idx[ np.where(s_sub[:n] ==  0)[0] ])
		pruning_indexes["flow"].append(idx[ np.where(s_sub[:n] ==  1)[0] ])

		print(r_sub[:n])

		print(len(pruning_indexes["frames"][-1]), len(pruning_indexes["flow"][-1]))

	return pruning_indexes


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