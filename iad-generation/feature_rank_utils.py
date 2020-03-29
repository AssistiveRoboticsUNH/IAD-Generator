import numpy as np 
np.set_printoptions(suppress=True)

from scipy.stats import rankdata

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

		order = r_sub.reshape(-1).argsort()

		d_sub, i_sub, r_sub = d_sub[order], i_sub[order], r_sub[order]
		d_sub, i_sub, r_sub = d_sub[::-1], i_sub[::-1], r_sub[::-1]

		#print(r_sub)

		keep_indexes.append(i_sub[:n].reshape(-1))

	return keep_indexes

def get_top_n_feature_indexes_combined(frames_file, flow_file, n, weights=np.ones((2,5))):
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
		

		#print(weights[0][d])
		#print(r_sub[np.argwhere(s_sub==0)][:10].reshape(-1))

		#print(r_sub[np.where(s_sub==0)][:10])
		#print(r_sub[np.where(s_sub==1)][:10])
		
		rgb_max, rgb_min = np.max(r_sub[np.where(s_sub==0)]), np.min(r_sub[np.where(s_sub==0)])
		flow_max, flow_min = np.max(r_sub[np.where(s_sub==1)]), np.min(r_sub[np.where(s_sub==1)])

		#print(rgb_max, rgb_min)
		#print(flow_max, flow_min)

		r_sub[np.where(s_sub==0)] -= rgb_min
		r_sub[np.where(s_sub==0)] /= (rgb_max - rgb_min)

		r_sub[np.where(s_sub==1)] -= flow_min
		r_sub[np.where(s_sub==1)] /= (flow_max - flow_min)

		#print('--------')

		#print(r_sub[np.where(s_sub==0)][:10])
		#print(r_sub[np.where(s_sub==1)][:10])


		r_sub[np.where(s_sub==0)] *= weights[0][d]
		r_sub[np.where(s_sub==1)] *= weights[1][d]

		#print('--------')

		#print(r_sub[np.where(s_sub==0)][:10])
		#print(r_sub[np.where(s_sub==1)][:10])

		#print(r_sub[np.argwhere(s_sub==0)][:10].reshape(-1))
		#print('==============')

		order = r_sub.reshape(-1).argsort()#[::-1]
		s_sub, d_sub, i_sub, r_sub = s_sub[order], d_sub[order], i_sub[order], r_sub[order]
		s_sub, d_sub, i_sub, r_sub = s_sub[::-1],  d_sub[::-1],  i_sub[::-1],  r_sub[::-1]

		# get the indexes of the top ranked features
		idx = i_sub[:n].reshape(-1)

		# organize the ranks depending on whether they came from the frames dataset or 
		# the flow dataset

		pruning_indexes["frames"].append(idx[ np.where(s_sub[:n] ==  0)[0] ])
		pruning_indexes["flow"].append(idx[ np.where(s_sub[:n] ==  1)[0] ])

		print(d, len(pruning_indexes["frames"][d]), len(pruning_indexes["flow"][d]))

		print(len(pruning_indexes["frames"][-1]), len(pruning_indexes["flow"][-1]))

	return pruning_indexes


def view_feature_rankings(file):
	depth, index, rank = order_feature_ranks(file)

	loc = depth == 3
	depth = depth[loc][::-1]
	index = index[loc][::-1]
	rank = rank[loc][::-1]

	for i in range(50):
		print(depth[i], index[i], rank[i])

	end = file.split('/')[-1][:-4]+"_chk.npz"
	print("end: ", end)
	np.save(end, index[:50])

	'''
	#define color on layer
	colors = ['r', 'g', 'b', 'y', 'k']
	c = []
	for i in depth:
		c.append(colors[int(i)])

	import matplotlib.pyplot as plt 

	plt.scatter(np.arange(len(rank)), rank, facecolors='none', edgecolors=c)
	plt.show()
	'''

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	parser.add_argument('input_file', help='npz with feature ranks')
	FLAGS = parser.parse_args()

	view_feature_rankings(FLAGS.input_file)