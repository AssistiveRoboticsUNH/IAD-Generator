DEPTH = 0
CNN_FEATURE_COUNT = []
class BackBone:

	def open_model(self, max_length=8, start_idx=0):
		pass

	def predict(self, csv_input, max_length=1):
		pass

	def process(self, csv_input, max_length=1):
		pass
		#return iad_data, length_ratio

	def rank(self, csv_input, max_length=1):
		pass
		#return rank_data

	def __init__(self, checkpoint_file, num_classes, feature_idx=None):
		self.num_classes = num_classes