class BackBone:

    def open_model(self, max_length=8, start_idx=0):
        pass

    def predict(self, csv_input, max_length=1):
        pass

    def process(self, csv_input, max_length=1):
        pass
        #return iad_data, rank_data, length_ratio

    def __init__(self, checkpoint_file, num_classes):
		self.CNN_FEATURE_COUNT = []