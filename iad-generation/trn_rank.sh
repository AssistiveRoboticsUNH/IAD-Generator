python rank_features_mt.py \
	trn \
	~/models/TRN_somethingv2_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar \
	~/datasets/Something-Something/ \
	~/datasets/Something-Something/ss.csv \
	174 \
	1 \
	--dataset_size=50000 \
	--num_procs=2 \
