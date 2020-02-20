python rank_features_mt.py \
	tsm \
	~/models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth \
	~/datasets/Something-Something/ \
	~/datasets/Something-Something/ss.csv \
	174 \
	1 \
	--dataset_size=50000 \
	--num_procs=2 \