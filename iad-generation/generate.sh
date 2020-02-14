# develop IADs for training dataset
python iad_generator_flex.py \
	tsm \
	~/temporal-shift-module/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth \
	~/datasets/Something-Something/ \
	~/datasets/Something-Something/ss.csv \
	174 \
	1 \
	~/datasets/Something-Something/iad_frames_1/feature_ranks_1.npz  \
	100 \
	--num_procs=4

# develop IADs for test dataset, 
python iad_generator_flex.py \
	tsm \
	~/temporal-shift-module/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth \
	~/datasets/Something-Something/ \
	~/datasets/Something-Something/ss.csv \
	174 \
	0 \
	~/datasets/Something-Something/iad_frames_1/feature_ranks_1.npz  \
	100 \
	--num_procs=4