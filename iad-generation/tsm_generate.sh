# develop IADs for dataset

python iad_generator_flex.py \
	tsm \
	~/models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth \
	~/datasets/Something-Something/ \
	~/datasets/Something-Something/ss.csv \
	174 \
	1 \
	~/datasets/Something-Something/iad_tsm_frames_1/feature_ranks_1.npz  \
	70 \
	--num_procs=2 \ 
	--gpu 1
