# develop IADs for dataset

python iad_generator_flex.py \
	i3d \
	~/models/TRN_somethingv2_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar \
	~/datasets/Something-Something/ \
	~/datasets/Something-Something/ss.csv \
	174 \
	1 \
	~/datasets/Something-Something/iad_i3d_frames_1/feature_ranks_1.npz  \
	10 \
	--num_procs=1