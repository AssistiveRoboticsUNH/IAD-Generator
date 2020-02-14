# develop IADs for dataset

python iad_generator_flex.py \
	trn \
	~/models/TRN_somethingv2_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar \
	~/datasets/Something-Something/ \
	~/datasets/Something-Something/ss.csv \
	174 \
	1 \
	~/datasets/Something-Something/iad_trn_frames_1/feature_ranks_1.npz  \
	9 \
	--num_procs=1
