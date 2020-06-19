# develop IADs for dataset

python iad_generator_flex.py \
	tsm \
	~/models/saved_bottleneck_model_128.pt \
	~/datasets/Something-Something/ \
	~/datasets/Something-Something/ss.csv \
	174 \
	1 \
	70 \
	--gpu 0 \
	--num_procs 4 
