# IAD-Generator
The provided files allow for the generation and evaluation of Interval Algebraic Descriptors (IADs). IADs are generated from contemporary 3D-CNN architectures, we specifically focused on C3D and I3D. The generated IADs were evaluated using an ensemble model which we provide.

## Installation
We used the following dependencies when running our code
-Tensorflow 1.14
-OpenCV 3.3.1
-NumPy 1.16.4
-Pillow 6.1.0

We used the C3D implementation provided [here](https://github.com/hx173149/C3D-tensorflow.git). We trained this model from scratch.

We used the I3D implementation provided [here](https://github.com/deepmind/kinetics-i3d.git) and trained the model using the code available [here](https://github.com/LossNAN/I3D-Tensorflow.git).

## Usage
Generating IADs occurs in several steps. The main piplieine occurs as follows: 
1. _Optional_: Train a 3D-CNN
2. Use pretrained features to generate IADs
3. Evaluate the IADs using the provided ensemble code. 

### Training C3D/I3D
When training 

### Generating IADs


### Evaluating IADs

To execute run: python iad_gen.py

you will need to set the variables for the threshold options and compression options

python iad_generator.py ~/i3d/train_i3d/experiments/ucf-101/models/ucf_i3d_pretrained_01_100/model.ckpt ~/datasets/UCF-101/listFiles/trainlist01_100.list


ToDo: imporve ReadMe file, include examples
