# IAD-Generator
The provided files allow for the generation and evaluation of Interval Algebraic Descriptors (IADs). IADs are generated from contemporary 3D-CNN architectures, we specifically focused on C3D and I3D. The generated IADs were evaluated using an ensemble model which we provide.

## Installation
We used the following dependencies when running our code
-Tensorflow 1.14
-OpenCV 3.3.1
-NumPy 1.16.4
-Pillow 6.1.0

We used the C3D implementation provided [here](https://github.com/VisionLearningGroup/R-C3D.git). We trained this model from scratch. We used the I3D implementation provided [here](https://github.com/deepmind/kinetics-i3d.git) and trained the model using the code available [here](https://github.com/LossNAN/I3D-Tensorflow.git). Follow the installation directiosn provided at the given links before running the provided code.

The [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset and the [HMDB-51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) datasets can be downloaded from the provided links.

## Usage
Generating IADs occurs in several steps. In our system we evaluated I3D using pre-trained ImageNet features and so we omit the training descriptions for that network.

### Training C3D
If training the network using C3D first move the contents of the provided "caffe3d" folder into the caffe3d/examples/directory of the provided C3D implementation. To train the C3D network run the following code:
```
run code
```
You can evaluate the code using 
```
run code
```
Having trained the network in caffe you must convert the saved model to a tensorlfow interpretable format. Runn the given command to do so
```
run code
```

### Generating IADs
Once you have a working network you can generate IADs. To do so you need to provide the tensorflow checkpoint directory where the saved model is located, a prefix, and a list_file. The prefix defines the provided listfile as containing either training or test information in order to generate the output .iadlist correctly. The .lsiut file is a plaintext space-delimited document containing the location of the dataset files and the class label (see C3D for examples). To see other options for the code run with the -h flag.
```
python iad_generator.py <model name> train <.list file> --dst_directory <output directory>
```
The iad_generation code will output IADs into a directory listed by the --dst_directory flag. If the varibale is not set it will deposit IADs in a new directory named "generated_iads". IADs will be saved as .npy files and named the same as the input name. In addition to the IADs the iad_generation code will generate a min_maxes.npy file containing the values used to normalize the IADs. The code will also output a .iadlist file which is used by the evaluation code. Both the min_maxes.npy and the .iadlist files will be placed into the designated output directory. If you are generating IADs using for a test dataset you should use the following code that specifices the normalizing values to use in the min_maxes file
```
python iad_generator.py <model name> test <.list file> --dst_directory <output directory> --min_max_file <output directory/min_maxes.npy>
```

Note: you will need to update line 7 to specifiy whether you are using C3 D or I3D. 

### Evaluating IADs
Having generated IADs for both the training and test dataset you can evaluate them using the following codes. If the dataset is particulalry small you should be able to fit the entire ensmeble into memory. You can do so using this code.
```
python ensemble.py <model> <num_classes> <iad_dir> <prefix> <window_length>
```
The purpose of the parameters is described by running with the -h parameter. If your IADs are particualrly large however you may want to separate the ensemble into its component parts for trainign and evaluating. To do so run this code.
```
python ensemble_separate.py <model> <num_classes> <iad_dir> <prefix> <window_length>
```

ToDo: 
- [x] Write ReadMe file
- [ ] provide examples
- [ ] Provide cleaned up C3D/I3D integration
