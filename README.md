# IAD-Generator
The provided files allow for the generation and evaluation of Interval Algebraic Descriptors (IADs). IADs are generated from contemporary 3D-CNN architectures, we specifically focused on C3D and I3D. The generated IADs were evaluated using an ensemble model which we provide.

## Installation
We used the following dependencies when running our code
-Tensorflow 1.14
-OpenCV 3.3.1
-NumPy 1.16.4
-Pillow 6.1.0

We used the C3D implementation provided [here](https://github.com/VisionLearningGroup/R-C3D.git). We trained this model from scratch. We used the I3D implementation provided [here](https://github.com/deepmind/kinetics-i3d.git) and trained the model using the code available [here](https://github.com/LossNAN/I3D-Tensorflow.git). Follow the installation directiosn provided at the given links before running the provided code.

The [UCF-101]() dataset and the [HMDB-51]() dtaasets can be downloaded from the provided links.

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
Once you have a working network you can generate IADs. To do so you need to provide the tensorflow checkpoint directory where the saved model is located and a list_file. The list_file is a plaintext space-delimited document containing the location of the dataset files and the class label.
```
python iad_generator.py <model_name> <list_file>
```

### Evaluating IADs
Having generated IADs for both the training and test dataset you can evaluate them using the following codes. If the dataset is particulalry small you should be able to fit the entire ensmeble into memory. You can do so using this code.
```
run code
```
If your IADs are particualrly large however you may want to separate the ensemble into its component parts for trainign and evaluating. To do so run this code.
```
run code
```


ToDo: 
- [x] Write ReadMe file
- [ ] provide examples
- [ ] elaborate different C3D/I3D implementations
