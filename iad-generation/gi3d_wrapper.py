from backbone_wrapper import BackBone

import sys, os
sys.path.append("/home/mbc2004/gluon")

import cv2
import numpy as np

import mxnet as mx
import mxnet.ndarray as F
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from mxnet import gluon, nd, gpu, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import *

from gluoncv.data.transforms import video
from gluon_i3d import i3d_resnet50_v1_sthsthv2 as model


DEPTH_SIZE = 4
CNN_FEATURE_COUNT = [256, 512, 1024, 2048]

class I3DBackBone(BackBone):
         
    def open_file(self, csv_input, start_idx=0, batch_now=True):
        
        folder_name = csv_input['raw_path']
        assert os.path.exists(folder_name), "cannot find frames folder: "+folder_name
        files = os.listdir(folder_name)

        # collect the frames
        data = []
        
        for i in range(self.max_length):
            frame = start_idx+i
            if(frame < len(files)): 
                data.append( cv2.imread(os.path.join(folder_name, files[frame])) ) 
            else:
                # fill out rest of video with blank data
                data.append( np.zeros_like(data[0], np.uint8) )

        # process the frames
        data = self.transform(data)
        print("data:", data[0].shape)

        data = np.array(data).astype(np.float32, copy=False)
        if (batch_now):
            out = data.reshape(-1, self.max_length, 3, 224,224)
            out =  np.transpose(out, [0,2,1,3,4])
            out = mx.ndarray.array(out)
            return out.copyto(self.ctx) 
        out = data.reshape(self.max_length, 3, 224,224)
        return np.transpose(out, [1,0,2,3])

    def open_file_as_batch(self, csv_input):
        
        folder_name = csv_input['raw_path']
        assert os.path.exists(folder_name), "cannot find frames folder: "+folder_name
        files = os.listdir(folder_name)

        # collect the frames
        end_frame = csv_input['length'] - (csv_input['length']%self.max_length)
        batch = [ self.open_file(csv_input, start_idx, batch_now=False) for start_idx in range(0, end_frame, 4) ]
        
        # process the frames
        return torch.stack(batch).cuda()

    def predict(self, csv_input):

        print("data_in:", data_in.shape)
        self.net.forward(is_train=False, data=data_in)

    def rank(self, csv_input):

        summed_ranks = []
        L = gluon.loss.SoftmaxCrossEntropyLoss()


        data_in = self.open_file(csv_input)
        print("data_in:", data_in.shape)

        rank_out = []

        for i in range(DEPTH_SIZE):

            self.net.record_point = i

            # record gradient information
            with ag.record():
                out = self.net(data_in)
                label_correct = mx.nd.array([csv_input["label"]]).copyto(mx.gpu(0))
                loss = L(out, label_correct)

            # do backward pass
            
            # calculate Taylor Expansion for network
            layers = self.net.activation_points
            out.backward()

            print(len(layers), i)

            for j in range(1):
                l = layers[i]

                try:

                    activation = l[0].asnumpy()
                    gradient = l.grad[0].asnumpy()

                    print(activation.shape)
                    
                    rank = np.multiply(activation, gradient)
                    rank_norm_size = rank.shape[1]*rank.shape[2]*rank.shape[3]
                    rank = np.sum(rank, axis = (1,2,3)) / float(rank_norm_size)

                    rank_out.append(rank)
                except:
                    # we need to skip the first asnumpy call on the activation to prevent the 
                    # asnumpy conversion error
                    print("")

        return rank_out

  
   
    def process(self, csv_input):

        data_in = self.open_file(csv_input)
        length_ratio = csv_input['length']/float(self.max_length)

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)

        print("data_in:", data_in.shape)
        
        # pass data through network to obtain activation maps
        # rst is not used and not need to store grads

        self.net(data_in)
        activations = self.net.activation_points

        for i in range(len(activations)):
            # convert actvitaion from PyTorch to Numpy
            activations[i] = activations[i].asnumpy()

            print("a0:", activations[i].shape)
            # prune low-quality filters
            activations[i] = activations[i][:, self.feature_idx[i], :, :]
            print("a1:", activations[i].shape)

            # compress spatial dimensions
            activations[i] = np.max(activations[i], axis=(2,3))
            print("a2:", activations[i].shape)
            activations[i] = activations[i].T
            print("a3:", activations[i].shape)
        
        return activations, length_ratio

    def __init__(self, checkpoint_file, num_classes, max_length=16, feature_idx=None, gpu=0):
        self.is_shift = None
        self.net = None
        self.arch = None
        self.num_classes = num_classes
        self.max_length = max_length
        self.feature_idx = feature_idx

        self.transform = None

        # get data
        image_norm_mean = [0.485, 0.456, 0.406]
        image_norm_std = [0.229, 0.224, 0.225]
        
        self.transform = video.VideoGroupValTransform(size=224, mean=image_norm_mean, std=image_norm_std)
        self.ctx = mx.gpu(gpu)

        net = model(nclass=self.num_classes, pretrained=False, num_segments=1, num_crop=1, feat_ext=False)
        
        net.cast('float32')
        net.collect_params().reset_ctx([self.ctx])

        self.net = net
