from backbone_wrapper import BackBone

import sys, os
sys.path.append("/home/mbc2004/TRN-pytorch")

import torch.nn as nn
from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule
import datasets_video
import pdb

import numpy as np
from PIL import Image

class TRNBackBone(BackBone):
         
    def open_file(self, csv_input, start_idx=0, batch_now=True):
        
        folder_name = csv_input['raw_path']
        assert os.path.exists(folder_name), "cannot find frames folder: "+folder_name
        files = os.listdir(folder_name)

        # collect the frames
        data = []
        
        for i in range(self.max_length):
            frame = start_idx+i
            if(frame < len(files)): 
                data.append( Image.open(os.path.join(folder_name, files[frame])).convert('RGB') ) 
            else:
                # fill out rest of video with blank data
                data.append( Image.new('RGB', (data[0].width, data[0].height)) )

        # process the frames
        data = self.transform(data)
        if (batch_now):
            return data.view(-1, self.max_length, 3, 256,256)
        return data.view(self.max_length, 3, 256,256)


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

        data_in = self.open_file_as_batch(csv_input)

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)

        print("data_in:", data_in.shape)
        
        # predict value
        with torch.no_grad():
            return self.net(data_in)

    def rank(self, csv_input):

        summed_ranks = []

        end_frame = csv_input['length'] - (csv_input['length']%self.max_length)
        for i in range(0, end_frame, 4):
            data_in = self.open_file(csv_input, start_idx = i)#self.open_file_as_batch(csv_input)

            # data has shape (batch size, segment length, num_ch, height, width)
            # (6,8,3,256,256)

            #print("data_in:", data_in.shape)
            
            # pass data through network to obtain activation maps
            # do need grads for taylor expansion
            rst = self.net(data_in)

            # compute gradient and do SGD step
            self.loss(rst, torch.tensor( [csv_input['label']]*data_in.size(0) ).cuda() ).backward()

            for j, rd in enumerate(self.ranks):
                if(i == 0):
                    summed_ranks.append(rd)
                else:
                    summed_ranks[j] = np.add(summed_ranks[j], rd)

        return summed_ranks

    def process(self, csv_input):

        data_in = self.open_file(csv_input)
        length_ratio = csv_input['length']/float(self.max_length)

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)

        print("data_in:", data_in.shape)
        
        # pass data through network to obtain activation maps
        # rst is not used and not need to store grads
        with torch.no_grad():
            rst = self.net(data_in)

            for i in range(len(self.activations)):
                # convert actvitaion from PyTorch to Numpy
                self.activations[i] = self.activations[i].cpu().numpy()

                # prune low-quality filters
                self.activations[i] = self.activations[i][:, self.feature_idx[i], :, :]

                # compress spatial dimensions
                self.activations[i] = np.max(self.activations[i], axis=(2,3))
                self.activations[i] = self.activations[i].T

        return self.activations, length_ratio

    def __init__(self, checkpoint_file, num_classes, max_length=25, feature_idx=None):
        self.is_shift = None
        self.net = None
        self.arch = None
        self.num_classes = num_classes
        self.max_length = max_length
        self.feature_idx = feature_idx

        self.transform = None

        self.CNN_FEATURE_COUNT = [256, 512, 1024, 2048]

        #checkpoint_file = TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth
        #checkpoint_file = TRN_somethingv2_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar

        modality = 'RGB'
        crop_fusion_type = 'TRNmultiscale'
        net = TSN(self.num_classes, 25, modality,
                  base_model=self.arch,
                  consensus_type=crop_fusion_type,
                  img_feature_dim=256
                  )
        print(net.base_model)

        # load checkpoint file
        checkpoint = torch.load(this_weights)

        # add activation and ranking hooks
        self.activations = [None]*4
        self.ranks = [None]*4
        def activation_hook(idx):
            def hook(model, input, output):
                #prune features and only get those we are investigating 
                activations = output.detach()
                
                self.activations[idx] = activations
 
            return hook

        def taylor_expansion_hook(idx):
            def hook(model, input, output):
                # perform taylor expansion
                grad = input[0].detach()
                activation = self.activations[idx]
                
                # sum values together
                values = torch.sum((activation * grad), dim = (0,2,3)).data

                # Normalize the rank by the filter dimensions
                values = values / (activation.size(0) * activation.size(2) * activation.size(3))

                self.ranks[idx] = values.cpu().numpy()

            return hook

        # Will always need the activations (whether for out or for ranking)
        net.base_model.layer1.register_forward_hook(activation_hook(0))
        net.base_model.layer2.register_forward_hook(activation_hook(1))
        net.base_model.layer3.register_forward_hook(activation_hook(2))
        net.base_model.layer4.register_forward_hook(activation_hook(3))

        if(self.feature_idx == None):
            # Need to get rank information
            net.base_model.layer1.register_backward_hook(taylor_expansion_hook(0))
            net.base_model.layer2.register_backward_hook(taylor_expansion_hook(1))
            net.base_model.layer3.register_backward_hook(taylor_expansion_hook(2))
            net.base_model.layer4.register_backward_hook(taylor_expansion_hook(3))
        else:
            # Need to shorten network so that base_model doesn't get to FC layers
            net.base_model.fc = nn.Identity()
        
        # Combine network together so that the it can have parameters set correctly
        # I think, I'm not 100% what this code section actually does and I don't have 
        # the time to figure it out right now
        checkpoint = checkpoint['state_dict']
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                        'base_model.classifier.bias': 'new_fc.bias',
                        }
        for k, v in replace_dict.items():
            if k in base_dict:
                base_dict[v] = base_dict.pop(k)

        net.load_state_dict(base_dict)
        
        # define image modifications
        self.transform = torchvision.transforms.Compose([
                           torchvision.transforms.Compose([
                                GroupScale(net.scale_size),
                                GroupCenterCrop(net.scale_size),
                            ]),
                           #torchvision.transforms.Compose([ GroupFullResSample(net.scale_size, net.scale_size, flip=False) ]),
                           Stack(roll=(self.arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(self.arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                           ])

        # place net onto GPU and finalize network
        net = torch.nn.DataParallel(net.cuda())
        net.eval()

        # network variable
        self.net = net

        # loss variable (used for generating gradients when ranking)
        if(self.feature_idx == None):
            self.loss = torch.nn.CrossEntropyLoss().cuda()
