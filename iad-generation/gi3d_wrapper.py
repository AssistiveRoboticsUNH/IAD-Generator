from backbone_wrapper import BackBone

import sys, os
sys.path.append("/home/mbc2004/gluon")
'''
import torch.nn as nn
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
'''
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
from gluoncv.data import UCF101, Kinetics400, SomethingSomethingV2, HMDB51
#from gluoncv.model_zoo import get_model
from gluon_i3d import i3d_resnet50_v1_sthsthv2 as model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load



import numpy as np
from PIL import Image

depth_size = 5

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

        #data_in = self.open_file_as_batch(csv_input)

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)

        print("data_in:", data_in.shape)
        
        # predict value
        #with torch.no_grad():
        #    return self.net(data_in)
        self.net.forward(is_train=False, data=data_in)

    def rank(self, csv_input):

        summed_ranks = []
        #L = gluon.loss.SoftmaxCrossEntropyLoss()


        data_in = self.open_file(csv_input)
        #data_in = self.open_file(csv_input, start_idx = i)
        print("data_in:", data_in.shape)


        # record gradient information
        with ag.record(train_mode=False):
            out = self.net(data_in)


        # do backward pass
        one_hot_target = mx.nd.one_hot(mx.nd.array([csv_input["label"]]), self.num_classes)
        out.backward(one_hot_target, train_mode=False)
        
        # calculate Taylor Expansion for network
        layers = self.net.activation_points
        rank_out = []
        for i, l in enumerate(layers):
            #print(type(l[0]), type(l.grad[0]))
            #activ = l[0]
            #grad = l.grad[0]

            print(l[0].shape, l.grad[0].shape)
            print("activation1", type(l[0]), "grad", type(l.grad[0]))
            print("activation1", l[0].dtype, "grad", l.grad[0].dtype)

            #print(l[0])

            mul = mx.nd.sum(mx.nd.multiply(l[0], l.grad[0]), axis = (1,2,3))
            print(mul.shape)
            mul.copyto(mx.cpu(0))
            mx.nd.save('rank_values', mul)
            #print(mul)



            #activation = l[0].asnumpy()
            '''
            gradient = l.grad[0].asnumpy()

            print("activation2", type(activation), "grad", type(gradient))

            rank = np.multiply(activation, gradient)
            print("rank1:", rank.shape)
            rank_norm_size = rank.shape[1]*rank.shape[2]*rank.shape[3]
            rank = np.sum(rank, axis = (1,2,3)) / float(rank_norm_size)
            print("rank2:", rank.shape)
            '''
            rank = []
            rank_out.append(rank)
        
        print("return ranks here")
        print (rank_out)
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
            activations[i] = activations[i].cpu().numpy()

            # prune low-quality filters
            activations[i] = activations[i][:, self.feature_idx[i], :, :]

            # compress spatial dimensions
            activations[i] = np.max(activations[i], axis=(2,3))
            activations[i] = activations[i].T
        
        return activations, length_ratio

    def __init__(self, checkpoint_file, num_classes, max_length=16, feature_idx=None, gpu=0):
        self.is_shift = None
        self.net = None
        self.arch = None
        self.num_classes = num_classes
        self.max_length = max_length
        self.feature_idx = feature_idx

        self.transform = None

        self.CNN_FEATURE_COUNT = [256, 512, 1024, 2048]

        #checkpoint_file = TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth

        # input variables
        #this_weights = checkpoint_file
        #this_test_segments = self.max_length
        #test_file = None

        #model variables
        #self.is_shift, shift_div, shift_place = True, 8, 'blockres'

        
        #self.arch = this_weights.split('TSM_')[1].split('_')[2]
        modality = 'RGB'
        

        # dataset variables
        #num_class, train_list, val_list, root_path, prefix = dataset_config.return_dataset('somethingv2', modality)
        #print('=> shift: {}, shift_div: {}, shift_place: {}'.format(self.is_shift, shift_div, shift_place))

        # define model
        





        '''
        opt = parse_args()
        print(opt)

        # Garbage collection, default threshold is (700, 10, 10).
        # Set threshold lower to collect garbage more frequently and release more CPU memory for heavy data loading.
        gc.set_threshold(100, 5, 5)

        # set env
        num_gpus = opt.num_gpus
        batch_size = opt.batch_size
        context = [mx.cpu()]
        if num_gpus > 0:
            batch_size *= max(1, num_gpus)
            context = [mx.gpu(i) for i in range(num_gpus)]
        
        num_workers = opt.num_workers
        print('Total batch size is set to %d on %d GPUs' % (batch_size, num_gpus))
        '''
        # get data
        image_norm_mean = [0.485, 0.456, 0.406]
        image_norm_std = [0.229, 0.224, 0.225]
        
        self.transform = video.VideoGroupValTransform(size=224, mean=image_norm_mean, std=image_norm_std)
        self.ctx = mx.gpu(gpu)





        
        #net = get_model(name=model_name, nclass=classes, pretrained=opt.use_pretrained, num_segments=opt.num_segments, num_crop=opt.num_crop)
        net = model(nclass=self.num_classes, pretrained=False, num_segments=1, num_crop=1, feat_ext=True)
        
        net.cast('float32')
        net.collect_params().reset_ctx([self.ctx])

        #net.hybridize(static_alloc=True, static_shape=True)
        



        self.net = net






        


        """
        print(net.res_layers)
        print('---------')
        print(net.res_layers[0][2].bottleneck[0])
        

        layers = [
            net.res_layers[0][2].bottleneck[0],
            #net.res_layers[1][3].bottleneck,
            #net.res_layers[2][5].bottleneck,
            #net.res_layers[3][2].bottleneck,
        ]

        self.activations = []
        self.ranks = []

        def activation_hook(idx):
            print("add hook:")

            def hook(model, input, output):
                #prune features and only get those we are investigating 
                #activation = output#.detach()
                #print("activation:", activation.shape)
                #self.activations[idx] = activation
                print("in_function", input[0].get_params(), output.get_params())
                

                sym, arg_params, aux_params = mx.model.load_checkpoint('/home/mbc2004/gluon/gluon_i3d', 0)

                activ = mx.mod.Module(symbol=output, label_names=None, context=mx.gpu())
                activ.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
                activ.set_params(arg_params, aux_params)

                self.activ = activ
                

 
            return hook

        def taylor_expansion_hook(idx):
            def hook(model, input, output):
                # perform taylor expansion
                grad = output[0].detach()
                activation = self.activations[idx]

                #print("activ: ", activation.shape)
                #print("grad: ", grad.shape)
                
                # sum values together
                values = torch.sum((activation * grad), dim = (0,2,3)).data

                # Normalize the rank by the filter dimensions
                values = values / (activation.size(0) * activation.size(2) * activation.size(3))

                self.ranks[idx] = values.cpu().numpy()

            return hook

        sym, arg_params, aux_params = mx.model.load_checkpoint('/home/mbc2004/gluon/gluon_i3d', 0)

        
        print("arg_params:")
        for k in arg_params.keys():
            print(k)

        print("aux_params:")
        for k in aux_params.keys():
            print(k)

        #print(type(layers[0].get_params()))
        """
        """
        for idx, layer in enumerate(layers):
            self.activations.append([])
            self.ranks.append([])

            # Will always need the activations (whether for out or for ranking)
            layer.register_forward_hook(activation_hook(idx))
            print("add hook here:", layer)
            #if(self.feature_idx == None):
                # Need to get rank information
                #layer.register_backward_hook(taylor_expansion_hook(idx))
        """
        """












        '''
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
        '''
        # network variable
        self.net = net

        # loss variable (used for generating gradients when ranking)
        #if(self.feature_idx == None):
        #    self.loss = torch.nn.CrossEntropyLoss().cuda()
        """

        """
        path = "/home/mbc2004/gluon/"
        net = gluon.nn.SymbolBlock.imports(path+"gluon_i3d-symbol.json", ['data'], path+"gluon_i3d-0000.params", ctx=self.ctx)

        all_layers = net.symbol.get_internals()
        
        self.output_layer_you_want = all_layers['output_layer_name']

        #new_model = mx.model.create()  # use this api, pass necessary symbol(the output_layer_you_want and aug_param, etc)
        #new_model.predict()

        print(all_layers)

        self.net = net
        """