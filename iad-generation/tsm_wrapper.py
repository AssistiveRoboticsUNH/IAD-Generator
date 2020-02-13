from backbone_wrapper import BackBone

import sys, os
sys.path.append("/home/mbc2004/temporal-shift-module")

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config

import numpy as np
from PIL import Image


class TSMBackBone(BackBone):
    '''
    class NetworkWrapper(TSN):
        def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
            TSN.__init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False)

        def forward(self, input, no_reshape=False):
            
            if not no_reshape:
                sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

                if self.modality == 'RGBDiff':
                    sample_len = 3 * self.new_length
                    input = self._get_diff(input)

                base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
            else:
                base_out = self.base_model(input)

            if self.dropout > 0:
                base_out = self.new_fc(base_out)

            if not self.before_softmax:
                base_out = self.softmax(base_out)

            if self.reshape:
                if self.is_shift and self.temporal_pool:
                    base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
                else:
                    base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
                output = self.consensus(base_out)
                return output.squeeze(1)
    '''      

    def open_file(self, folder_name, max_length=8, start_idx=0):
        
        max_length = 8
        assert os.path.exists(folder_name), "cannot find frames folder: "+folder_name

        # collect the frames
        data = []
        for frame in os.listdir(folder_name)[start_idx:start_idx+max_length]:
            data.append(Image.open(os.path.join(folder_name, frame)).convert('RGB')) 

        print(data)

        # process the frames
        data = self.transform(data)
        print("data.shape:", data.shape)
        return data.view(-1, max_length, 3, 256,256)

    def predict(self, csv_input, max_length=8):


        data_in = self.open_file(csv_input['raw_path'], max_length=8)

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)

        print("data_in:", data_in.shape)
        
        # predict value
        with torch.no_grad():
            return self.net(data_in)

    def process(self, csv_input, max_length=8):

        data_in = self.open_file(csv_input['raw_path'], max_length=8)
        length_ratio = csv_input['length']/float(max_length)

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)

        print("data_in:", data_in.shape)
        
        # pass data through network to obtain activation maps
        # rst is not used and not need to store grads
        with torch.no_grad():
            rst = self.net(data_in)

        return self.activations, length_ratio

    def rank(self, csv_input, max_length=8):
        data_in = self.open_file(csv_input['raw_path'], max_length=8)

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)

        print("data_in:", data_in.shape)
        
        # pass data through network to obtain activation maps
        # do need grads for taylor expansion
        rst = self.net(data_in)
        # compute gradient and do SGD step
        self.loss(rst, torch.tensor(csv_input['label'])).backward()

        print(len(self.activations), len(self.ranks))
        for i in range(len(self.activations)):

            print("activ: {0}, grad: {1}".format(self.activations[i].shape, self.ranks[i].shape))

        return self.ranks


    def __init__(self, checkpoint_file, num_classes, features_kept=None):
        self.is_shift = None
        self.net = None
        self.arch = None

        self.transform = None

        self.CNN_FEATURE_COUNT = []

        #checkpoint_file = TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth

        # input variables
        this_weights = checkpoint_file
        this_test_segments = 8
        test_file = None

        #model variables
        self.is_shift, shift_div, shift_place = True, 8, 'blockres'

        
        self.arch = this_weights.split('TSM_')[1].split('_')[2]
        modality = 'RGB'
        

        # dataset variables
        num_class, train_list, val_list, root_path, prefix = dataset_config.return_dataset('somethingv2', modality)
        print('=> shift: {}, shift_div: {}, shift_place: {}'.format(self.is_shift, shift_div, shift_place))

        # define model
        net = TSN(num_class, this_test_segments if self.is_shift else 1, modality,
                  base_model=self.arch,
                  consensus_type='avg',
                  img_feature_dim=256,
                  pretrain='imagenet',
                  is_shift=self.is_shift, shift_div=shift_div, shift_place=shift_place,
                  non_local='_nl' in this_weights,
                  )

        # load checkpoint file
        checkpoint = torch.load(this_weights)

        # add activation and ranking hooks
        self.activations = []
        self.ranks = []
        def activation_hook():
            def hook(model, input, output):
                #prune features and only get those we are investigating 
                activations = output.detach()
                if(features_kept):
                    pass # do stuff
                self.activations.append( activations ) 
 
            return hook

        def taylor_expansion_hook():
            print("here1")
            def hook(model, input, output):
                # perform taylor expansion
                print("here2")
                grad = input.detach()

                self.ranks.append( grad ) 


                '''
                activation_index = len(self.activations) - self.grad_index - 1
                activation = self.activations[activation_index]
                values = \
                    torch.sum((activation * grad), dim = 0).\
                        sum(dim=2).sum(dim=3)[0, :, 0, 0].data
                
                # Normalize the rank by the filter dimensions
                values = \
                    values / (activation.size(0) * activation.size(2) * activation.size(3))

                if activation_index not in self.filter_ranks:
                    self.filter_ranks[activation_index] = \
                        torch.FloatTensor(activation.size(1)).zero_().cuda()

                self.filter_ranks[activation_index] += values
                self.grad_index += 1
                '''

            return hook

        
        net.base_model.layer1.register_forward_hook(activation_hook())
        net.base_model.layer2.register_forward_hook(activation_hook())
        net.base_model.layer3.register_forward_hook(activation_hook())
        net.base_model.layer4.register_forward_hook(activation_hook())

        print(features_kept, features_kept == None)
        if(features_kept == None):
            net.base_model.layer1.register_backward_hook(taylor_expansion_hook())
            net.base_model.layer2.register_backward_hook(taylor_expansion_hook())
            net.base_model.layer3.register_backward_hook(taylor_expansion_hook())
            net.base_model.layer4.register_backward_hook(taylor_expansion_hook())
        
        # modify network so that...
        print("checkpoint.keys()", checkpoint.keys())

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
                           GroupNormalize(net.input_mean, net.input_std)])

        # place net onto GPU and finalize network
        net = torch.nn.DataParallel(net.cuda())
        net.eval()

        # network variable
        self.net = net

        self.loss = torch.nn.CrossEntropyLoss().cuda()
