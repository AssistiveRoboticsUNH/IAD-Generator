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

    def open_file(self, folder_name, max_length=8, start_idx=0):
        
        assert os.path.exists(folder_name), "cannot find frames folder: "+folder_name

        # collect the frames
        data = []
        for frame in os.listdir(folder_name)[start_idx:start_idx+max_length]:
            data.append(Image.open(os.path.join(folder_name, frame)).convert('RGB')) 

        # process the frames
        data = self.transform(data)
        print("data.shape:", data.shape)
        return data.view(-1, max_length, 3, 256,256)

    def predict(self, csv_input, max_length=8):


        data_in = self.open_file(csv_input['raw_path'], max_length=8)

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)

        print("data_in:", data_in.shape)
        
        with torch.no_grad():

            # predict value
            rst = self.net(data_in)
            #rst = rst.reshape(batch_size, num_crop, -1).mean(1)


        return rst

    def process(self, csv_input, max_length=8):

        data_in = self.open_file(csv_input['raw_path'], max_length=8)

        pass
        #return iad_data, rank_data, length_ratio


    def __init__(self, checkpoint_file, num_classes):
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
        '''
        def parse_shift_option_from_log_name(log_name):
            if 'shift' in log_name:
                strings = log_name.split('_')
                for i, s in enumerate(strings):
                    if 'shift' in s:
                        break
                return True, int(strings[i].replace('shift', '')), strings[i + 1]
            else:
                return False, None, None
        '''
        self.is_shift, shift_div, shift_place = True, 8, 'blockres'#parse_shift_option_from_log_name(this_weights)

        
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
                           torchvision.transforms.Compose([ GroupFullResSample(net.scale_size, net.scale_size, flip=False) ]),
                           Stack(roll=(self.arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(self.arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std)])

        # place net onto GPU and finalize network
        net = torch.nn.DataParallel(net.cuda())
        net.eval()

        # network variable
        self.net = net
