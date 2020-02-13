
from backbone_wrapper import BackBone

import sys
sys.path.append("~/temporal-shift-module/")


from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config

class TSMBackBone(BackBone):

    def __init__(self):
        self.is_shift = None
        self.net = None

    def open_model(checkpoint_file):

        #checkpoint_file = TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth

        # input variables
        this_weights = checkpoint_file
        this_test_segments = 8
        test_file = None

        #model variables
        self.is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
        modality = 'RGB'
        this_arch = this_weights.split('TSM_')[1].split('_')[2]

        # dataset variables
        num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset('somethingv2',
                                                                                                modality)
        print('=> shift: {}, shift_div: {}, shift_place: {}'.format(self.is_shift, shift_div, shift_place))

        # define model
        net = TSN(num_class, this_test_segments if self.is_shift else 1, modality,
                  base_model=this_arch,
                  consensus_type=args.crop_fusion_type,
                  img_feature_dim=args.img_feature_dim,
                  pretrain=args.pretrain,
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


        # place net onto GPU and finalize network
        net = torch.nn.DataParallel(net.cuda())
        net.eval()
        self.net = net

    def predict(data_in):

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)
        
        with torch.no_grad():

            # predict value
            rst = net(data_in)
            rst = rst.reshape(batch_size, num_crop, -1).mean(1)

class DatasetParser:
    def __init__(self):
        pass

        itr = 0
        epoch = 0

    def getNext(self):
        pass

        itr += 1
        if(itr == dataset_size):
            itr = 0
            epoch += 1
        
        return data_in, label, file_name 