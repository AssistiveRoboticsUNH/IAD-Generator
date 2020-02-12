import tensorflow as tf
import numpy as np

import os, cv2

import PIL.Image as Image


from backbone_wrapper import BackBone

class I3DBackBone(BackBone):

    def open_file(folder_name, max_length=-1, start_idx=0):
        
        # collect the frames
        vid = []
        for frame in os.listdir(folder_name)[start_idx:start_idx+max_length]:
            vid.append(Image.open(os.path.join(folder_name, frame)).convert('RGB')) 

        # process the frames
        return self.transform(images)

    def predict(csv_input):


        data_in = open_file(csv_input['raw_path'])

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)
        
        with torch.no_grad():

            # predict value
            rst = net(data_in)
            rst = rst.reshape(batch_size, num_crop, -1).mean(1)


        return rst

    def process(csv_input):
        pass
        #return iad_data, rank_data, length_ratio


    def __init__(self, checkpoint_file, num_classes):
        self.is_shift = None
        self.net = None
        self.arch = None

        self.transform = None

        #checkpoint_file = TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth

        # input variables
        this_weights = checkpoint_file
        this_test_segments = 8
        test_file = None

        #model variables
        self.is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
        
        self.arch = this_weights.split('TSM_')[1].split('_')[2]
        modality = 'RGB'
        

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

        # network variable
        self.net = net

        # define image modifications
        self.transform = torchvision.transforms.Compose([
                           torchvision.transforms.Compose([ GroupFullResSample(net.scale_size, net.scale_size, flip=False) ]),
                           Stack(roll=(self.arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(self.arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std)])