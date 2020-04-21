from backbone_wrapper import BackBone

import sys, os
sys.path.append("/home/mbc2004/temporal-shift-module")

import torch.nn as nn
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config

import numpy as np
from PIL import Image

# added for pruning
from torch.autograd import Variable
import scipy.cluster.hierarchy as hcluster
import scipy.cluster.hierarchy as hac
import scipy.cluster.hierarchy as fclusterdata
from sklearn.preprocessing import normalize

DEPTH_SIZE = 4
CNN_FEATURE_COUNT = [256, 512, 1024, 2048]

def cluster_weights_agglo(weight, threshold, average=True):
    weight = weight.T
    weight = normalize(weight, norm='l2', axis=1)
    threshold =  1.0-threshold   # Conversion to distance measure
    clusters = hcluster.fclusterdata(weight, threshold, criterion="distance", metric='cosine', depth=1, method='centroid')
    z = hac.linkage(weight, metric='cosine', method='complete')
    labels = hac.fcluster(z, threshold, criterion="distance")

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    #print(n_clusters_)
    #elapsed_time = time.time() - t0
    # print(elapsed_time)

    a=np.array(labels)
    sort_idx = np.argsort(a)
    a_sorted = a[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    first_ele = [unq_idx[idx][-1] for idx in xrange(len(unq_idx))]
    return n_clusters_, first_ele



class TSMBackBone(BackBone):
         
    def open_file(self, csv_input, start_idx=0, batch_now=True):
        
        folder_name = csv_input['raw_path']
        assert os.path.exists(folder_name), "cannot find frames folder: "+folder_name
        files = os.listdir(folder_name)

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

    def prune(self):

        print(self.net)

        threshold = 0.54

        ii = 0
        first_ele = None
        nb_remanining_filters = []
        total_flop_after_pruning = 0
        rr = 1

        for layer in self.net.modules():

            if isinstance(layer, nn.ReLU):
                rr+=1
            if isinstance(layer, nn.MaxPool2d):
                rr+=1

            if isinstance(layer, nn.Conv2d):
                print("")
                print("layer:", layer)


                # get weights and biases
                weight = layer.weight.data.cpu().numpy()

                #reshape weights for clustering
                if first_ele is not None:
                    weight_layers_rearranged = np.transpose(weight, (1, 0, 2, 3))
                    weight_layers_rearranged_pruned = weight_layers_rearranged[first_ele]
                    weight_layers_rearranged_pruned = np.transpose(weight_layers_rearranged_pruned, (1, 0, 2, 3))
                else:
                    weight_layers_rearranged_pruned = weight
                weight_layers_rearranged = np.reshape(weight_layers_rearranged_pruned, [weight_layers_rearranged_pruned.shape[0], -1])
                
                # cluster weights and identify which features to retain (first_ele)
                n_clusters_,first_ele = cluster_weights_agglo(weight_layers_rearranged.T, threshold)
                first_ele = sorted(first_ele)
                nb_remanining_filters.append(n_clusters_)

                # prune filters
                weight_pruned = weight_layers_rearranged[first_ele]

                # correct pruned weight shape
                weight_pruned = np.reshape(weight_pruned, [n_clusters_, weight_layers_rearranged_pruned.shape[1],weight_layers_rearranged_pruned.shape[2],weight_layers_rearranged_pruned.shape[3]])

                params_1 = np.shape(weight_pruned)
                layer.out_channels = params_1[0]
                layer.in_channels = params_1[1]

                weight_tensor = torch.from_numpy(weight_pruned)
                layer.weight = torch.nn.Parameter(weight_tensor)

                if (layer.bias  != None):
                    bias = layer.bias.data.cpu().numpy()
                    bias_pruned = bias[first_ele]
                    bias_tensor = torch.from_numpy(bias_pruned)
                    layer.bias = torch.nn.Parameter(bias_tensor)


                params_1 = np.shape(weight_pruned)
                C1_1 = int(params_1[0])
                C2_1 = int(params_1[1])
                K1_1 = int(params_1[2])
                K2_1 = int(params_1[3])
                x = Variable(torch.randn(1,3, 32, 32))

                print("====================")
                print("self.net.base_model:", self.net.base_model.children())

                nett_1 = nn.Sequential(*list(self.net.base_model.children())[:rr])
                out_1 = nett_1(x)
                img_size_1 = out_1.size()
                # print('feature map size is:', img_size_1)
                # print('weight size is:', params_1)

                H_1 = img_size_1[2]
                W_1 = img_size_1[3]
                if ii==0:
                    H_1 = 32
                    W_1 = 32

                flops_1 = C1_1*C2_1*K1_1*K2_1*H_1*W_1
                print('flop is ',flops_1, '\n')
                total_flop_after_pruning +=flops_1
                ii+=1
                rr+=1

            if isinstance(layer, nn.BatchNorm2d) and first_ele is not None:
                print("")
                print("layer:", layer)

                # fix BatchNorm2D layers so that the pruned layers still work

                bnorm_weight = layer.weight.data.cpu().numpy()
                bnorm_weight = bnorm_weight[first_ele]
                bnorm_bias = layer.bias.data.cpu().numpy()
                bnorm_bias = bnorm_bias[first_ele]

                bnorm_tensor = torch.from_numpy(bnorm_weight)
                bias_tensor = torch.from_numpy(bnorm_bias)
                layer.weight = torch.nn.Parameter(bnorm_tensor)
                layer.bias = torch.nn.Parameter(bias_tensor)

                layer.num_features = int(np.shape(bnorm_weight)[0])
                bnorm_rm = layer.running_mean.cpu().numpy()
                bnorm_rm = bnorm_rm[first_ele]
                bnorm_rv = layer.running_var.cpu().numpy()
                bnorm_rv = bnorm_rv[first_ele]
                running_mean = torch.from_numpy(bnorm_rm)
                layer.running_mean = running_mean
                running_var = torch.from_numpy(bnorm_rv)
                layer.running_var = running_var
                rr+=1

            if isinstance(layer, nn.Linear):
                print("")
                print("layer:", layer)

                # fix Linear layers so that the pruned layers still work

                weight_linear = layer.weight.data.cpu().numpy()
                weight_linear_rearranged = np.transpose(weight_linear, (1, 0))
                weight_linear_rearranged_pruned = weight_linear_rearranged[first_ele]
                weight_linear_rearranged_pruned = np.transpose(weight_linear_rearranged_pruned, (1, 0))
                layer.in_features = int(np.shape(weight_linear_rearranged_pruned)[1])
                linear_tensor = torch.from_numpy(weight_linear_rearranged_pruned)
                layer.weight = torch.nn.Parameter(linear_tensor)

                #update nn.Linear to comply with newer Pytorch versions
                layer.track_running_stats = 1

                params_linear = np.shape(weight_linear_rearranged_pruned)
                C1_1 = params_linear[0]
                C2_1 = params_linear[1]

                flops_1 = C1_1*C2_1
                total_flop_after_pruning +=flops_1


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

    



    def __init__(self, checkpoint_file, num_classes, max_length=8, feature_idx=None):
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
        this_test_segments = self.max_length
        test_file = None

        #model variables
        self.is_shift, shift_div, shift_place = True, 8, 'blockres'

        
        self.arch = checkpoint_file.split('TSM_')[1].split('_')[2]
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
                  non_local='_nl' in checkpoint_file,
                  )

        # load checkpoint file
        checkpoint = torch.load(checkpoint_file)

        # add activation and ranking hooks
        self.activations = [None]*4
        self.ranks = [None]*4
        def activation_hook(idx):
            def hook(model, input, output):
                #prune features and only get those we are investigating 
                activation = output.detach()
                
                self.activations[idx] = activation
 
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

        '''
        if(self.feature_idx == None):
            # Need to get rank information
            net.base_model.layer1.register_backward_hook(taylor_expansion_hook(0))
            net.base_model.layer2.register_backward_hook(taylor_expansion_hook(1))
            net.base_model.layer3.register_backward_hook(taylor_expansion_hook(2))
            net.base_model.layer4.register_backward_hook(taylor_expansion_hook(3))
        else:
        '''
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

        '''
        # place net onto GPU and finalize network
        net = torch.nn.DataParallel(net.cuda())
        net.eval()
        '''

        # network variable
        self.net = net

        # loss variable (used for generating gradients when ranking)
        if(self.feature_idx == None):
            self.loss = torch.nn.CrossEntropyLoss().cuda()

if __name__ == '__main__':

    model_filename = "/home/mbc2004/models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth"
    num_classes=174

    model = TSMBackBone(model_filename, num_classes)
    model.prune()

