from backbone_wrapper import BackBone

import sys, os
sys.path.append("/home/mbc2004/temporal-shift-module")

import torch.nn as nn
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from ops.transforms import *
from ops.utils import AverageMeter, accuracy

import numpy as np
from PIL import Image
import time

DEPTH_SIZE = 4
CNN_FEATURE_COUNT = [256, 512, 1024, 2048]

class TSMIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:  # single-process data loading, return the full iterator
                iter_start = self.start
                iter_end = self.end
            else:  # in a worker process
                # split workload
                per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
                worker_id = worker_info.id
                iter_start = self.start + worker_id * per_worker
                iter_end = min(iter_start + per_worker, self.end)
            return iter(range(iter_start, iter_end))


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
            print("rst1:", rst.shape)
           
            # convert actvitaion from PyTorch to Numpy
            rst = rst.cpu().numpy()
            #rst = rst.reshape((-1, 128, 8,8))
            rst = rst.reshape((-1, self.bottleneck_size))

            print("rst2:", rst.shape)
            assert False

            # compress spatial dimensions
            rst = np.max(rst, axis=(2,3))
            rst = rst.T

        return rst, length_ratio

    def __init__(self, checkpoint_file, num_classes, max_length=8, trim_net=False, checkpoint_is_model=False, bottleneck_size=128):
        self.is_shift = None
        self.net = None
        self.arch = None
        self.num_classes = num_classes
        self.max_length = max_length
        self.bottleneck_size = bottleneck_size
        #self.feature_idx = feature_idx

        self.transform = None

        self.CNN_FEATURE_COUNT = [256, 512, 1024, 2048]

        # input variables
        this_test_segments = self.max_length
        test_file = None

        #model variables
        self.is_shift, shift_div, shift_place = True, 8, 'blockres'

        
        self.arch = 'resnet101'
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

        '''
        The checkpoint file appears to be an entire TSMBackBone Object. this needs to be
        handled acordingly. Either find a way to convert it back to a weights file or maniuplate it 
        to work with the system.
        '''


        # load checkpoint file
        checkpoint = torch.load(checkpoint_file)
        print(net)
        #include
        print("self.bottleneck_size:", self.bottleneck_size, type(self.bottleneck_size))
        net.base_model.avgpool = nn.Sequential(
            nn.Conv2d(2048, self.bottleneck_size, (1,1)),
            #nn.MaxPool2d()

            #BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #nn.ReLU(inplace=True)
            nn.MaxPool2d((8,8)),
            #nn.AdaptiveAvgPool2d(output_size=1),
            #nn.Linear(2048, self.bottleneck_size)
            #nn.Linear(2048, self.bottleneck_size),
            #nn.Conv1d(2048, self.bottleneck_size, ),

        )


        
        if(not trim_net):
            print("no trim")
            net.new_fc = nn.Linear(self.bottleneck_size, 174)
        else:
            print("trim")
            net.consensus = nn.Identity()
            net.new_fc = nn.Identity()

        #net.base_model.fc = nn.Identity() # sets the dropout value to None
       # print(net) 
        
        # Combine network together so that the it can have parameters set correctly
        # I think, I'm not 100% what this code section actually does and I don't have 
        # the time to figure it out right now
        #print("checkpoint------------------------")
        #print(checkpoint)
        
        if (checkpoint_is_model):
            checkpoint = checkpoint.net.state_dict()
        else:
            checkpoint = checkpoint['state_dict']

        
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        
        
        #include
        replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                        'base_model.classifier.bias': 'new_fc.bias',
                        }
        for k, v in replace_dict.items():
            if v in base_dict:
                base_dict.pop(v)
            if k in base_dict:
                base_dict.pop(k)
                #base_dict[v] = base_dict.pop(k)
        
        net.load_state_dict(base_dict, strict=False)

        


        
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
        self.model = net
        net = torch.nn.DataParallel(net.cuda())
        net.eval()


        # network variable
        self.net = net

        

        # loss variable (used for generating gradients when ranking)
        if(not trim_net):
            self.loss = torch.nn.CrossEntropyLoss().cuda()


def get_train_loader(model):
        root_path = '/home/mbc2004/datasets/Something-Something/frames/'
        train_list = '/home/mbc2004/datasets/Something-Something/annotations/train_videofolder.txt'
        num_segments = 8
        modality = 'RGB'
        dense_sample = False
        batch_size = 8#64
        workers = 16
        arch = 'resnet50'

        prefix = '{:06d}.jpg'

        print('#' * 20, 'NO FLIP!!!')
        train_augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66])])

        return torch.utils.data.DataLoader(
            TSNDataSet(root_path, train_list, num_segments=num_segments,
                   new_length=1,
                   modality=modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
                       IdentityTransform(),
                   ]), dense_sample=dense_sample),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
            drop_last=True)  # prevent something not % n_GPU

def train(model, epoch):

    train_loader = get_train_loader(model)
    
    criterion = torch.nn.CrossEntropyLoss().cuda()

    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005

    params = list(model.model.new_fc.parameters())

    optimizer = torch.optim.SGD(params,
                                lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    for i, (input, target) in enumerate(train_loader):
        if(i % 100 == 0):
            print("iter: {:6d}/{:6d}".format(i, len(train_loader)))

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model.net(input_var)
        print("output.shape:", output.shape)

        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        
    torch.save(model, "./saved_bottleneck_model_"+str(model.bottleneck_size)+".pt")




def get_val_loader(model):
        root_path = '/home/mbc2004/datasets/Something-Something/frames/'
        train_list = '/home/mbc2004/datasets/Something-Something/annotations/val_videofolder.txt'
        num_segments = 8
        modality = 'RGB'
        dense_sample = False
        batch_size = 8#64
        workers = 16
        arch = 'resnet50'

        prefix = '{:06d}.jpg'

        print('#' * 20, 'NO FLIP!!!')
        train_augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66])])

        return torch.utils.data.DataLoader(
            TSNDataSet(root_path, train_list, num_segments=num_segments,
                   new_length=1,
                   modality=modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
                       IdentityTransform(),
                   ]), dense_sample=dense_sample),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
            drop_last=True)  # prevent something not % n_GPU

def validate(model, epoch):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    

    '''
    Make sure that the validation works so I can confirm that there is an improvement/loss in
    accuracy when running this model.

    Run model
    '''
    val_loader = get_val_loader(model)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output = model.net(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                output = ('Test: [{0}/{1}]\t'
                          #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    
    return top1.avg

