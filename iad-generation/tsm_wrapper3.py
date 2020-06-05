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
    '''
    def trainloader_from_csv_input(csv_input):

        #dataset = torch.utils.dataset()

        dataset = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),



        return torch.utils.data.DataLoader(dataset, 
                batch_size=1, shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0, collate_fn=None,
                pin_memory=False, drop_last=False, timeout=0,
                worker_init_fn=None)
    '''
    def predict(self, csv_input):

        data_in = self.open_file_as_batch(csv_input)

        # data has shape (batch size, segment length, num_ch, height, width)
        # (6,8,3,256,256)

        print("data_in:", data_in.shape)
        
        # predict value
        with torch.no_grad():
            return self.net(data_in)

    '''
    def train_model(self, csv_input):

        model = torch.nn.DataParallel(self.net).cuda()

        optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


        model.train()

        #end_frame = csv_input['length'] - (csv_input['length']%self.max_length)
        #for i in range(0, end_frame, 4):

        
        root_path = ?
        train_list = ?
        num_segments
        modality = ?
        dense_sample = ?
        batch_size = 128
        workers = ?
        arch = ?

        train_loader = torch.utils.data.DataLoader(
            TSNDataSet(root_path, train_list, num_segments=num_segments,
                   new_length=1,
                   modality=modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=dense_sample),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
            drop_last=True)  # prevent something not % n_GPU
        

        #trainloader = trainloader_from_csv_input(csv_input)

        for epoch in range(2):
            for i, data in enumerate(trainloader, 0):

                data_in = self.open_file(csv_input, start_idx = i)#self.open_file_as_batch(csv_input)

                # data has shape (batch size, segment length, num_ch, height, width)
                # (6,8,3,256,256)

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
    '''

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

        #net.base_model.fc = nn.Identity()

        net.new_fc = nn.Sequential(
            nn.Conv2d(2048, 128, (1,1)),
            nn.ReLU(),
            nn.Linear(128, 174)
            )# nn.Identity()
        print("TSM wrapper")
        print(net)

        # Will always need the activations (whether for out or for ranking)

        # add bottllneck

        #net.base_model.layer4.register_forward_hook(activation_hook(3))

        if(self.feature_idx == None):
            # Need to get rank information
            #net.base_model = nn.Sequential(net.base_model.layer4, YourModule())
            net.base_model.fc = nn.Identity()
            
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
        net = torch.nn.DataParallel(net.cuda())
        net.eval()

        # network variable
        self.net = net

        # loss variable (used for generating gradients when ranking)
        if(self.feature_idx == None):
            self.loss = torch.nn.CrossEntropyLoss().cuda()







def get_train_loader(model):
        root_path = '/home/mbc2004/datasets/Something-Something/20bn-something-something-v1'
        train_list = '/home/mbc2004/datasets/Something-Something/train_videofolder.txt'
        num_segments = 8
        modality = 'RGB'
        dense_sample = False
        batch_size = 64
        workers = 16
        arch = 'resnet50'

        prefix = '{:05d}.jpg'

        print('#' * 20, 'NO FLIP!!!')
        train_augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(model.input_size, [1, .875, .75, .66])])

        return torch.utils.data.DataLoader(
            TSNDataSet(root_path, train_list, num_segments=num_segments,
                   new_length=1,
                   modality=modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=dense_sample),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
            drop_last=True)  # prevent something not % n_GPU

def train(model, epoch):#, log, tf_writer):
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    '''
    train_loader = get_train_loader(model)
    model.module.partialBN(True)

    criterion = torch.nn.CrossEntropyLoss().cuda()


    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005
    optimizer = torch.optim.SGD(model.get_optim_policies(),
                                lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        '''
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        '''
        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    '''
        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    '''


'''
def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg
'''
