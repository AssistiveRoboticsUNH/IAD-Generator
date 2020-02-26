# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from ..resnetv1b import resnet18_v1b, resnet34_v1b, resnet50_v1b, resnet101_v1b, resnet152_v1b

__all__ = ['resnet18_v1b_sthsthv2', 'resnet34_v1b_sthsthv2', 'resnet50_v1b_sthsthv2',
           'resnet101_v1b_sthsthv2', 'resnet152_v1b_sthsthv2', 'resnet18_v1b_kinetics400',
           'resnet34_v1b_kinetics400', 'resnet50_v1b_kinetics400', 'resnet101_v1b_kinetics400',
           'resnet152_v1b_kinetics400', 'resnet50_v1b_ucf101', 'resnet50_v1b_hmdb51',
           'resnet50_v1b_custom']

class ActionRecResNetV1b(HybridBlock):
    r"""ResNet models for video action recognition
    Deep Residual Learning for Image Recognition, CVPR 2016
    https://arxiv.org/abs/1512.03385
    Parameters
    ----------
    depth : int, default is 50.
        Depth of ResNet, from {18, 34, 50, 101, 152}.
    nclass : int
        Number of classes in the training dataset.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    dropout_ratio : float, default is 0.5.
        The dropout rate of a dropout layer.
        The larger the value, the more strength to prevent overfitting.
    init_std : float, default is 0.001.
        Standard deviation value when initialize the dense layers.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    Input: a single video frame or N images from N segments when num_segments > 1
    Output: a single predicted action label
    """
    def __init__(self, depth, nclass, pretrained_base=True,
                 dropout_ratio=0.5, init_std=0.01,
                 num_segments=1, num_crop=1,
                 partial_bn=False, **kwargs):
        super(ActionRecResNetV1b, self).__init__()

        if depth == 18:
            pretrained_model = resnet18_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 1
        elif depth == 34:
            pretrained_model = resnet34_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 1
        elif depth == 50:
            pretrained_model = resnet50_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        elif depth == 101:
            pretrained_model = resnet101_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        elif depth == 152:
            pretrained_model = resnet152_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        else:
            print('No such ResNet configuration for depth=%d' % (depth))

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.feat_dim = 512 * self.expansion
        self.num_segments = num_segments
        self.num_crop = num_crop

        with self.name_scope():
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.relu = pretrained_model.relu
            self.maxpool = pretrained_model.maxpool
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
            self.avgpool = pretrained_model.avgpool
            self.flat = pretrained_model.flat
            self.drop = nn.Dropout(rate=self.dropout_ratio)
            self.output = nn.Dense(units=nclass, in_units=self.feat_dim,
                                   weight_initializer=init.Normal(sigma=self.init_std))
            self.output.initialize()

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

         # place to chew up first call
        x.attach_grad()


        outs = []

        x = self.layer1(x)
        if self.record_point == 0:
            x.attach_grad()
        outs.append(x)

        x = self.layer2(x)
        if self.record_point == 1:
            x.attach_grad()
        outs.append(x)

        x = self.layer3(x)
        if self.record_point == 2:
            x.attach_grad()
        outs.append(x)

        x = self.layer4(x)
        if self.record_point == 3:
            x.attach_grad()
        outs.append(x)

        self.activation_points = outs

        x = self.avgpool(x)
        x = self.flat(x)
        x = self.drop(x)

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        x = self.output(x)
        return x

def resnet50_v1b_sthsthv2(nclass=174, pretrained=False, pretrained_base=True,
                          use_tsn=False, partial_bn=False,
                          num_segments=1, num_crop=1, root='~/.mxnet/models',
                          ctx=mx.cpu(), **kwargs):
    r"""ResNet50 model trained on Something-Something-V2 dataset.
    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    """
    model = ActionRecResNetV1b(depth=50,
                               nclass=nclass,
                               partial_bn=partial_bn,
                               num_segments=num_segments,
                               num_crop=num_crop,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet50_v1b_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model
