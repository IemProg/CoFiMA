import copy
import torch
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50
from convs.linears import SimpleContinualLinear, SimpleLinear
from convs.vits import (vit_base_patch16_224_in21k, vit_base_patch16_224_mocov3,
                        vit_tiny_patch16_224_in21k, vit_large_patch16_224_in21k,
                        vit_base_patch16_sam_224, vit_base_patch16_dinov2,
                        vit_base_patch16_224_mae,
                        vit_base_patch16_224_clip, vit_base_patch16_224_in21k_cil)
from easydict import EasyDict
from collections import OrderedDict
import math
import torch.nn.functional as F
from convs.linears import CosineLinear


def get_convnet(convnet_type, pretrained=True, args=None):
    name = convnet_type.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet18_cifar':
        return resnet18(pretrained=pretrained, cifar=True)
    elif name == 'resnet18_cifar_cos':
        return resnet18(pretrained=pretrained, cifar=True, no_last_relu=True)
    elif name == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name == 'vit-b-p16':
        return vit_base_patch16_224_in21k(pretrained=pretrained)
    elif name == 'vit-tiny-p16':
        return vit_tiny_patch16_224_in21k(pretrained=pretrained)
    elif name == 'vit-large-p16':
        return vit_large_patch16_224_in21k(pretrained=pretrained)
    elif name == 'vit-base-p16-sam':
        return vit_base_patch16_sam_224(pretrained=pretrained)
    elif name == 'vit-b-p16-mocov3':
        return vit_base_patch16_224_mocov3(pretrained=True)
    elif name == 'vit-b-p16-dino2':
        return vit_base_patch16_dinov2(pretrained=True)
    elif name == 'vit-b-p16-clip':
        return vit_base_patch16_224_clip(pretrained=True)
    elif name == 'vit-b-p16-cil':
        return vit_base_patch16_224_in21k_cil(pretrained=True)
    elif name== "vit-b-p16-mae":
        return vit_base_patch16_224_mae(pretrained=True)
    else:
        raise NotImplementedError('Unknown type {}'.format(convnet_type))


class BaseNet(nn.Module):

    def __init__(self, convnet_type, pretrained, args=None):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(convnet_type, pretrained, args)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)['features']

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        '''
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        '''
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class FinetuneIncrementalNet(BaseNet):
    def __init__(self, convnet_type, pretrained, fc_with_ln=False, args=None):
        super().__init__(convnet_type, pretrained, args)
        self.old_fc = None
        self.fc_with_ln = fc_with_ln

    def extract_layerwise_vector(self, x, pool=True):
        with torch.no_grad():
            features = self.convnet(x, layer_feat=True)['features']
        for f_i in range(len(features)):
            if pool:
                features[f_i] = features[f_i].mean(1).cpu().numpy() 
            else:
                features[f_i] = features[f_i][:, 0].cpu().numpy() 
        return features

    def update_fc(self, nb_classes, freeze_old=True):
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old)

    def save_old_fc(self):
        if self.old_fc is None:
            self.old_fc = copy.deepcopy(self.fc)
        else:
            self.old_fc.heads.append(copy.deepcopy(self.fc.heads[-1]))

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleContinualLinear(in_dim, out_dim)

        return fc

    def forward(self, x, bcb_no_grad=False, fc_only=False):
        if fc_only:
            fc_out = self.fc(x)
            if self.old_fc is not None:
                old_fc_logits = self.old_fc(x)['logits']
                fc_out['old_logits'] = old_fc_logits
            return fc_out
        if bcb_no_grad:
            with torch.no_grad():
                x = self.convnet(x)
        else:
            x = self.convnet(x)
        out = self.fc(x['features'])
        out.update(x)

        return out

class SimpleVitNet(BaseNet):
    def __init__(self, convnet_type, pretrained, fc_with_ln=False, args=None):
        super().__init__(convnet_type, pretrained, args)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet(x)

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        # out.update(x)
        return out