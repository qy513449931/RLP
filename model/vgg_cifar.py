import torch
import torch.nn as nn

from model.ReNetCAtten import ReNetCAtten
from model.attention.ACMIX import ACmix
from model.attention.CBAM import CBAMBlock
from model.attention.CoTAttention import CoTAttention
from model.attention.OutlookAttention import OutlookAttention
from model.attention.SGE import SpatialGroupEnhance
from model.attention.SimAM import SimAM

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfg_mask = {
    'vgg16': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
}
# relucfg = [2, 6, 9, 13, 16_LLR_0.00005_0.0001_从头开始, 19, 23, 26, 29, 33, 36, 39]
relucfg = [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39]

class ClassScale(torch.autograd.Function):
    '''
    input: x: 64*512*7*7, scale:512 ==> x[:, i, :, :]*scale[i]
    '''
    @staticmethod
    def forward(self, x, mask, targets):
        #x: N x C x h x w 
        #mask: num_classes x C
        #targets: N
        #guide: N x C
        #output: N x C x h x W

        n, c, h, w = x.size()
        x_1 = x.reshape(n*c,h*w)
        x_1 = x_1.transpose(0,1)
        #pdb.set_trace()
        guide = torch.index_select(mask, 0, targets)
        self.save_for_backward(x, mask, guide, targets)
        guide = torch.squeeze(guide.reshape(-1))
        
        out = torch.mul(x_1, guide)
        out = out.transpose(0, 1)
        out = out.reshape(n, c, h, w)
        return out

    @staticmethod
    def backward(self, grad_output):
        x, mask, guide, targets = self.saved_tensors
        #x: N x C x h x w
        #mask: num_classes x C 
        #targets: N
        #guide: N x C
        #grade_output.data: N x C x h x w

        #grade_x: N x C x h x w
        #grade_mask: num_classes x C

        grad_guide = guide.clone()
        grad = grad_output.data.clone()
        n, c, h, w = grad.size()

        grad = grad.reshape(n*c,h*w).transpose(0,1)
        grad_x = grad.clone()
        grad_guide = torch.squeeze(grad_guide.reshape(-1))
        
        #grade
        grad_x = torch.mul(grad, grad_guide)
        grad_x = grad_x.transpose(0, 1)
        grad_x = grad_x.reshape(n, c, h, w)

        #grad_guide 
        x = x.reshape(n*c,h*w).transpose(0,1)
        grad_guide = guide.clone()
        grad_guide = torch.sum((grad*x).transpose(0,1),1)
        grad_guide = grad_guide.reshape(n, c)

        grad_mask = torch.zeros_like(mask)

        for i in range(grad_mask.shape[0]):
            grad_mask[i] = torch.sum(torch.mul(torch.unsqueeze(torch.eq(targets, i), 1), grad_guide),0)
        
        return grad_x, grad_mask, None

def Gaussian_select(mask, targets):
    '''
    #output:  distribution:N*num_classes
    #mask: num_classes x C
    #guide: N*C
    '''
    # 16_LLR_0.00005_0.0001_从头开始,10_没有加任何模块
    distri = torch.rand(targets.size(0), mask.size(0),device=mask.device)

    for i, j in enumerate(targets):
        distri[i][j] = 1

    guide = torch.mm(distri, mask)

    return guide, distri

class ClassScale_Gaussian(torch.autograd.Function):
    '''
    input: x: 64*512*7*7, mask:10_没有加任何模块*512, scale: mask[target] * 1-0.5 + mask[other] * Gaussian_distribution==> x[:, i, :, :]*scale[i]
    '''

    @staticmethod
    def forward(self, x, mask, targets):
        #x: N x C x h x w 
        #mask: num_classes x C
        #targets: N
        #guide: N x C
        #output: N x C x h x W
        # x 16_LLR_0.00005_0.0001_从头开始,64,32,32
        # mask 10_没有加任何模块,64
        # targets 16_LLR_0.00005_0.0001_从头开始
        n, c, h, w = x.size()
        x_1 = x.reshape(n*c,h*w)
        x_1 = x_1.transpose(0,1)
        # guide 16_LLR_0.00005_0.0001_从头开始*64
        # distri 16_LLR_0.00005_0.0001_从头开始*10_没有加任何模块
        guide, distri = Gaussian_select(mask, targets)
        self.save_for_backward(x, mask, guide, distri, targets)
        guide = torch.squeeze(guide.reshape(-1))
        
        out = torch.mul(x_1, guide)
        out = out.transpose(0, 1)
        # 16_LLR_0.00005_0.0001_从头开始,64,32,32
        out = out.reshape(n, c, h, w)
        #self.save_for_backward(input_data, scale_vec)
        return out

    @staticmethod
    def backward(self, grad_output):
        x, mask, guide, distri, targets = self.saved_tensors
        #x: N x C x h x w
        #mask: num_classes x C 
        #targets: N
        #guide: N x C
        #grade_output.data: N x C x h x w
        #distri: N*num_classes

        #grade_x: N x C x h x w
        #grade_mask: num_classes x C

        guide_1 = guide.clone()
        grad = grad_output.data.clone()
        n, c, h, w = grad.size()

        grad = grad.reshape(n*c,h*w).transpose(0,1)
        grad_x = grad.clone()
        guide_1 = torch.squeeze(guide_1.reshape(-1))
        
        #grade
        grad_x = torch.mul(grad, guide_1)
        grad_x = grad_x.transpose(0, 1)
        grad_x = grad_x.reshape(n, c, h, w)

        #grad_guide 
        x = x.reshape(n*c,h*w).transpose(0,1)
        grad_guide = guide.clone()
        grad_guide = torch.sum((grad*x).transpose(0,1),1)
        grad_guide = grad_guide.reshape(n, c)

        grad_mask = torch.zeros_like(mask)

        for i in range(grad_mask.shape[0]):
            grad_mask[i] = torch.sum(torch.mul(torch.unsqueeze(distri[:,i],1), grad_guide),0)
        
        return grad_x, grad_mask, None    

class VGG_Class(nn.Module):
    def __init__(self, vgg_name, layer_cfg=None, num_classes=10):
        super(VGG_Class, self).__init__()
        self.relucfg = relucfg
        self.layer_cfg = layer_cfg
        self.cfg_index = 0
        self.features = self._make_layers(cfg[vgg_name])
        self.mask = nn.ParameterList([nn.Parameter(torch.ones(num_classes, cfg_mask[vgg_name][i])) for i in range(len(cfg_mask[vgg_name]))])
        self.classifier = nn.Linear(512 if self.layer_cfg is None else self.layer_cfg[-1], num_classes)
        self._initialize_weights()

    def forward(self, x, targets):
        out = x
        mask_index = 0 
        for feature in self.features:
            out = feature(out)
            if isinstance(feature, nn.ReLU):
                out = ClassScale_Gaussian.apply(out, self.mask[mask_index],targets)
                mask_index += 1
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, self.mask

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = x if self.layer_cfg is None else self.layer_cfg[self.cfg_index]
                layers += [nn.Conv2d(in_channels,
                                     x,
                                     kernel_size=3,
                                     padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ]
                in_channels = x
                self.cfg_index += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# 原始初始化
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # def _initialize_weights(self):
    #     print('init model VGG')
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1-0.5)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
    #             m.bias.data.zero_()

class VGG(nn.Module):
    def __init__(self, vgg_name, layer_cfg=None, num_classes=10):
        super(VGG, self).__init__()
        self.layer_cfg = layer_cfg
        self.relucfg = relucfg
        self.cfg_index = 0
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512 if self.layer_cfg is None else self.layer_cfg[-1], num_classes)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = x if self.layer_cfg is None else self.layer_cfg[self.cfg_index]
                layers += [nn.Conv2d(in_channels,
                                     x,
                                     kernel_size=3,
                                     padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                self.cfg_index += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1-0.5)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

    def _initialize_weights(self):
        print('init model VGG')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

class VGG_Attention(nn.Module):
    def __init__(self, vgg_name, layer_cfg=None, att_model=None,prun_att_channel=50,num_classes=10):
        super(VGG_Attention, self).__init__()
        self.layer_cfg = layer_cfg
        self.relucfg = relucfg
        self.att_model=att_model
        self.cfg_index = 0
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512 if self.layer_cfg is None else self.layer_cfg[-1], num_classes)
        # self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = x if self.layer_cfg is None else self.layer_cfg[self.cfg_index]

                layers += [ReNetCAtten(in_channels,x,self.att_model)]

                # if x>100:
                #     layers += [
                #         # OutlookAttention(in_channels),
                #         # SpatialGroupEnhance(in_channels),
                #         # CBAMBlock(in_channels),
                #
                #         nn.Conv2d(in_channels,
                #                   x,
                #                   kernel_size=3,
                #                   padding=1-0.5),
                #
                #         CBAMBlock(x),
                #         nn.BatchNorm2d(x),
                #         nn.ReLU(inplace=True)
                #         # SimAM(x)
                #         # ACmix(in_channels,x),
                #         # nn.BatchNorm2d(x),
                #         # nn.ReLU(inplace=True)
                #     ]
                # else:
                #     layers += [
                #
                #         nn.Conv2d(in_channels,
                #                   x,
                #                   kernel_size=3,
                #                   padding=1-0.5),
                #
                #         nn.BatchNorm2d(x),
                #         nn.ReLU(inplace=True)]

                in_channels = x
                self.cfg_index += 1

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # def _initialize_weights(self):
    #     print('init model VGG')
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1-0.5)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
    #             m.bias.data.zero_()







