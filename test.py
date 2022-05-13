import math

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict
import cv2
import os
import shutil
import pickle as pkl
import time
import numpy as np
import hashlib
############################
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from IPython import embed
from collections import OrderedDict
####################################
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#######################################
from torch.autograd import Variable
from torch import nn


model_url = {
    'cifar10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth',
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',    
}

known_models = [
    'mnist', 'svhn', # 28 x 28
    'cifar10', 'cifar100', # 32 x 32
    'stl10', # 96 x 96
    'alexnet', # 224
    'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',# 224 x 224
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'squeezenet_v0', 'squeezenet_v1', # 224
    'inception_v3', # 229 x 229
]

def auto_select_gpu(mem_bound = 500, utility_bound = 0, gpus = (0,1,2,3,4,5,6,7), num_gpu=1, selected_gpus=None):
    import sys
    import os
    import subprocess
    import re
    import time
    import numpy as np
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5):
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert(len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]
        
        if len(ideal_gpus) < num_gpu:
            print('No sufficient resource, available: {}, require {} gpu'.format(ideal_gpus, num_gpu))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')
        
    print('Setting GPU: {}'.format(selected_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
    return selected_gpus

def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)
        
def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def cifar10(cuda=True, model_root=None):
    print('Building and initializing cifar10 parameters')
    # need model.py and dataset.py
    def make_layers(cfg, batch_norm=True):
        layers = [] # define list
        in_channels = 3
        for i, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                padding = v[1] if isinstance(v, tuple) else 1 # if v is tuple then v[1] else 1
                out_channels = v[0] if isinstance(v, tuple) else v # if v is tuple then v[0] else v
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = out_channels
        return nn.Sequential(*layers)
    
    class CIFAR(nn.Module):
        def __init__(self, features, n_channel, num_classes):
            super(CIFAR, self).__init__()
            assert isinstance(features, nn.Sequential), type(features)
            self.features = features
            self.classifier = nn.Sequential(
                nn.Linear(n_channel, num_classes)
            )
            print(self.features) # print features
            print(self.classifier)
            
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1) # flatten size rows
            x = self.classifier(x) 
            return x

    def _cifar10(n_channel, pretrained=None):
        cfg = [n_channel, n_channel, 'M', 2 * n_channel, 2 * n_channel, 'M', 4 * n_channel, 4 * n_channel, 'M',
               (8 * n_channel, 0), 'M']

        layers = make_layers(cfg, batch_norm=True)  # nn.Sequential(*[])
        model = CIFAR(layers, n_channel=8 * n_channel, num_classes=10)
        if pretrained is not None:
            m = model_zoo.load_url(model_url['cifar10'])
            state_dict = m.state_dict() if isinstance(m, nn.Module) else m
            assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
            model.load_state_dict(state_dict)
        return model
######## !!!!!!!
    def get10(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
        data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
        num_workers = kwargs.setdefault('num_workers', 1)
        kwargs.pop('input_size', None)
        print("Building CIFAR-10 data loader with {} workers".format(num_workers))
        ds = []
        if train:
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    root=data_root, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
                batch_size=batch_size, shuffle=True, **kwargs)
            ds.append(train_loader)
        if val:
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    root=data_root, train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
                batch_size=batch_size, shuffle=False, **kwargs)
            ds.append(test_loader)
        ds = ds[0] if len(ds) == 1 else ds
        return ds

    m = _cifar10(128, pretrained=os.path.join(model_root, 'cifar10.pth'))

    if cuda:
        m = m.cuda()
    return m, get10, False

    
def select(model_name, **kwargs):
    assert model_name in known_models, model_name
    kwargs.setdefault('model_root', os.path.expanduser('~/.torch/models'))
    return eval('{}'.format(model_name))(**kwargs) # model_name is not defined

# input
def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1) # take into flatten
    sorted_value = abs_value.sort(dim=0, descending=True)[0] #
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = float(v.data.cpu())
    sf = math.ceil(math.log2(v + 1e-12))
    return sf

def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, - sf)
    bound = math.pow(2.0, bits - 1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta # take
    return clipped_value

def min_max_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    min_val, max_val = input.min(), input.max()

    if isinstance(min_val, Variable):
        max_val = float(max_val.data.cpu().numpy()[0])
        min_val = float(min_val.data.cpu().numpy()[0])
    input_rescale = (input - min_val) / (max_val - min_val)

    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = v * (max_val - min_val) + min_val
    return v

def tanh_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    input = torch.tanh(input)
    input_rescale = (input + 1.0) / 2
    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = 2 * v - 1
    v = 0.5 * torch.log((1 + v) / (1 - v))
    return v

def log_minmax_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = min_max_quantize(input0, bits - 1)
    v = torch.exp(v) * s
    return v

def log_linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0
    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = linear_quantize(input0, sf, bits-1)
    v = torch.exp(v) * s
    return v


class LinearQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LinearQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            sf_new = self.bits - 1 - compute_integral_part(input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)


class LogQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LogQuant, self).__init__()
        self.name = name
        self._counter = counter
        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            log_abs_input = torch.log(torch.abs(input))
            sf_new = self.bits - 1 - compute_integral_part(log_abs_input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = log_linear_quantize(input, self.sf, self.bits)
            return output
    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

class NormalQuant(nn.Module):
    def __init__(self, name, bits, quant_func):
        super(NormalQuant, self).__init__()
        self.name = name
        self.bits = bits
        self.quant_func = quant_func

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        output = self.quant_func(input, self.bits)
        return output

    def __repr__(self):
        return '{}(bits={})'.format(self.__class__.__name__, self.bits)
#### need to test detailly !!!!!!
def duplicate_model_with_quant(model, bits, overflow_rate=0.0, counter=10, type='linear'):
    assert type in ['linear', 'minmax', 'log', 'tanh']
    if isinstance(model, nn.Sequential):
        l = OrderedDict()
        for k, v in model._modules.items():
            if isinstance(v, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.AvgPool2d)):
                l[k] = v
                if type == 'linear':
                    quant_layer = LinearQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                elif type == 'log':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=log_minmax_quantize)
                elif type == 'minmax':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=min_max_quantize)
                else:
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=tanh_quantize)
                l['{}_{}_quant'.format(k, type)] = quant_layer
            else:
                l[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        m = nn.Sequential(l)
        return m
    else:
        for k, v in model._modules.items():
            model._modules[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        return model


#################### !!!!!!! 需要探究一下细节代码
def eval_model(model, ds, n_sample=None, ngpu=1, is_imagenet=False):
    import tqdm
    import torch
    from torch import nn
    from torch.autograd import Variable

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        def forward(self, input):
            input.data.div_(255.) # normalize data first  (x - mean) / sigma
            input.data[:, 0, :, :].sub_(self.mean[0]).div_(self.std[0])
            input.data[:, 1, :, :].sub_(self.mean[1]).div_(self.std[1])
            input.data[:, 2, :, :].sub_(self.mean[2]).div_(self.std[2])
            return self.model(input)

    correct1, correct5 = 0, 0
    n_passed = 0
    if is_imagenet:
        model = ModelWrapper(model)
    model = model.eval() # set mode in evaluation mode
    model = torch.nn.DataParallel(model, device_ids=range(ngpu)).cuda() # multi gpu test

    n_sample = len(ds) if n_sample is None else n_sample
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        n_passed += len(data)
        data =  Variable(torch.FloatTensor(data)).cuda()
        indx_target = torch.LongTensor(target)
        output = model(data)
        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]

        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum()
        correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum()

        if idx >= n_sample - 1:
            break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5

if __name__ == '__main__':
    # test linear quantization first
    # set type quant_method batch_size gpu ngpu seed model_root data_root logdir
    # input_size n_sample param_bits bn_bits fwd_bits overflow_rate
    gpu = auto_select_gpu(utility_bound=0, num_gpu=1, selected_gpus=None)
    ngpu = len(gpu)
    ensure_dir("log/default")
    model_root = expand_user('~/.torch/models/')
    data_root = expand_user('/data/public_dataset/pytorch/')
    input_size = 224
    types = 'cifar10'
    input_size = 299 if 'inception' in types else input_size
    quant_method = 'log'
    param_bits = 6
    bn_bits = 32
    fwd_bits = 6
    overflow_rate = 0
    n_sample = 20
    batch_size = 100 #
    assert torch.cuda.is_available(), 'no cuda'
    # set seeds
    torch.manual_seed(117)
    torch.cuda.manual_seed(117)
    
    #load model dataset #
    model_raw, ds_fetcher, is_imagenet = select(types, model_root=model_root)
    # print('ds_fetcher: ', ds_fetcher)
    ngpu = ngpu if is_imagenet else 1
    # print model_raw applied dictionary
    for param_state in model_raw.state_dict():
        print(param_state, '\t', model_raw.state_dict()[param_state].size())

    if param_bits < 32:
        state_dict = model_raw.state_dict()
        state_dict_quant = OrderedDict()
        for k, v in state_dict.items(): # items k, v
            if 'running' in k: #BN Layer
                if bn_bits >= 32:
                    print('Ignoring {}'.format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = bn_bits
            else:
                bits = param_bits # layer bits

            if quant_method == 'linear':
                sf = bits - 1. - compute_integral_part(v, overflow_rate=overflow_rate)
                v_quant = linear_quantize(v, sf, bits=bits)
            elif quant_method == 'log':
                v_quant = log_minmax_quantize(v, bits=bits)
            elif quant_method == 'minmax':
                v_quant = min_max_quantize(v, bits=bits)
            else:
                v_quant = tanh_quantize(v, bits=bits)
            # print('k: ', k, ' v_quant: ', v_quant)
            state_dict_quant[k] = v_quant
            print(k, bits)
        model_raw.load_state_dict(state_dict_quant) # take pretrained model params into model_raw

        #quantize forward activation
        if fwd_bits < 32:
            model_raw = duplicate_model_with_quant(model_raw, bits=fwd_bits, overflow_rate=overflow_rate,
                                                   counter=n_sample, type=quant_method)
            # duplicate moel with quantize error!!!!!!!!
            print('after duplicate model ', model_raw)
            val_ds_tmp = ds_fetcher(10, data_root=data_root, train=False, input_size=input_size)
            eval_model(model_raw, val_ds_tmp, ngpu=1, n_sample=n_sample, is_imagenet=is_imagenet)
            #
            print('--------------------- forward finished! --------------------------')
        # eval model read dataset to do evaludation test !!!!!
        val_ds = ds_fetcher(batch_size, data_root=data_root, train=False, input_size=input_size)
        acc1, acc5 = eval_model(model_raw, val_ds, ngpu=ngpu, is_imagenet=is_imagenet)

        #print sf
        print(model_raw)
        res_str = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
            types, quant_method, param_bits, bn_bits, fwd_bits, overflow_rate, acc1, acc5)
        print(res_str)
