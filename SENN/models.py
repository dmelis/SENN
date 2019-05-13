# -*- coding: utf-8 -*-
""" Code for training and evaluating Self-Explaining Neural Networks.
Copyright (C) 2018 David Alvarez-Melis <dalvmel@mit.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# -*- coding: utf-8 -*-
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

DEBUG = False


#===============================================================================
#====================      SIMPLE FC and CNN MODELS  ===========================
#===============================================================================

class FCNet(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 548)
        self.bc1 = nn.BatchNorm1d(548)
        self.fc2 = nn.Linear(548, 252)
        self.bc2 = nn.BatchNorm1d(252)
        self.fc3 = nn.Linear(252, 10)

    def forward(self, x):
        x = x.view((-1, 784))
        h = self.fc1(x)
        h = self.bc1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        h = self.fc2(h)
        h = self.bc2(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)

        h = self.fc3(h)
        out = F.log_softmax(h)
        return out


class SENNModel(nn.Module):
    def __init__(self, din, h, dout):
        self.dout = dout
        self.din = din

        super(SENNModel, self).__init__()

        self.complex_part = nn.Sequential(
            nn.Linear(din, 548),
            nn.BatchNorm1d(548),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(548, 252),
            nn.BatchNorm1d(252),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(252, din * dout),
        )

    def forward(self, x):
        x = x.view((-1, self.din))
        # print(self.complex_part(x).size())
        params = self.complex_part(x).view(-1, self.dout, self.din)
        self.params = params
        out = torch.bmm(params, x.unsqueeze(2)).squeeze()
        out = F.softmax(out)
        return out

    def forward_with_params(self, x):
        x = x.view((-1, self.din))
        if self.params is None:
            raise ValueError('must have run forward first!')
        out = torch.bmm(self.params.repeat(x.size(0), 1, 1),
                        x.unsqueeze(2)).squeeze()
        out = F.softmax(out)
        return out


class SENN_FFFC(nn.Module):
    def __init__(self, din, h, dout):
        self.dout = dout
        self.din = din

        super(SENN_FFFC, self).__init__()

        self.complex_part = nn.Sequential(
            nn.Linear(din, 548),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(548, 252),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(252, din * dout),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view((-1, self.din))
        # print(self.complex_part(x).size())
        params = self.complex_part(x).view(-1, self.dout, self.din)
        self.params = params
        out = torch.bmm(params, x.unsqueeze(2)).squeeze()
        out = F.log_softmax(out)
        return out

    def forward_with_params(self, x):
        x = x.view((-1, self.din))
        if self.params is None:
            raise ValueError('must have run forward first!')
        out = torch.bmm(self.params.repeat(x.size(0), 1, 1),
                        x.unsqueeze(2)).squeeze()
        out = F.log_softmax(out)
        return out


class LENET(nn.Module):
    def __init__(self, din, h, dout):
        super(LENET, self).__init__()
        self.dout = dout
        self.din = din
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, din * dout)

    def forward(self, x):
        p = F.relu(F.max_pool2d(self.conv1(x), 2))
        p = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(p)), 2))
        p = p.view(-1, 320)
        #p = F.tanh(self.fc1(p))
        p = self.fc1(p)
        out = F.dropout(p, training=self.training).view(-1,
                                                        self.dout, self.din)
        return out


class SENN_LENET(nn.Module):
    def __init__(self, din, h, dout):
        super(SENN_LENET, self).__init__()
        self.dout = dout
        self.din = din
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, din * dout)

    def forward(self, x):
        p = F.relu(F.max_pool2d(self.conv1(x), 2))
        p = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(p)), 2))
        p = p.view(-1, 320)
        #p = F.tanh(self.fc1(p))
        p = self.fc1(p)
        params = F.dropout(
            p, training=self.training).view(-1, self.dout, self.din)
        self.params = params
        out = torch.bmm(params, x.view((-1, self.din, 1))).squeeze()
        out = F.log_softmax(out)
        return out

    def forward_with_params(self, x):
        x = x.view((-1, self.din, 1))
        if self.params is None:
            raise ValueError('must have run forward first!')
        out = torch.bmm(self.params.repeat(x.size(0), 1, 1),
                        x.unsqueeze(2)).squeeze()
        out = F.log_softmax(out)
        return out


class GSENN(nn.Module):
    ''' Wrapper for GSENN with H-learning'''

    def __init__(self, conceptizer, parametrizer, aggregator):
        super(GSENN, self).__init__()
        # self.dout = dout
        # self.din = din
        #self.encoder      = encoder
        #self.decoder      = decoder
        self.conceptizer = conceptizer
        self.parametrizer = parametrizer
        self.aggregator = aggregator
        self.learning_H = conceptizer.learnable
        self.reset_lstm = hasattr(
            conceptizer, 'lstm') or hasattr(parametrizer, 'lstm')

    def forward(self, x):
        #DEBUG = True
        if DEBUG:
            print('Input to GSENN:', x.size())

        # Get interpretable features
        #h_x         = self.encoder(x.view(x.size(0), -1)).view(-1, self.natoms, self.dout)
        #self.recons = self.decoder(h_x.view(-1, self.dout*self.natoms))
        if self.learning_H:
            h_x, x_tilde = self.conceptizer(x)
            self.recons = x_tilde
            # if self.sparsity:
            # Store norm for regularization (done by Trainer)
            # .mul(self.l1weight) # Save sparsity loss, will be used by trainer
            self.h_norm_l1 = h_x.norm(p=1)
        else:
            h_x = self.conceptizer(
                autograd.Variable(x.data, requires_grad=False))

        self.concepts = h_x  # .data

        if DEBUG:
            print('Encoded concepts: ', h_x.size())
            if self.learning_H:
                print('Decoded concepts: ', x_tilde.size())

        # Get relevance scores (~thetas)
        thetas = self.parametrizer(x)

        # When theta_i is of dim one, need to add dummy dim
        if len(thetas.size()) == 2:
            thetas = thetas.unsqueeze(2)

        # Store local Parameters
        self.thetas = thetas  # .data

        if DEBUG:
            print('Theta: ', thetas.size())

        if len(h_x.size()) == 4:
            # Concepts are two-dimensional, so flatten
            h_x = h_x.view(h_x.size(0), h_x.size(1), -1)

        #print(h_x.shape, thetas.shape)

        out = self.aggregator(h_x, thetas)

        # if self.aggregator.nclasses ==  1:
        #     out = out.squeeze() # Squeeze out single class dimension

        if DEBUG:
            print('Output: ', out.size())

        return out

    def predict_proba(self, x, to_numpy=False):
        if type(x) is np.ndarray:
            to_numpy = True
            x_t = torch.from_numpy(x).float()
        elif type(x) is Tensor:
            x_t = x.clone()
        else:
            print(type(x))
            raise ValueError("Unrecognized data type")
        out = torch.exp(self(Variable(x_t, volatile=True)).data)
        if to_numpy:
            out = out.numpy()
        return out

    def forward_with_params(self, x):
        #x = x.view((-1, self.din, 1))
        if self.learning_H:
            h_x, _ = self.conceptizer(x)
        else:
            h_x = self.conceptizer(x)

        if len(h_x.size()) == 4:
            # Concepts are two-dimensional, so flatten
            h_x = h_x.view(h_x.size(0), h_x.size(1), -1)

        if self.thetas is None:
            raise ValueError('must have run forward first!')
        if len(self.thetas.size()) == 2:
            # CAn happen if scalar parametrization and we squeezed out. THough should be correctyed.
            print('Warning: thetas should always have 3 dim. Check!')
            thetas = self.thetas.unsqueeze(-1)
        else:
            thetas = self.thetas

        out = self.aggregator(h_x, thetas)
        return out

    def explain(self, x, y=None, skip_bias=True):
        """
            Args:
                - y: class to explain (only useful for multidim outputs), if None, explains predicted
        """
        out = self.forward(x)
        theta = self.thetas.data.cpu()
        print("In construction")
        if theta.shape[-1] == 1:
            # single class
            attr = theta
        elif type(y) in [list, np.array]:
            y = torch.Tensor(y)
            attr = theta.gather(
                2, y.view(-1, 1).unsqueeze(2).repeat(1, theta.shape[1], theta.shape[2]))[:, :, 0]
        elif y == 'max':
            # desired class
            _, idx = torch.max(out, 1)
            y = idx.data
            
            attr = theta.gather(
                2, y.view(-1, 1).unsqueeze(2).repeat(1, theta.shape[1], theta.shape[2]))[:, :, 0]
        elif (y == 'all') or (y is None):
            # retrieve explanation for all classes
            attr = theta
        
        if (not skip_bias) and self.conceptizer.add_bias:
            pdb.set_trace()
            print('here')
            attr = torch.index_select(
                attr, -1, torch.LongTensor(range(attr.shape[-1] - 1)))
            pdb.set_trace()
        return attr


#===============================================================================
#====================      VGG MODELS FOR CIFAR  ===============================
#===============================================================================

# Note that these are tailored to native 32x32 resolution

cfg_cifar = {
    'vgg8':  [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_CIFAR(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG_CIFAR, self).__init__()
        self.features = self._make_layers(cfg_cifar[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

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
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg11_cifar():
    return VGG_CIFAR('vgg11')


def vgg13_cifar():
    return VGG_CIFAR('vgg13')


def vgg16_cifar():
    return VGG_CIFAR('vgg16')


def vgg19_cifar():
    return VGG_CIFAR('vgg19')
