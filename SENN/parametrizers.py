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

# Torch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision
import pdb
import numpy as np

# Local Imports
from .models import VGG_CIFAR


class dfc_parametrizer2(nn.Module):
    """ Deep fully connceted parametrizer for generic vector feature inputs. """
    def __init__(self, din, layer_dims, nconcept, dout, layers = 2):
        super(dfc_parametrizer2, self).__init__()
        dims   = [din] + list(layer_dims) + [nconcept*dout]
        layers = []
        for i, d in enumerate(dims[1:]):
            layers.append(nn.Linear(dims[i],d))
        self.linears = nn.ModuleList(layers)
        for layer in layers:
            print(layer)

    def forward(self, x):
        
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears):
                
                
                x = F.relu(x)
        return x

class dfc_parametrizer(nn.Module):
    """ Deep fully connceted parametrizer for generic vector feature imputs.
        Args:
            din (int): input concept dimension
            dout (int): output dimension (1 or number of label classes usually)

        Inputs:
            x:  Image tensor (b x c x d^2) [TODO: generalize to set maybe?]

        Output:
            Th:  Theta(x) vector of concept scores (b x nconcept x dout) (TODO: generalize to multi-class scores)
    """

    def __init__(self, din, hdim1, hdim2, hdim3, nconcept, dout, layers = 2):
        super(dfc_parametrizer, self).__init__()
        self.nconcept = nconcept
        self.din = din
        self.dout = dout
        self.linear1 = nn.Linear(din, hdim1)
        self.linear2 = nn.Linear(hdim1, hdim2)
        self.linear3 = nn.Linear(hdim2, hdim3)
        self.linear4 = nn.Linear(hdim3, nconcept * dout)

    def forward(self, x):
        
        p = F.tanh(self.linear1(x))
        p = F.tanh(self.linear2(p))
        p = F.tanh(self.linear3(p))
        #p = F.dropout(p, training=self.training)
        p = self.linear4(p) 
        if self.dout > 1:
            p = p.view(p.shape[0], self.nconcept, self.dout)
        return p




class torchvision_parametrizer(nn.Module):
    """ Parametrizer function - wrapper around architectures from torchvision.

        Args:
            din (int): input concept dimension
            dout (int): output dimension (1 or number of label classes usually)

        Inputs:
            x:  Image tensor (b x c x d^2) [TODO: generalize to set maybe?]

        Output:
            Th:  Theta(x) vector of concept scores (b x nconcept x dout) (TODO: generalize to multi-class scores)
    """

    def __init__(self, din, nconcept, dout, arch = 'alexnet', nchannel = 1, only_positive = False):
        super(torchvision_parametrizer, self).__init__()
        self.nconcept = nconcept
        self.dout = dout
        self.din  = din
        model_class = getattr(torchvision.models, arch)
        self.net = model_class(num_classes = nconcept*dout)
        # if arch == 'alexnet':
        #     self.net = torchvision.models.alexnet(num_classes = nconcept*dout)
        # elif arch == 'vgg11':
        #     self.net = torchvision.models.vgg11(num_classes = nconcept*dout)
        # elif arch == 'vgg16':
        #     self.net = torchvision.models.vgg16(num_classes = nconcept*dout)
        # elif arch == 'vgg16':
        #     self.net = torchvision.models.vgg16(num_classes = nconcept*dout)

        self.positive = only_positive

    def forward(self, x):
        p = self.net(x)
        out = F.dropout(p, training=self.training).view(-1,self.nconcept,self.dout)
        if self.positive:
            #out = F.softmax(out, dim = 1) # For fixed outputdim, sum over concepts = 1
            out = F.sigmoid(out) # For fixed outputdim, sum over concepts = 1
        else:
            out = F.tanh(out)
        return out

class vgg_parametrizer(nn.Module):
    """ Parametrizer function - VGG

        Args:
            din (int): input concept dimension
            dout (int): output dimension (1 or number of label classes usually)

        Inputs:
            x:  Image tensor (b x c x d^2) [TODO: generalize to set maybe?]

        Output:
            Th:  Theta(x) vector of concept scores (b x nconcept x dout) (TODO: generalize to multi-class scores)
    """

    def __init__(self, din, nconcept, dout, arch = 'alexnet', nchannel = 1, only_positive = False):
        super(vgg_parametrizer, self).__init__()
        self.nconcept = nconcept
        self.dout = dout
        self.din  = din
        self.net = VGG_CIFAR(arch, num_classes = nconcept*dout)
        # if arch == 'alexnet':
        #     self.net = torchvision.models.alexnet(num_classes = nconcept*dout)
        # elif arch == 'vgg11':
        #     self.net = torchvision.models.vgg11(num_classes = nconcept*dout)
        # elif arch == 'vgg16':
        #     self.net = torchvision.models.vgg16(num_classes = nconcept*dout)
        # elif arch == 'vgg16':
        #     self.net = torchvision.models.vgg16(num_classes = nconcept*dout)

        self.positive = only_positive

    def forward(self, x):
        p = self.net(x)
        out = F.dropout(p, training=self.training).view(-1,self.nconcept,self.dout)
        if self.positive:
            #out = F.softmax(out, dim = 1) # For fixed outputdim, sum over concepts = 1
            out = F.sigmoid(out) # For fixed outputdim, sum over concepts = 1
        else:
            out = F.tanh(out)
        return out

class image_parametrizer(nn.Module):
    """ Simple CNN-based parametrizer function for generic image imputs.

        Args:
            din (int): input concept dimension
            dout (int): output dimension (1 or number of label classes usually)

        Inputs:
            x:  Image tensor (b x c x d^2) [TODO: generalize to set maybe?]

        Output:
            Th:  Theta(x) vector of concept scores (b x nconcept x dout) (TODO: generalize to multi-class scores)
    """

    def __init__(self, din, nconcept, dout, nchannel = 1, only_positive = False):
        super(image_parametrizer, self).__init__()
        self.nconcept = nconcept
        self.dout = dout
        self.din  = din
        self.conv1 = nn.Conv2d(nchannel, 10, kernel_size=5)   # b, 10, din - (k -1), same
        # after ppol layer with stride=2: din/2 - (k -1)/2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)         # b, 20, din/2 - 3(k -1)/2, same
        # after ppol layer with stride=2: din/4 - 3(k -1)/4
        self.dout_conv = int(np.sqrt(din)//4 - 3*(5-1)//4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*(self.dout_conv**2), nconcept*dout)
        self.positive = only_positive

    def forward(self, x):
        p = F.relu(F.max_pool2d(self.conv1(x), 2))
        p = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(p)), 2))
        p = p.view(-1, 20*(self.dout_conv**2))
        p = self.fc1(p)
        out = F.dropout(p, training=self.training).view(-1,self.nconcept,self.dout)
        if self.positive:
            #out = F.softmax(out, dim = 1) # For fixed outputdim, sum over concepts = 1
            out = F.sigmoid(out) # For fixed outputdim, sum over concepts = 1
        else:
            out = F.tanh(out)
        return out


#
# class LSTMTagger(nn.Module):
#
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
#         super(LSTMTagger, self).__init__()
#         self.hidden_dim = hidden_dim
#
#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
#
#         # The LSTM takes word embeddings as inputs, and outputs hidden states
#         # with dimensionality hidden_dim.
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)
#
#         # The linear layer that maps from hidden state space to tag space
#         self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
#         self.hidden = self.init_hidden()
#
#     def init_hidden(self):
#         # Before we've done anything, we dont have any hidden state.
#         # Refer to the Pytorch documentation to see exactly
#         # why they have this dimensionality.
#         # The axes semantics are (num_layers, minibatch_size, hidden_dim)
#         return (Variable(torch.zeros(1, 1, self.hidden_dim)),
#                 Variable(torch.zeros(1, 1, self.hidden_dim)))
#
#     def forward(self, sentence):
#         print(sentence.size())
#         embeds = self.word_embeddings(sentence.squeeze()).transpose(0,1)
#         print(embeds.size())
#         lstm_out, self.hidden = self.lstm(embeds, self.hidden)
# #         tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
#         tag_space = self.hidden2tag(lstm_out)
#         tag_scores = F.log_softmax(tag_space, dim=1).transpose(0,1)
#         return tag_scores

class text_parametrizer(nn.Module):
    """ Parametrizer function for text imputs.

        Args:
            din (int): input concept dimension
            dout (int): output dimension (number of concepts)

        Inputs:
            x:  Image tensor (b x 1 x L) [TODO: generalize to set maybe?]

        Output:
            Th:  Theta(x) vector of concept scores (b x nconcept x cdim) (TODO: generalize to multi-class scores)
    """

    #def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
    def __init__(self, dout, vocab_size= None, hidden_dim = None, embeddings = None, train_embeddings = False, layers = 1):
        super(text_parametrizer, self).__init__()
        self.lstm = True # Lets GSENN know that it has lstm and thus tells trainer to reset after each batch
        if embeddings is not None:
            vocab_size, hidden_dim = embeddings.shape
            self.embedding_layer = nn.Embedding(vocab_size,hidden_dim)
            self.embedding_layer.weight.data = torch.from_numpy( embeddings )
            print('Text parametrizer: initializing embeddings')
        else:
            assert (vocab_size is not None) and (hidden_dim is not None)
            self.embedding_layer = nn.Embedding(vocab_size,hidden_dim)

        self.hidden_dim = hidden_dim
        self.dout = dout
        #self.embedding_layer.weight.requires_grad = train_embeddings
        if embeddings is None and not train_embeddings:
            print('Warning: embeddings not initialized from pre-trained and train = False')
        self.lstm = nn.LSTM(hidden_dim, hidden_dim) #, num_layers = layers)
        self.hidden2label = nn.Linear(hidden_dim, dout)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        
        embeds = self.embedding_layer(sentence.squeeze(1))
        x = embeds.transpose(0,1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out) 
        #out = F.sigmoid(y).transpose(0,1)  # Now it's b x L x 1
        out = F.softmax(y, 1).transpose(0,1)
        return out
