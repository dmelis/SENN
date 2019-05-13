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

import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from torch.legacy import nn as nn_legacy
from torch.autograd import Variable

#===============================================================================
#==========================      REGULARIZERS        ===========================
#===============================================================================

# From https://discuss.pytorch.org/t/how-to-create-a-sparse-autoencoder-neural-network-with-pytorch/3703
class L1Penalty(Function):
    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(self.l1weight)
        grad_input += grad_output
        return grad_input


#===============================================================================
#=======================       MODELS FOR IMAGES       =========================
#===============================================================================

class input_conceptizer(nn.Module):
    """ Dummy conceptizer for images: each input feature (e.g. pixel) is a concept.

        Args:
            indim (int): input concept dimension
            outdim (int): output dimension (num classes)

        Inputs:
            x: Image (b x c x d x d) or Generic tensor (b x dim)

        Output:
            - H:  H(x) matrix of concepts (b x dim  x 1) (for images, dim = x**2)
                  or (b x dim +1 x 1) if add_bias = True
    """

    def __init__(self, add_bias = True):
        super(input_conceptizer, self).__init__()
        self.add_bias = add_bias
        self.learnable = False

    def forward(self, x):
        if len(list(x.size())) == 4:
            # This is an images
            out = x.view(x.size(0), x.size(-1)**2, 1)
        else:
            out = x.view(x.size(0), x.size(1), 1)
        if self.add_bias:
            pad = (0,0,0,1) # Means pad to next to last dim, 0 at beginning, 1 at end
            out = F.pad(out, pad, mode = 'constant', value = 1)
        return out


class AutoEncoder(nn.Module):
    """
        A general autoencoder meta-class with various penalty choices.

        Takes care of regularization, etc. Children of the AutoEncoder class
        should implement encode() and decode() functions.
        Encode's output should be same size/dim as decode input and viceversa.
        Ideally, AutoEncoder should not need to do any resizing (TODO).

    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # self.sparsity = sparsity is not None
        # self.l1weight = sparsity if (sparsity) else 0.1

    def forward(self, x):
        encoded = self.encode(x)
        # if self.sparsity:
        #     #encoded = L1Penalty.apply(encoded, self.l1weight)    # Didn't work
        decoded = self.decode(encoded)
        return encoded, decoded.view_as(x)

class image_fcc_conceptizer(AutoEncoder):
    """ MLP-based conceptizer for concept basis learning.

        Args:
            din (int): input size
            nconcept (int): number of concepts
            cdim (int): concept dimension

        Inputs:
            x: Image (b x c x d x d)

        Output:
            - Th: Tensor of encoded concepts (b x nconcept x cdim)
    """

    def __init__(self, din, nconcept, cdim): #, sparsity = None):
        super(image_fcc_conceptizer, self).__init__()
        self.din      = din        # Input dimension
        self.nconcept = nconcept   # Number of "atoms"/concepts
        self.cdim     = cdim       # Dimension of each concept
        self.learnable = True

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 12), nn.ReLU(True),
            nn.Linear(12, nconcept*cdim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(nconcept*cdim, 12), nn.ReLU(True),
            nn.Linear(12, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )
    def encode(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x).view(-1, self.nconcept, self.cdim)
        return encoded

    def decode(self, z):
        decoded = self.decoder(z.view(-1, self.cdim*self.nconcept))
        return decoded

    # def forward(self, x):
    #     x_dims = x.size()
    #     x = x.view(x.size(0), -1)
    #     encoded = self.encoder(x).view(-1, self.nconcept, self.cdim)
    #     decoded = self.decoder(encoded.view(-1, self.cdim*self.nconcept)).view(x_dims)
    #     return encoded, decoded
    #


class image_cnn_conceptizer(AutoEncoder):
    """ CNN-based conceptizer for concept basis learning.

        Args:
            din (int): input size
            nconcept (int): number of concepts
            cdim (int): concept dimension

        Inputs:
            x: Image (b x c x d x d)

        Output:
            - Th: Tensor of encoded concepts (b x nconcept x cdim)
    """

    def __init__(self, din, nconcept, cdim=None, nchannel =1): #, sparsity = None):
        super(image_cnn_conceptizer, self).__init__()
        self.din      = din        # Input dimension
        self.nconcept = nconcept   # Number of "atoms"/concepts
        self.cdim     = cdim       # Dimension of each concept
        self.nchannel = nchannel
        self.learnable = True
        self.add_bias = False
        self.dout     = int(np.sqrt(din)//4 - 3*(5-1)//4) # For kernel = 5 in both, and maxppol stride = 2 in both

        # Encoding
        self.conv1  = nn.Conv2d(nchannel,10, kernel_size=5)    # b, 10, din - (k -1),din - (k -1)
        # after pool layer (functional)                        # b, 10,  (din - (k -1))/2, idem
        self.conv2  = nn.Conv2d(10, nconcept, kernel_size=5)   # b, 10, (din - (k -1))/2 - (k-1), idem
        # after pool layer (functional)                        # b, 10,  din/4 - 3(k-1))/4, idem
        self.linear = nn.Linear(self.dout**2, self.cdim)       # b, nconcepts, cdim

        # Decoding
        self.unlinear = nn.Linear(self.cdim,self.dout**2)                # b, nconcepts, dout*2
        self.deconv3  = nn.ConvTranspose2d(nconcept, 16, 5, stride = 2)  # b, 16, (dout-1)*2 + 5, 5
        self.deconv2  = nn.ConvTranspose2d(16, 8, 5)                     # b, 8, (dout -1)*2 + 9
        self.deconv1  = nn.ConvTranspose2d(8, nchannel, 2, stride=2, padding=1) # b, nchannel, din, din


    def encode(self, x):
        
        p       = F.relu(F.max_pool2d(self.conv1(x), 2))
        p       = F.relu(F.max_pool2d(self.conv2(p), 2))
        encoded = self.linear(p.view(-1, self.nconcept, self.dout**2))
        return encoded

    def decode(self, z):
        q       = self.unlinear(z).view(-1, self.nconcept, self.dout, self.dout)
        q       = F.relu(self.deconv3(q))
        q       = F.relu(self.deconv2(q))
        decoded = F.tanh(self.deconv1(q))
        return decoded
    #
    #
    # def forward(self, x):
    #     
    #
    #     # Encoding
    #     p       = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     p       = F.relu(F.max_pool2d(self.conv2(p), 2))
    #     encoded = self.linear(p.view(-1, self.nconcept, 4 * 4))
    #
    #     # Decoding
    #     q       = self.unlinear(encoded).view(-1, self.nconcept, 4, 4)
    #     q       = F.relu(self.deconv3(q))
    #     q       = F.relu(self.deconv2(q))
    #     decoded = F.tanh(self.deconv1(q))
    #     # decoded =
    #     #
    #     # encoded  = self.Linear(conv_out.view())
    #     # decoded = self.decoder(encoded)
    #     # print(encoded.size())
    #     # encoded = encoded.view(x.size(0), self.nconcept, self.cdim)
    #     return encoded, decoded

#===============================================================================
#=======================       MODELS FOR TEXT       ===========================
#===============================================================================


class text_input_conceptizer(nn.Module):
    """ Dummy conceptizer for images: each token is a concept.

        Args:

        Inputs:
            x: Text tensor (one hot) (b x 1 x L)

        Output:
            - H:  H(x) matrix of concepts (b x L x 1) [TODO: generalize to set maybe?]
    """

    def __init__(self):
        super(text_input_conceptizer, self).__init__()
        # self.din    = din
        # self.nconcept = nconcept
        # self.cdim   = cdim

    def forward(self, x):
        #return x.view(x.size(0), x.size(-1)**2, 1)
        #return x.transpose(1,2)._fill(1)
        return Variable(torch.ones(x.size())).transpose(1,2)


class text_embedding_conceptizer(nn.Module):
    """ H(x): word embedding of word x.

        Can be used in a non-learnt way (e.g. if embeddings are already trained)
        TODO: Should we pass this to parametrizer?

        Args:
            embeddings (optional): pretrained embeddings to initialize method

        Inputs:
            x: array of word indices (L X B X 1)

        Output:
            enc: encoded representation (L x B x D)
    """

    def __init__(self, embeddings = None, train_embeddings = False):
        super(text_embedding_conceptizer, self).__init__()
        vocab_size, hidden_dim = embeddings.shape
        self.hidden_dim = hidden_dim
        self.embedding_layer = nn.Embedding(vocab_size,hidden_dim)
        print(type(embeddings))
        if embeddings is not None:
            self.embedding_layer.weight.data = torch.from_numpy( embeddings )
            print('Text conceptizer: initializing embeddings')
        self.embedding_layer.weight.requires_grad = train_embeddings
        if embeddings is None and not train_embeddings:
            print('Warning: embeddings not initialized from pre-trained and train = False')


    def forward(self, x):
        encoded = self.embedding_layer(x.squeeze(1))
        #encoded = encoded.transpose(0,1) # To have Batch dim again in 0
        return encoded
