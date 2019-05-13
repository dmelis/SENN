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
import torch
import torch.nn as nn
import torch.nn.functional as F

class additive_scalar_aggregator(nn.Module):
    """ Linear aggregator for interpretable classification.

        Aggregates a set of concept representations and their
        scores, generates a prediction probability output from them.

        Args:
            cdim (int):     input concept dimension
            nclasses (int): number of target classes

        Inputs:
            H:   H(x) vector of concepts (b x k x 1) [TODO: generalize to set maybe?]
            Th:  Theta(x) vector of concept scores (b x k x nclass)

        Output:
            - Vector of class probabilities (b x o_dim)

        TODO: add number of layers as argument, construct in for?
    """

    def __init__(self, cdim, nclasses):
        super(additive_scalar_aggregator, self).__init__()
        self.cdim      = cdim       # Dimension of each concept
        self.nclasses  = nclasses   # Numer of output classes
        self.binary = (nclasses == 1)

    def forward(self, H, Th):
        assert H.size(-2) == Th.size(-2), "Number of concepts in H and Th don't match"
        assert H.size(-1) == 1, "Concept h_i should be scalar, not vector sized"
        assert Th.size(-1) == self.nclasses, "Wrong Theta size"
        combined = torch.bmm(Th.transpose(1,2), H).squeeze(dim=-1)
        if self.binary:
            out = F.sigmoid(combined)
        else:
            out =  F.log_softmax(combined, dim = 1)
        return out


class linear_scalar_aggregator(nn.Module):
    """ Linear aggregator for interpretable classification.

        Aggregates a set of concept representations and their
        scores, generates a prediction probability output from them.

        Args:
            cdim (int):     input concept dimension
            nclasses (int): number of target classes

        Inputs:
            H:   H(x) vector of concepts (b x k x 1) [TODO: generalize to set maybe?]
            Th:  Theta(x) vector of concept scores (b x k x nclass)

        Output:
            - Vector of class probabilities (b x o_dim)

        TODO: add number of layers as argument, construct in for?
    """

    def __init__(self, cdim, nclasses, softmax_pre = True):
        super(linear_scalar_aggregator, self).__init__()
        self.cdim      = cdim       # Dimension of each concept
        self.nclasses  = nclasses   # Numer of output classes

        self.linear = nn.Linear(din, dout)
        self.softmax_pre = softmax_pre

    def forward(self, H, Th):
        assert H.size(-2) == Th.size(-2), "Number of concepts in H and Th don't match"
        assert H.size(-1) == 1, "Concept h_i should be scalar, not vector sized"
        assert Th.size(-1) == self.nclasses, "Wrong Theta size"

        # Previously:
        #combined = torch.bmm(H.transpose(1,2), Th).squeeze(dim=-1) # Don't want to squeeze batch dim!

        # New - do softmax before aggregation:
        if self.softmax_pre:
            H_soft = F.log_softmax(self.linear(H), dim=2)
            combined = torch.bmm(H_soft.transpose(1,2), Th).squeeze(dim=-1) # Don't want to squeeze batch dim!
        else:
            combined = torch.bmm(self.linear(H).transpose(1,2), Th).squeeze(dim=-1) # Don't want to squeeze batch dim!
            combined = F.log_softmax(combined)

        return combined



class linear_vector_aggregator(nn.Module):
    """ Linear aggregator for interpretable classification.

        Aggregates a set of concept representations and their
        scores, generates a prediction probability output from them.

        Args:
            din (int): input concept dimension
            dout (int): output dimension (num classes)

        Inputs:
            H:  H(x) matrix of concepts (b x k x c_dim) [TODO: generalize to set maybe?]
            Th:  Theta(x) vector of concept scores (b x k x 1) (TODO: generalize to multi-class scores)

        Output:
            - Vector of class probabilities (b x o_dim x 1)

        TODO: add number of layers as argument, construct in for?
    """

    def __init__(self, din, dout, softmax_pre = True):
        super(linear_vector_aggregator, self).__init__()
        self.din    = din
        self.dout   = dout
        self.linear = nn.Linear(din, dout)
        self.softmax_pre = softmax_pre

    def forward(self, H, Th):
        assert(H.size(-2) == Th.size(-2))
        assert(H.size(-1) == self.din)

        # Previously:
        #combined = torch.bmm(H.transpose(1,2), Th).squeeze(dim=-1) # Don't want to squeeze batch dim!

        # New - do softmax before aggregation:
        if self.softmax_pre:
            H_soft = F.log_softmax(self.linear(H), dim=2)
            combined = torch.bmm(H_soft.transpose(1,2), Th).squeeze(dim=-1) # Don't want to squeeze batch dim!
        else:
            combined = torch.bmm(self.linear(H).transpose(1,2), Th).squeeze(dim=-1) # Don't want to squeeze batch dim!
            combined = F.log_softmax(combined)

        return combined


# Set version:
#     def forward(self, h, th):
#         assert(h.size(0) == th.size(0))
#         agg = torch.zeros(self.dout, 1)
#         for k in len(h):
#             agg += self.concept_encoder(h[k],th[k])
#         return agg
