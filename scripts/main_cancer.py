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

# Standard Imports
import sys, os
import numpy as np
import pdb
import pickle
import argparse
import operator
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Torch Imports
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader

# Scikit-learn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Local imports
from os.path import dirname, realpath
sys.path.append(os.path.join(dirname(realpath(__file__)),'codebase/'))


from SENN.arglist import parse_args
from SENN.utils import plot_theta_stability, generate_dir_names
from SENN.eval_utils import sample_local_lipschitz, estimate_dataset_lipschitz
from SENN.models import GSENN
from SENN.conceptizers import input_conceptizer
from SENN.parametrizers import  dfc_parametrizer
from SENN.aggregators import additive_scalar_aggregator
from SENN.trainers import VanillaClassTrainer, GradPenaltyTrainer

def load_cancer_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size=64):
    data = pd.read_csv("../data/Cancer/data.csv")
    x = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
    diag = { "M": 1, "B": 0}
    y = data["diagnosis"].replace(diag)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=85)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=85)
    Tds = []
    Loaders = []
    for (foldx, foldy) in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
        scaler = StandardScaler()
        transformed = scaler.fit_transform(foldx)
        tds = TensorDataset(torch.from_numpy(transformed).float(),
                            torch.from_numpy(foldy.as_matrix()).view(-1, 1).float())
        loader = DataLoader(tds, batch_size=batch_size, shuffle=False)
        Tds.append(tds)
        Loaders.append(loader)

    return (*Loaders, *Tds, data)


def main():
    np.random.seed(2018)
    args = parse_args()
    args.nclasses = 1
    args.theta_dim = args.nclasses
    args.print_freq = 100
    args.epochs = 10
    train_loader, valid_loader, test_loader, train, valid, test, data  = load_cancer_data()

    layer_sizes = (10,10,5)
    input_dim = 30

    # model

    if args.h_type == 'input':
        conceptizer  = input_conceptizer()
        args.nconcepts = input_dim + int(not args.nobias)
    elif args.h_type == 'fcc':
        args.nconcepts +=     int(not args.nobias)
        conceptizer  = image_fcc_conceptizer(11, args.nconcepts, args.concept_dim) #, sparsity = sparsity_l)
    else:
        raise ValueError('Unrecognized h_type')

    model_path, log_path, results_path = generate_dir_names('cancer', args)


    parametrizer = dfc_parametrizer(input_dim, *layer_sizes, args.nconcepts, args.theta_dim)

    aggregator   = additive_scalar_aggregator(args.concept_dim,args.nclasses)

    model        = GSENN(conceptizer, parametrizer, aggregator)#, learn_h = args.train_h)

    if args.theta_reg_type == 'unreg':
        trainer = VanillaClassTrainer(model, args)
    elif args.theta_reg_type == 'grad1':
        trainer = GradPenaltyTrainer(model, args, typ = 1)
    elif args.theta_reg_type == 'grad2':
        trainer = GradPenaltyTrainer(model, args, typ = 2)
    elif args.theta_reg_type == 'grad3':
        trainer = GradPenaltyTrainer(model, args, typ = 3)
    elif args.theta_reg_type == 'crosslip':
        trainer = CLPenaltyTrainer(model, args)
    else:
        raise ValueError('Unrecoginzed theta_reg_type')


    trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)

    trainer.plot_losses(save_path=results_path)

    # Load Best One
    checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'))
    model = checkpoint['model']
    model        = GSENN(conceptizer, parametrizer, aggregator)#, learn_h = args.train_h)


    results = {}

    train_acc = trainer.validate(train_loader, fold = 'train')
    valid_acc = trainer.validate(valid_loader, fold = 'valid')
    test_acc = trainer.validate(test_loader, fold = 'test')

    results['train_accuracy'] = train_acc
    results['valid_accuracy']  = valid_acc
    results['test_accuracy']  = test_acc
    print('Train accuracy: {:8.2f}'.format(train_acc))
    print('Valid accuracy: {:8.2f}'.format(valid_acc))
    print('Test accuracy: {:8.2f}'.format(test_acc))


    #noise_stability_plots(model, test_tds, cuda = args.cuda, save_path = results_path)

    lips, argmaxes = sample_local_lipschitz(model, test, mode = 2, top_k = 10, max_distance = 1)


    results['test_discrete_glip']      = lips
    results['test_discrete_glip_argmaxes'] = argmaxes


    print('Local discrete g-Lipschitz estimate: {:8.2f}'.format(lips.mean()))


    pointwise_test_loader = dataloader.DataLoader(test, shuffle=True, batch_size=1,num_workers=4)
    # Need to run in mode one, h is identity here
    Lips_mean, Lips = estimate_dataset_lipschitz(model, pointwise_test_loader,
                continuous=True, mode = 1, eps = 0.2, tol = 1e-2, maxpoints=100,
                maxit = 1e3, log_interval = 10, patience = 5, cuda= args.cuda, verbose = True)

    results['test_cont_glip'] = Lips

    print('Local dcontinuous g-Lipschitz estimate: {:8.2f}'.format(Lips.mean()))


    pickle.dump(results, open(results_path + '/model_metrics.pkl', "wb"))


if __name__ == "__main__":
    main()
