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

import sys, os
import numpy as np
import pdb
import pandas as pd
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Local imports
from os.path import dirname, realpath
from SENN.arglist import parse_args
from SENN.utils import plot_theta_stability, generate_dir_names
from SENN.eval_utils import sample_local_lipschitz

from SENN.models import GSENN
from SENN.conceptizers import input_conceptizer
from SENN.parametrizers import  dfc_parametrizer
from SENN.aggregators import additive_scalar_aggregator
from SENN.trainers import VanillaClassTrainer, GradPenaltyTrainer

from robust_interpret.utils import lipschitz_feature_argmax_plot

def find_conflicting(df, labels, consensus_delta = 0.2):
    """
        Find examples with same exact feat vector but different label.
        Finds pairs of examples in dataframe that differ only
        in a few feature values.

        Args:
            - differ_in: list of col names over which rows can differ
    """
    def finder(df, row):
        for col in df:
            df = df.loc[(df[col] == row[col]) | (df[col].isnull() & pd.isnull(row[col]))]
        return df

    groups = []
    all_seen = set([])
    full_dups = df.duplicated(keep='first')
    for i in tqdm(range(len(df))):
        if full_dups[i] and (not i in all_seen):
            i_dups = finder(df, df.iloc[i])
            groups.append(i_dups.index)
            all_seen.update(i_dups.index)

    pruned_df  = []
    pruned_lab = []
    for group in groups:
        scores = np.array([labels[i] for i in group])
        consensus = round(scores.mean())
        for i in group:
            if (abs(scores.mean() - 0.5)< consensus_delta) or labels[i] == consensus:
                # First condition: consensus is close to 50/50, can't consider this "outliers", so keep them all
                #print(scores.mean(), len(group))
                pruned_df.append(df.iloc[i])
                pruned_lab.append(labels[i])
    return pd.DataFrame(pruned_df), np.array(pruned_lab)

def load_compas_data(valid_size=0.1, shuffle=True, random_seed=2008, batch_size=64):
    df= pd.read_csv("/Users/david/pkg/fairml/doc/example_notebooks/propublica_data_for_fairml.csv")
    # Binarize num of priors var? Or normalize it 0,1?
    df['Number_of_Priors'] = np.sqrt(df['Number_of_Priors'])/(np.sqrt(38))
    compas_rating = df.score_factor.values # This is the target??
    df = df.drop("score_factor", 1)

    pruned_df, pruned_rating = find_conflicting(df, compas_rating)
    x_train, x_test, y_train, y_test   = train_test_split(pruned_df, pruned_rating, test_size=0.1, random_state=85)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=85)

    feat_names = list(x_train.columns)
    x_train = x_train.values # pandas -> np
    x_test  = x_test.values

    Tds = []
    Loaders = []
    for (foldx, foldy) in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
        scaler = StandardScaler(with_std=False, with_mean=False) # DOn't scale to make consitient with LIME/SHAP script
        transformed = scaler.fit_transform(foldx)
        #transformed = foldx
        tds = TensorDataset(torch.from_numpy(transformed).float(),
                            torch.from_numpy(foldy).view(-1, 1).float())
        loader = DataLoader(tds, batch_size=batch_size, shuffle=False)
        Tds.append(tds)
        Loaders.append(loader)

    return (*Loaders, *Tds, df, feat_names)

def main():
    args = parse_args()
    args.nclasses = 1
    args.theta_dim = args.nclasses
    args.print_freq = 100
    args.epochs = 10
    train_loader, valid_loader, test_loader, train, valid, test, data, feat_names  = load_compas_data()

    layer_sizes = (10,10,5)
    input_dim = 11

    if args.h_type == 'input':
        conceptizer  = input_conceptizer()
        args.nconcepts = 11 + int(not args.nobias)
    elif args.h_type == 'fcc':
        args.nconcepts +=     int(not args.nobias)
        conceptizer  = image_fcc_conceptizer(11, args.nconcepts, args.concept_dim) #, sparsity = sparsity_l)
    else:
        raise ValueError('Unrecognized h_type')

    model_path, log_path, results_path = generate_dir_names('compas', args)


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
        raise ValueError('Unrecognized theta_reg_type')

    trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)

    trainer.plot_losses(save_path=results_path)

    # Load Best One
    checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'))
    model = checkpoint['model']

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

    lips, argmaxes = sample_local_lipschitz(model, test, mode = 2, top_k = 10, max_distance = 3)

    max_lip = lips.max()
    imax = np.unravel_index(np.argmax(lips), lips.shape)[0]
    jmax = argmaxes[imax][0][0]
    print('Max Lip value: {}, attained for pair ({},{})'.format(max_lip, imax, jmax))

    x      = test.data_tensor[imax]
    argmax = test.data_tensor[jmax]

    pred_x = model(Variable(x.view(1,-1), volatile = True)).data
    att_x = model.thetas.data.squeeze().numpy().squeeze()

    pred_argmax = model(Variable(argmax.view(1,-1), volatile = True)).data
    att_argmax = model.thetas.data.squeeze().numpy().squeeze()

    pdb.set_trace()
    results['x_max']      = x
    results['x_argmax']      = argmax
    results['test_discrete_glip']      = lips
    results['test_discrete_glip_argmaxes'] = argmaxes


    print('Local g-Lipschitz estimate: {:8.2f}'.format(lips.mean()))

    fpath = os.path.join(results_path, 'discrete_lip_gsenn')
    ppath = os.path.join(results_path, 'relevance_argmax_gsenn')

    pickle.dump(results,  open(fpath+'.pkl',"wb")) # FOrmerly model_metrics

    lipschitz_feature_argmax_plot(x, argmax, att_x, att_argmax,
                                  feat_names = feat_names,
                                  digits=2, figsize=(5,6), widths=(2,3),
                                  save_path=ppath + '.pdf')


if __name__ == "__main__":
    main()
