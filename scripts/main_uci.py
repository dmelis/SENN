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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import glob
# Necessary for parallelism in joblib + keras
os.environ['JOBLIB_START_METHOD'] = 'forkserver'
# 0: all logs shown, 1: filter out INFO logs 2: filter out WARNING logs, and 3 to additionally filter out ERROR log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tempfile
import sys
sys.path.insert(0, os.path.abspath('..'))
import warnings
from functools import partial
import scipy
import numpy as np
import matplotlib.pyplot as plt

import pdb
import pickle
import argparse
import operator
import shutil


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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata

# Local imports
#from os.path import dirname, realpath
#sys.path.append(os.path.join(dirname(realpath(__file__)), 'robust_interpret/'))
import robust_interpret


from robust_interpret.explainers import gsenn_wrapper, lime_wrapper, shap_wrapper
from lime.wrappers.scikit_image import SegmentationAlgorithm

#from utils import deepexplain_plot, lipschitz_boxplot, lipschitz_argmax_plot
#from robust_interpret.utils import generate_dir_names  # plot_theta_stability,
from robust_interpret.utils import plot_attribution_stability
from robust_interpret.utils import lipschitz_feature_argmax_plot

from SENN.utils import plot_prob_drop
from SENN.arglist import get_senn_parser #parse_args as parse_senn_args
from SENN.utils import plot_theta_stability, generate_dir_names
from SENN.eval_utils import sample_local_lipschitz

from SENN.models import GSENN
from SENN.conceptizers import input_conceptizer
from SENN.parametrizers import  dfc_parametrizer
from SENN.aggregators import additive_scalar_aggregator
from SENN.trainers import VanillaClassTrainer, GradPenaltyTrainer

def grid_search_train(dataname, input_dim, layer_dims, nclasses, args, train_loader, valid_loader, save_path =None):
    crit = 'crossent' if nclasses > 1 else 'bce_logit'
    params = {
        'seed': range(3),
        'lr': [0.0005, 0.001, 0.002, 0.005, 0.01],
        #'reg': [0, 0.0001, 0.001, 0.01],
        #'lr': [0.01, 0.02, 0.05, 0.1],
        'epochs': [5, 10, 20, 40],
        'batch': [8,16,32,64,128],
        #'nonlin': ['relu','sigmoid'] 
    }

    param_names = list(params.keys())
    vals = list((params.values()))

        # # if dataname in check_completed_datasets(results_path) or (dataname is 'abalone'):
        # #     print('{} found, skipping'.format(dataname))
        # #     continue
        # train_loader, valid_loader, test_loader, \
        # train, valid, test, dataset, features, classes, binary  = load_uci_data(dataname)


    val_accs = {}
    best_model = None
    best_acc   = 0
    for partuple in itertools.product(*vals):
        print('='*10)
        param_dict = dict(zip(param_names,partuple))
        print(param_dict)
        args.lr  = param_dict['lr']
        args.batch  = param_dict['batch']
        train_loader, valid_loader,_,_,_,_,_,_,_,_  = load_uci_data(dataname,
        random_seed=param_dict['seed'], batch_size= param_dict['batch'])
        trainer = make_model(input_dim, layer_dims, nclasses, args)
        trainer.train(train_loader, valid_loader,epochs = param_dict['epochs'])
        val_acc = trainer.validate(valid_loader, fold = 'valid')
        val_accs[partuple] = val_acc
        if val_acc > best_acc:
            best_model = trainer
            best_acc   = val_acc
            if best_acc > 99.95:
                break

    best_params = max(val_accs.items(), key=operator.itemgetter(1))[0]
    is_best = True

    print(val_accs[best_params])
    print('Best parameters:')
    for k, v in dict(zip(param_names, best_params)).items():
        print('{:15} = {:>5}'.format(k,v))

    best_params = dict(zip(params.keys(), best_params))
    trainer = make_model(input_dim, layer_dims, nclasses, args)
    trainer.train(train_loader, valid_loader, epochs = best_params['epochs'], save_path = save_path)
    args.lr  = best_params['lr']
    args.batch  = best_params['batch']
    return trainer, best_params


def load_uci_data(dataname, valid_size=0.1, shuffle=True, random_seed=2008, batch_size=64):
    dataset = fetch_mldata(dataname, target_name='label', data_name='data',transpose_data=True)
    print(dataset['DESCR'])
    feat_names = dataset['COL_NAMES']
    #print(feat_names)
    y = dataset['target']
    if scipy.sparse.issparse(y):
        y = np.array(y.todense().transpose())
        y = np.argmax(y, axis = 1).flatten()
    #print(dataset.data.shape)

    feat_names = ['X'+ str(i) for i in range(dataset.data.shape[1])]
    dataset.feat_names = feat_names
    classes = np.unique(y)
    classes.sort()
    if len(set(classes)) == 2:
        class_names = ['Negative', 'Positive']
        binary = True
    else:
        class_names = ['C' + str(i) for i in classes]
        binary = False


    if binary and min(classes) == -1:
        y[y==-1] = 0
    elif binary and (set(classes) != set([0, 1])):
        y[y==classes[0]] = 0
        y[y==classes[1]] = 1
    else:
        if min(classes) == 1:
            y -= 1
            classes -= 1

    if int(y.max() + 1) != len(classes):
        #Some classes not represented in dataset. Cheap fix: pretend they do exist, but haven't seen dthe,
        classes = list(range(int(y.max() + 1)))

    nclasses = 1 if binary else len(classes)

    x_train, x_test, y_train, y_test = \
        train_test_split(dataset.data, y, train_size=0.80)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, \
                                                test_size=valid_size, random_state=random_seed)

    Tds = []
    Loaders = []
    #np_data = []
    for (foldx, foldy) in [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]:
        scaler = StandardScaler(with_std=False, with_mean=False) # DOn't scale to make consitient with LIME/SHAP script
        transformed = scaler.fit_transform(foldx)
        #transformed = foldx
        # tds = TensorDataset(torch.from_numpy(transformed).float(),
        #                     torch.from_numpy(foldy).view(-1, 1).float())
        tdx = torch.from_numpy(transformed).float()
        if binary:
            tdy = torch.from_numpy(foldy).view(-1, 1).float()
        else:
            tdy = torch.from_numpy(foldy).long()

        tds = TensorDataset(tdx,tdy)


        loader = DataLoader(tds, batch_size=batch_size, shuffle=False)
        Tds.append(tds)
        Loaders.append(loader)

    return (*Loaders, *Tds, dataset, feat_names, classes, binary)

def train_classifier(x_train, y_train, x_test, y_test):
    classif = RandomForestClassifier(n_estimators=1000)
    classif.fit(x_train, y_train)
    print('Random Forest Classif Accuracy {:4.2f}\%'.format(100*classif.score(x_test, y_test)))
    return classif

def parse_args():

    senn_parser = get_senn_parser()
    # [args_senn, extra_args] = senn_parser.parse_known_args()

    #
    ### Local ones
    parser = argparse.ArgumentParser(parents =[senn_parser],add_help=False,
        description='Interpteratbility robustness evaluation on MNIST')

    # #setup
    parser.add_argument('-d','--datasets', nargs='+',
                        default = ['heart', 'ionosphere', 'breast-cancer','wine','heart',
                        'glass','diabetes','yeast','leukemia','abalone'], help='<Required> Set flag')
    parser.add_argument('--lip_calls', type=int, default=10,
                        help='ncalls for bayes opt gp method in Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=0.01,
                        help='eps for Lipschitz estimation')
    parser.add_argument('--lip_points', type=int, default=100,
                        help='sample size for dataset Lipschitz estimation')
    parser.add_argument('--optim', type=str, default='gp',
                        help='black-box optimization method')

    #####

    args = parser.parse_args()

    # # update args and print
    # args.filters = [int(k) for k in args.filters.split(',')]
    # if args.objective == 'mse':
    #     args.num_class = 1

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args

def check_completed_datasets(results_path):
    existing = list(glob.iglob(results_path + '/*.pkl'))
    done = set([os.path.splitext(os.path.basename(fpath))[0].split('_')[0] for fpath in existing])
    return done


def make_model(input_dim, layer_dims, nclasses, args):
    if args.h_type == 'input':
        conceptizer  = input_conceptizer()
        args.nconcepts = input_dim + int(not args.nobias)
    elif args.h_type == 'fcc':
        #args.nconcepts +=     int(not args.nobias)
        conceptizer  = image_fcc_conceptizer(input_dim, args.nconcepts, args.concept_dim) #, sparsity = sparsity_l)
    else:
        raise ValueError('Unrecognized h_type')

    #model_path, log_path, results_path = generate_dir_names('uci', args)

    parametrizer = dfc_parametrizer(input_dim, *layer_dims, args.nconcepts, args.theta_dim)
    aggregator   = additive_scalar_aggregator(args.concept_dim, nclasses)
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

    return trainer


def generate_dir_names(base, dataname, args, make = True):
    if args.h_type == 'input':
        suffix = '{}_H{}_Reg{:0.0e}'.format(
                    args.theta_reg_type,
                    args.h_type,
                    args.theta_reg_lambda
                    #args.lr, will be changed in grid
                    )
    else:
        suffix = '{}_H{}_Cpts{}_Reg{:0.0e}_Sp{}'.format(
                    args.theta_reg_type,
                    args.h_type,
                    args.nconcepts,
                    args.theta_reg_lambda,
                    args.h_sparsity
                    #args.lr,
                    )

    model_path     = os.path.join(args.model_path, base,suffix, dataname)
    log_path       = os.path.join(args.log_path, base,suffix, dataname)
    results_path   = os.path.join(args.results_path, base,suffix, dataname)

    if make:
        for p in [model_path, log_path, results_path]:
            if not os.path.exists(p):
                os.makedirs(p)

    return model_path, log_path, results_path


# def experiment_1():
#


def main():
    args = parse_args()
    print(" ******* Remember to revert to *4 calls in (2) !!!!!!!!!")

    do_consistency = True
    do_bb_stability_example = True
    do_bb_stability = True

    classif_dataset_names = args.datasets #['leukemia', 'ionosphere', 'breast-cancer' ,'abalone'] #'diabetes',

    for dataname in classif_dataset_names:
        All_Results = {'args': args}
        model_path, log_path, results_path = generate_dir_names(
            'uci',dataname, args, make=not args.debug)


        # if dataname in check_completed_datasets(results_path) or (dataname is 'abalone'):
        #     print('{} found, skipping'.format(dataname))
        #     continue
        train_loader, valid_loader, test_loader, \
        train, valid, test, dataset, features, classes, binary  = load_uci_data(dataname)

        nclasses = len(classes) if not binary else 1
        args.theta_dim = nclasses
        args.nclasses = nclasses # Used by trainer to define criterion

        input_dim = len(features)
        layer_dims = (int(input_dim/2),10,5)

        if args.train or (not os.path.isfile(os.path.join(model_path,dataname,'model_best.pth.tar'))):
            
            print('Will train model from scratch')
            trainer,_ = grid_search_train(dataname, input_dim, layer_dims,
             nclasses, args, train_loader, valid_loader, save_path = model_path)
            #trainer = make_model(input_dim, layer_dims, nclasses, args)
            #trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)
            #trainer.plot_losses(save_path=results_path)
            model = trainer.model
        else:
            # Load Best One
            checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'))
            model = checkpoint['model']
            # Dummy trainer just to compute eval
            trainer =  VanillaClassTrainer(model, args)

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


        SENN_wrap = gsenn_wrapper(model,
                            mode      = 'classification',
                            input_type = 'feature',
                            multiclass=True,
                            feature_names = features,
                            class_names   = classes,
                            train_data      = train_loader,
                            skip_bias = True,
                            verbose = False)


        x_test_np,_ = zip(*test)
        x_test_np = np.array([e.numpy() for e in x_test_np])
        maxpoints = 20
        mini_test = x_test_np[np.random.choice(len(x_test_np), min(maxpoints,len(x_test_np)), replace=False)]


        All_Results = {'model_performance': results}


        ### 1. Consistency analysis over multiple samples
        if do_consistency:
            print('**** Performing consistency estimation over subset of dataset ****')
            corrs = SENN_wrap.compute_dataset_consistency(mini_test)
            Consistency_dict = {'corrs': corrs}
            pickle.dump(Consistency_dict, open(results_path + '_consistency.pkl', "wb"))
            All_Results['Consistency'] = corrs


        ### 2. Single example lipschitz estimate with Black Box
        if do_bb_stability_example:
            print('**** Performing lipschitz estimation for a single point ****')
            Argmax_dict = {}#{k: {} for k in explainer_dict.keys()}

            idx = 0
            print('Example index: {}'.format(idx))
            x = x_test_np[idx]
            lip, argmax = SENN_wrap.local_lipschitz_estimate(x, bound_type='box_std',
                                                        optim=args.optim,
                                                        eps=args.lip_eps,
                                                        n_calls=args.lip_calls,
                                                        verbose=2)
            att_x = SENN_wrap(x, None, show_plot=False).squeeze()
            att_argmax = SENN_wrap(argmax, None, show_plot=False).squeeze()
            Argmax_dict = {'lip': lip, 'argmax': argmax, 'x': x}
            fpath = results_path + '_argmax_lip'
            pickle.dump(Argmax_dict, open(fpath+'.pkl',"wb"))
            if x_test_np.shape[1] < 30:
                # Beyond 30 is hard to visualize
                lipschitz_feature_argmax_plot(x, argmax, att_x, att_argmax,
                                              feat_names = SENN_wrap.feature_names,
                                              save_path=fpath + '.pdf')

            All_Results['lipshitz_argmax'] = Argmax_dict


        ### 3. Local lipschitz estimate over multiple samples with Black BOx Optim
        if do_bb_stability:
            print('**** Performing black-box lipschitz estimation over subset of dataset ****')
            lips = SENN_wrap.estimate_dataset_lipschitz(mini_test,
                                               n_jobs=-1, bound_type='box_std',
                                               eps=args.lip_eps, optim=args.optim,
                                               n_calls=args.lip_calls, verbose=0)
            Stability_dict = {'lips': lips}
            pickle.dump(Stability_dict, open(results_path + '_stability_blackbox.pkl', "wb"))
            All_Results['stability_blackbox'] = lips

        # for k, expl in explainer_dict.items():
        #     Lips = []
        #     if k in ['LIME', 'SHAP']:
        #         Lips = expl.estimate_dataset_lipschitz(mini_test[:40],  # LIME is too slow!
        #                                                n_jobs=1, bound_type='box_std',
        #                                                eps=args.lip_eps, optim=args.optim,
        #                                                n_calls=args.lip_calls, verbose=0)
        #     else:
        #         Lips = expl.estimate_dataset_lipschitz(mini_test,
        #                                                n_jobs=-1, bound_type='box_std',
        #                                                eps=args.lip_eps, optim=args.optim,
        #                                                n_calls=args.lip_calls, verbose=0)
        #     LipResults[k]['g_lip_dataset'] = Lips
        #     print('Local g-Lipschitz estimate for {}: {:8.2f}'.format(k, Lips.mean()))
        #     pickle.dump(LipResults[k], open(results_path+'/{}_robustness_metrics_{}.pkl'.format(dataname, k), "wb"))

        #df_lips = pd.DataFrame({k: LipResults[k]['g_lip_dataset'] for k in Results.keys()})
        # lipschitz_boxplot(df_lips, continuous=True,
        #                   save_path=os.path.join(results_path, 'local_glip_comparison.pdf'))

        pickle.dump(All_Results, open(results_path + '_combined_metrics.pkl'.format(dataname), "wb"))


if __name__ == '__main__':
    main()
