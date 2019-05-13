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

import os
import tqdm
import time
import pdb
import shutil
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from attrdict import AttrDict
import matplotlib.pyplot as plt

#from fairml import plot_dependencies

from .utils import AverageMeter


#===============================================================================
#====================      REGULARIZER UTILITIES    ============================
#===============================================================================

def tvd_loss(theta):
    loss = lambd * (
        torch.sum(torch.abs(params[:, :, :, :-1] - params[:, :, :, 1:])) +
        torch.sum(torch.abs(params[:, :, :-1, :] - params[:, :, 1:, :]))
    )
    return loss

def CL_loss(theta, n_class):
    """ Cross lipshitc loss from https://arxiv.org/pdf/1705.08475.pdf.
        Gradient based.
    """

    total = 0
    for i in range(n_class):
        for j in range(n_class):
            total += (grad[i] - grad[j]).norm()**2

    return total/(n_class)

def compute_jacobian_sum(x, fx):
    """ Much faster than compute_jacobian, but only correct for norm L1 stuff
    since it returns sum of gradients """
    n = x.size(-1)
    b = x.size(0)
    c = fx.size(-1)
    m = fx.size(-2)
    grad = torch.ones(b, m, c)
    if x.is_cuda:
        grad  = grad.cuda()
    g = torch.autograd.grad(outputs=fx, inputs = x, grad_outputs = grad, create_graph=True, only_inputs=True)[0]#, retain_graph = True)[0] -> not sure this should be true or not. Not needed! Defaults to value of create_graph
    return g

def compute_jacobian(x, fx):
    # Ideas from https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059/2
    
    
    
    b = x.size(0)
    n = x.size(-1)
    # if fx.dim() > 1:
    m = fx.size(-1)
    # else:
    #     #e.g. fx = theta and task is binary classifiction, fx is a vector
    #     m = 1
    #print(fx.size())
    #print(b,n,m)
    J = []
    for i in range(m):
        #print(i)
        grad = torch.zeros(b, m)
        grad[:,i] = 1
        if x.is_cuda:
            grad  = grad.cuda()
        #print(grad.size(), fx.size(), x.size())
        #pdb.set_trace()
        g = torch.autograd.grad(outputs=fx, inputs = x, grad_outputs = grad, create_graph=True, only_inputs=True)[0] #, retain_graph = True)[0]
        J.append(g.view(x.size(0),-1).unsqueeze(-1))
    #print(J[0].size())
    J = torch.cat(J,2)
    return J


#===============================================================================
#==================================   TRAINERS    ==============================
#===============================================================================


def save_checkpoint(state, is_best, outpath):
    # script_dir = dirname(dirname(realpath(__file__)))
    if outpath == None:
        outpath = os.path.join(script_dir, 'checkpoints')

    #outdir = os.path.join(outpath, '{}_LR{}_Lambda{}'.format(state['theta_reg_type'],state['lr'],state['theta_reg_lambda']))
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = os.path.join(outpath, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outpath,'model_best.pth.tar'))


class ClassificationTrainer():
    def __init__(self, model, args):
    #loss_type = 'ce', opt = 'adam', lr = 0.0002, cuda = False, log_interval = 100):
        #super(ClassificationTrainer, self).__init__()
        self.model = model
        self.args = args
        self.cuda = args.cuda

        self.nclasses = args.nclasses

        if args.nclasses <= 2 and args.objective == 'bce':
            self.prediction_criterion = F.binary_cross_entropy_with_logits
        elif args.nclasses <= 2:# THis will be default.  and args.objective == 'bce_logits':
            self.prediction_criterion = F.binary_cross_entropy # NOTE: This does not do sigmoid itslef
        elif args.objective == 'cross_entropy':
            self.prediction_criterion = F.cross_entropy
        else:
            self.prediction_criterion = F.nll_loss # NOTE: To be used with output of log_softmax

        if args.h_type != 'input':
            # Means conceptizer will be trained, need reconstruction loss for it
            self.learning_h = True
            self.h_reconst_criterion = F.mse_loss  #nn.MSELoss() 
            # if args.h_sparsity != -1:
            #     print('Will enforce sparsity on h')
            self.h_sparsity = args.h_sparsity
        else:
            self.learning_h = False

        self.loss_history = []  # Will acumulate losse
        self.print_freq = args.print_freq

        self.reset_lstm = model.reset_lstm # Trun on when model has an lstm

        #self.outpath = args.save_dir

        optim_betas = (0.9, 0.999)

        if args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr, betas=optim_betas)
        elif args.opt == 'rmsprop':
            #lrD, lrG = 5e-5, 5e-5 # Original WGAN code has 5e-5
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = args.lr)
        elif args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum=0.9)


        if self.cuda:
            self.model = self.model.cuda()
            #self.prediction_criterion = self.prediction_criterion.cuda()
            # if self.learning_h:
            #     self.h_reconst_criterion  = self.h_reconst_criterion.cuda()



    def train(self, train_loader, val_loader = None, epochs = 10, save_path = None):
        best_prec1 = 0
        for epoch in range(epochs):
            self.train_epoch(epoch, train_loader)

            if val_loader is not None:
                val_prec1 = self.validate(val_loader) # Ccompytes acc

            # Maybe add here computing of empirical robustness?

            # remember best prec@1 and save checkpoint
            is_best = val_prec1 > best_prec1
            best_prec1 = max(val_prec1, best_prec1)
            if save_path is not None:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'lr': self.args.lr,
                    'theta_reg_lambda': self.args.theta_reg_lambda,
                    'theta_reg_type': self.args.theta_reg_type,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : self.optimizer.state_dict(),
                    'model': self.model  
                 }, is_best, save_path)

        print('Training done')

    def train_batch(self):
        raise NotImplemented('ClassificationTrainers must define their train_batch method!')


    def concept_learning_loss(self, inputs, all_losses):
        recons_loss    = self.h_reconst_criterion(self.model.recons,
                                Variable(inputs.data, requires_grad = False))

        all_losses['reconstruction'] = recons_loss.data[0]
        if self.h_sparsity != -1:
            sparsity_loss   = self.model.h_norm_l1.mul(self.h_sparsity)
            all_losses['h_sparsity'] = sparsity_loss.data[0]
            recons_loss += sparsity_loss
        return recons_loss

    def train_epoch(self, epoch, train_loader):
        """
            Does mostly accounting. The actual trianing is done by the train_batch method.
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()

        for i, (inputs, targets) in enumerate(train_loader, 0):
            # measure data loading time
            data_time.update(time.time() - end)

            # get the inputs
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = Variable(inputs), Variable(targets)

            if self.reset_lstm:
                self.model.zero_grad()
                self.model.parametrizer.hidden = self.model.parametrizer.init_hidden()# detaching it from its history on the last instance.

            outputs, loss, loss_dict = self.train_batch(inputs, targets)
            loss_dict['iter'] = i + (len(train_loader)*epoch)
            
            # the dict here
            self.loss_history.append(loss_dict)

            # measure accuracy and record loss
            if self.nclasses > 4:
                prec1, prec5 = self.accuracy(outputs.data, targets.data, topk=(1, 5))
            elif self.nclasses in [3,4]:
                prec1, prec5 = self.accuracy(outputs.data, targets.data, topk=(1,self.nclasses))
            else:
                prec1, prec5 = self.binary_accuracy(outputs.data, targets.data), [100]

            #
            # if self.nclasses <= 2:
            #     prec1 = self.binary_accuracy(outputs.data, targets.data)
            #     prec5 = [100]
            # else:
            #     prec1, prec5 = self.accuracy(outputs.data, targets.data, topk=(1,5))
            losses.update(loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]  '
                      'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                      #'Data {data_time.val:.2f} ({data_time.avg:.2f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))


    def validate(self, val_loader, fold = None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):
            # get the inputs
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            input_var = torch.autograd.Variable(inputs, volatile=True)
            target_var = torch.autograd.Variable(targets, volatile=True)

            # compute output
            output = self.model(input_var)
            loss   = self.prediction_criterion(output, target_var)

            # measure accuracy and record loss
            if self.nclasses > 4:
                prec1, prec5 = self.accuracy(output.data, targets, topk=(1, 5))
            elif self.nclasses == 3:
                prec1, prec5 = self.accuracy(output.data, targets, topk=(1,3))
            else:
                prec1, prec5 = self.binary_accuracy(output.data, targets), [100]

            losses.update(loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg

    def evaluate(self, test_loader, fold = None):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, targets in test_loader:
            if self.cuda:
                data, targets = data.cuda(), targets.cuda()

            data, targets = Variable(data, volatile=True), Variable(targets)

            if self.reset_lstm:
                self.model.zero_grad()
                self.model.parametrizer.hidden = self.model.parametrizer.init_hidden()# detaching it from its history on the last instance.

            output = self.model(data)
            #test_loss += self.prediction_criterion(output, targets.view(targets.size(0))).data[0]
            test_loss += self.prediction_criterion(output, targets).data[0]

            #print(output)
            if self.nclasses == 2:
                pred = output.data.round()
            else:
                pred = output.data.max(1)[1] # get the index of the max log-probability
            #print(pred, targets)
            correct += pred.eq(targets.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(test_loader) # loss function already averages over batch size
        fold = '' if (fold is None) else ' (' + fold + ')'
        acc = 100. * correct / len(test_loader.dataset)
        print('\nEvaluation{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            fold, test_loss, correct, len(test_loader.dataset),acc))
        return acc

    def binary_accuracy(self, output, target):
        """Computes the accuracy"""
        return torch.Tensor(1).fill_((output.round().eq(target)).float().mean()*100)

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def accuracy_per_class(self, model, test_loader, classes):
        """ TODO: Homogenize with accuracy style and synbtax"""
        n = len(classes)
        class_correct = list(0. for i in range(n))
        class_total = list(0. for i in range(n))
        confusion_matrix = ConfusionMeter(n) #I have 2 classes here
        for data in test_loader:
            inputs, labels = data
            if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            confusion_matrix.add(predicted, labels)
            for i in range(labels.size()[0]):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1

    def plot_losses(self, save_path = None):
        loss_types = [k for k in self.loss_history[0].keys() if k != 'iter']
        losses = {k: [] for k in loss_types}
        iters  = []
        for e in self.loss_history:
            iters.append(e['iter'])
            for k in loss_types:
                losses[k].append(e[k])
        fig, ax = plt.subplots(1,len(loss_types), figsize = (4*len(loss_types), 5))
        if len(loss_types) == 1:
            ax = [ax] # Hacky, fix
        for i, k in enumerate(loss_types):
            ax[i].plot(iters, losses[k])
            ax[i].set_title('Loss: {}'.format(k))
            ax[i].set_xlabel('Iters')
            ax[i].set_ylabel('Loss')
        if save_path is not None:
            plt.savefig(save_path + '/training_losses.pdf', bbox_inches = 'tight', format='pdf', dpi=300)
        #plt.show(block=False)


"""

    Since the train_batch method abstracts away most of the details of training,
    just need to specifiy that for every training scheme. Everything else is
    shared.

"""



class VanillaClassTrainer(ClassificationTrainer):
    """
        The simplest classification trainer. No regularization, just normal
        prediction loss.
    """
    def __init__(self, model, args):
        super().__init__(model, args)

    def train_batch(self, inputs, targets):
        """ inputs, targets already variables """
        self.optimizer.zero_grad()
        pred = self.model(inputs)

        # Loss
        try:
            pred_loss       = self.prediction_criterion(pred, targets)
        except:
            pdb.set_trace()
        all_losses = {'prediction': pred_loss.data[0]}
        if self.learning_h:
            h_loss = self.concept_learning_loss(inputs, all_losses)
            loss = pred_loss + h_loss
        else:
            loss = pred_loss

        # Keep track of losses
        #self.loss_history.append(all_losses)

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        return pred, loss, all_losses


class CLPenaltyTrainer(ClassificationTrainer):
    """
        Uses the penalty:

            ( || dy/dx || - || theta ||)^2

    """
    def __init__(self, model, args):
        super().__init__(model, args)
        self.lambd =  args.theta_reg_lambda #regularization strenght
        self.R     = 1     # radius for lipschitz locality
        self.reconst_criterion = nn.MSELoss()

        if self.cuda: self.reconst_criterion = self.reconst_criterion.cuda()

    def train_batch(self, inputs, targets):
        """ inputs, targets already variables """
        # Init
        self.optimizer.zero_grad()

        inputs.requires_grad = True

        # Predict
        pred = self.model(inputs)

        # Calculate loss
        #loss = F.cross_entropy(y_pred, target)
        pred_loss       = self.prediction_criterion(pred, targets)
        torch.autograd.backward(pred_loss, create_graph=True)
        #update1 = model.weight.grad.data.clone()

        grad_penalty = self.calc_crosslip_penalty(self.model.parametrizer, inputs)#, pred)
        grad_penalty.backward() # this will be added to the grads w.r.t. the loss

        #print(pred_loss.data[0], grad_penalty.data[0])

        loss = pred_loss + self.lambd*grad_penalty

        self.loss_history.append([pred_loss.data[0], grad_penalty.data[0]])
        self.optimizer.step()

        return pred, loss

    def calc_crosslip_penalty(self, net, x):
        thetas = net(x)
        nclass = thetas.size(1)
        i = 0
        grad_outputs =  + 1
        #print(grad_outputs)

        # gradients =torch.autograd.grad(outputs=thetas,inputs=x,
        #                                   grad_outputs = torch.ones(thetas.size())  if self.cuda else torch.ones(
        #                                   thetas.size()),
        #                                   create_graph=True)[0].squeeze()


        #print(gradients.size())
        d_thetas = [] #torch.zeros(nclass)
        for i in range(nclass):
            grad_class_i = torch.autograd.grad(outputs=thetas[:,i],inputs=x,
                                              grad_outputs = torch.ones(thetas[:,i].size())  if self.cuda else torch.ones(
                                              thetas[:,i].size()),
                                              create_graph=True, retain_graph = True, only_inputs=True)[0].squeeze()
            d_thetas.append(grad_class_i.view(x.size(0), -1)) # B x all input dim

        total = Variable(torch.zeros(x.size(0)))
        for i in range(nclass):
            for j in range(nclass):
                total += (d_thetas[i] - d_thetas[j]).norm(dim=1).squeeze()**2 # B x 1

        #penalty = total/(nclass**2 * x.size(0))
        penalty = total.mean()/(nclass**2) # Mean over examples in batch.
        return penalty

class GradPenaltyTrainer(ClassificationTrainer):
    """ Gradient Penalty Trainer. Depending on the type, uses different penalty:
             Mode 1. || df/dx - theta ||^2
             Mode 2. || dtheta/dx  || / || dh / dx ||
             Mode 3. || df/dx - dh/dx*theta  || (=  || dth/dx*h  || )
    """
    def __init__(self, model, args, typ):
        super().__init__(model, args)

        self.lambd = args.theta_reg_lambda if ('theta_reg_lambda' in args) else 1e-6 #regularization strenght
        self.reconst_criterion = nn.MSELoss()
        self.penalty_type = typ
        self.norm = 2

        if self.cuda: self.reconst_criterion = self.reconst_criterion.cuda()

    def train_batch(self, inputs, targets):
        """ inputs, targets already variables """
        # Init
        self.optimizer.zero_grad()
        #self.model.zero_grad()

        inputs.requires_grad = True

        # Predict
        pred = self.model(inputs)

        # Calculate loss
        pred_loss       = self.prediction_criterion(pred, targets)
        all_losses = {'prediction': pred_loss.data[0]}
        if self.learning_h:
            h_loss = self.concept_learning_loss(inputs, all_losses)
            loss = pred_loss + h_loss
        else:
            loss = pred_loss

        #torch.autograd.backward(pred_loss, create_graph=True)
        #print(pred.grad.size())
        #update1 = model.weight.grad.data.clone()

        if self.penalty_type == 1:
            
            #raise NotImplementedError('Fix this')
            #  || df/dx - theta ||)^2
            #dTh = self.compute_parametrizer_jacobian(inputs)
            dF = torch.autograd.grad(outputs=pred.mean(),inputs=inputs, create_graph=True)[0]
            pdb.set_trace()
            grad_penalty =  (dTh - self.model.thetas).norm(self.norm) #.pow(2)
        elif self.penalty_type == 2:
            #     || dtheta/dx  || / || dh / dx ||
            dTh = self.compute_parametrizer_jacobian(inputs)
            if self.learning_h:
                dH  = self.compute_conceptizer_jacobian(inputs)
                grad_penalty = dTh.norm(self.norm)/dH.norm(self.norm)
            else:
                # We're working with inputs, dH is identity
                grad_penalty = dTh.norm(self.norm)/inputs.size(0)**(0.5)
        else:
            # (V1)  || dh/dx*theta - df/dx  || =  || dth/dx*h  ||  (V2)
            # For V1:
            dF = compute_jacobian(inputs, pred)#  pred.squeeze())  # Squeeze wwas braeking binary case
            if self.learning_h:
                dH  = self.compute_conceptizer_jacobian(inputs)
                prod = torch.bmm(dH, self.model.thetas)
            else:
                # We're working with inputs, dH is identity
                prod = self.model.thetas
                if self.model.conceptizer.add_bias:
                    # Need to take pad with zero derivatives for constant bias term
                    pad = (0,0,0,1) # Means pad to next to last dim, 0 at beginning, 1 at end
                    dF = F.pad(dF, pad, mode = 'constant', value = 0)
            ## For V2:
            #dTh = self.compute_parametrizer_jacobian(inputs).squeeze()
            # Then?? Need to do for, bmm does not do 4D. Maybe trhough sum approach?
            grad_penalty = (prod - dF).norm(self.norm) #.pow(2)

        all_losses['grad_penalty'] = grad_penalty.data[0]

        #grad_penalty.backward() # this will be added to the grads w.r.t. the loss

        #print(grad_penalty.data[0])
        loss = pred_loss + self.lambd*grad_penalty
        loss.backward()

        #self.loss_history.append([pred_loss.data[0], grad_penalty.data[0]])
        self.optimizer.step()

        return pred, loss, all_losses

    def compute_parametrizer_jacobian(self, x):
        thetas  = self.model.thetas
        nclass  = self.nclasses
        if self.norm == 1:
            JTh = compute_jacobian_sum(x,thetas.squeeze()).unsqueeze(-1)
        elif nclass == 1:
            JTh = compute_jacobian(x, thetas[:,:,0])
        else:
            JTh = []
            for i in range(nclass):
                JTh.append(compute_jacobian(x, thetas[:,:,i]).unsqueeze(-1))
            JTh = torch.cat(JTh, 3)
            assert list(JTh.size()) == [x.size(0), x.view(x.size(0),-1).size(1), thetas.size(-2)]
        return JTh

    def compute_conceptizer_jacobian(self, x):
        h = self.model.concepts
        Jh = compute_jacobian(x, h.squeeze())
        assert list(Jh.size()) == [x.size(0), x.view(x.size(0),-1).size(1), h.size(1)]
        return Jh

    def compute_fullmodel_gradient(self, x, ypred):
        grad = torch.autograd.grad(ypred, x,
                           grad_outputs=ypred.data.new(ypred.shape).fill_(1),
                           create_graph=True)[0]
        return grad


    #
    #
    # def calc_gradient_penalty_1(self, net,x,y):
    #     """
    #         ( || df/dx - theta ||)^2
    #
    #     """
    #     
    #     g = torch.autograd.grad(outputs=y.mean(),inputs=x, create_graph=True)[0]
    #     print(g.size())
    #     print(net.thetas.size())
    #
    #     thetas = net.parametrizer(x)
    #     nclass = thetas.size(-1)
    #     DTh = []
    #     for i in range(nclass):
    #         DTh.append(compute_jacobian(x, thetas[:,:,i]).unsqueeze(-1))
    #
    #     DTh = torch.cat(DTh, 3)
    #
    #
    #     #J = compute_jacobian(inputs, self.model.thetas)
    #     print(DTh.size())
    #
    #     diff = (g - net.thetas).norm().pow(2)
    #     return diff
    #
    # def calc_gradient_penalty_2(self,model,x,y, norm = 1):
    #     """
    #         Uses the penalty.
    #
    #              || dtheta/dx  || / || dh / dx ||
    #     """
    #     
    #     # the variables not the data
    #     thetas = model.thetas #model.parametrizer(x)
    #     # if True:
    #     #     return y.norm()
    #     nclass = thetas.size(-1)
    #     if norm == 1:
    #         DTh = compute_jacobian_sum(x,thetas.squeeze()).unsqueeze(-1)
    #     else:
    #         DTh = []
    #         for i in range(nclass):
    #             DTh.append(compute_jacobian(x, thetas[:,:,i]).unsqueeze(-1))
    #         DTh = torch.cat(DTh, 3)
    #
    #     #h,_ = model.conceptizer(x)
    #     h = model.concepts
    #     Jh = compute_jacobian(x, h.squeeze())
    #     assert list(Jh.size()) == [x.size(0), x.view(x.size(0),-1).size(1), h.size(1)]
    #     ratio = DTh.norm(norm)/Jh.norm(norm)
    #     #DTh.data.zero_()
    #     #Jh.data.zero_()
    #     return ratio#**2
    #
    # def calc_gradient_penalty_3(self, net,x,y, norm =2):
    #     """
    #         Uses the penalty.
    #
    #              || dh/dx*theta - df/dx  || =  || dth/dx*h  ||
    #
    #     """
    #     
    #     # the variables not the data
    #     thetas = model.thetas #model.parametrizer(x)
    #     # if True:
    #     #     return y.norm()
    #     nclass = thetas.size(-1)
    #     if norm == 1:
    #         DTh = compute_jacobian_sum(x,thetas.squeeze()).unsqueeze(-1)
    #     else:
    #         DTh = []
    #         for i in range(nclass):
    #             DTh.append(compute_jacobian(x, thetas[:,:,i]).unsqueeze(-1))
    #         DTh = torch.cat(DTh, 3)
    #
    #     print(DTh.size())
    #
    #
    #     g = torch.autograd.grad(outputs=y.mean(),inputs=x, create_graph=True)[0]
    #     print(g.size())
    #     print(net.thetas.size())
    #
    #     #h,_ = model.conceptizer(x)
    #     h = model.concepts
    #     Jh = compute_jacobian(x, h.squeeze())
    #     assert list(Jh.size()) == [x.size(0), x.view(x.size(0),-1).size(1), h.size(1)]
    #     ratio = DTh.norm(norm)/Jh.norm(norm)
    #     #DTh.data.zero_()
    #     #Jh.data.zero_()
    #     return ratio#**2

### DEPRECATED



class HLearningClassTrainer(ClassificationTrainer):
    """
        Trainer for end-to-end training of conceptizer H
    """
    def __init__(self, model, args):
        super().__init__(model, args)
        self.sparsity = args.h_sparsity
        self.reconst_criterion = nn.MSELoss() 

    def train_batch(self, inputs, targets):
        """ inputs, targets already variables """
        pred = self.model(inputs)

        # Loss
        pred_loss       = self.prediction_criterion(pred, targets)
        reconst_loss    = self.reconst_criterion(self.model.recons,
                            Variable(inputs.data, requires_grad = False))

        loss = pred_loss + reconst_loss

        # Sparsity penalty on encoded H
        if self.sparsity is not None:
            sparsity_loss   = self.model.h_norm_l1.mul(self.sparsity)
            loss += sparsity_loss

        # Keep track of losses
        self.loss_history.append([pred_loss.data[0], reconst_loss.data[0]])

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return pred, loss


class GradPenaltyTrainer_old(ClassificationTrainer):
    """
        Uses the penalty.

            ( || df/dx - theta ||)^2

    """
    def __init__(self, model, args):
        super().__init__(model, args)

        self.lambd = args.lambd if ('lambd' in args) else 1e-6 #regularization strenght
        self.reconst_criterion = nn.MSELoss()

    def train_batch(self, inputs, targets):
        """ inputs, targets already variables """
        # Init
        self.optimizer.zero_grad()

        inputs.requires_grad = True

        # Predict
        pred = self.model(inputs)

        # Calculate loss
        #loss = F.cross_entropy(y_pred, target)
        pred_loss       = self.prediction_criterion(pred, targets)
        torch.autograd.backward(pred_loss, create_graph=True)
        #update1 = model.weight.grad.data.clone()

        grad_penalty = self.calc_gradient_penalty(self.model, inputs, pred)
        grad_penalty.backward() # this will be added to the grads w.r.t. the loss

        loss = pred_loss + self.lambd*grad_penalty

        self.loss_history.append([pred_loss.data[0], grad_penalty.data[0]])
        self.optimizer.step()

        return pred, loss

    def calc_gradient_penalty(self, net,x,y):
        
        g = torch.autograd.grad(outputs=y.mean(),inputs=x, create_graph=True)[0]
        print(g.size())
        print(net.thetas.size())

        thetas = net.parametrizer(x)
        nclass = thetas.size(-1)
        DTh = []
        for i in range(nclass):
            DTh.append(compute_jacobian(x, thetas[:,:,i]).unsqueeze(-1))

        DTh = torch.cat(DTh, 3)


        #J = compute_jacobian(inputs, self.model.thetas)
        print(DTh.size())

        diff = (g - net.thetas).norm().pow(2)
        return diff



class GradPenaltyTrainer3(ClassificationTrainer):
    """
        Uses the penalty.

             || dh/dx*theta - df/dx  || =  || dth/dx*h  ||

    """
    def __init__(self, model, args):
        super().__init__(model, args)

        self.lambd = args.lambd if ('lambd' in args) else 1e-6 #regularization strenght
        self.reconst_criterion = nn.MSELoss()

    def train_batch(self, inputs, targets):
        """ inputs, targets already variables """
        # Init
        self.optimizer.zero_grad()
        #self.model.zero_grad()

        inputs.requires_grad = True

        # Predict
        pred = self.model(inputs)

        # Calculate loss
        #loss = F.cross_entropy(y_pred, target)
        pred_loss       = self.prediction_criterion(pred, targets)
        #torch.autograd.backward(pred_loss, create_graph=True)
        #update1 = model.weight.grad.data.clone()

        grad_penalty = self.calc_gradient_penalty(self.model, inputs, pred, norm = 1)
        #grad_penalty.backward() # this will be added to the grads w.r.t. the loss

        print(grad_penalty.data[0])
        loss = pred_loss + self.lambd*grad_penalty
        loss.backward()

        self.loss_history.append([pred_loss.data[0], grad_penalty.data[0]])
        self.optimizer.step()

        return pred, loss

    def calc_gradient_penalty(self,model,x,y, norm = 1):
        
        # the variables not the data
        thetas = model.thetas #model.parametrizer(x)
        # if True:
        #     return y.norm()
        nclass = thetas.size(-1)
        if norm == 1:
            DTh = compute_jacobian_sum(x,thetas.squeeze()).unsqueeze(-1)
        else:
            DTh = []
            for i in range(nclass):
                DTh.append(compute_jacobian(x, thetas[:,:,i]).unsqueeze(-1))
            DTh = torch.cat(DTh, 3)

        #h,_ = model.conceptizer(x)
        h = model.concepts
        Jh = compute_jacobian(x, h.squeeze())
        assert list(Jh.size()) == [x.size(0), x.view(x.size(0),-1).size(1), h.size(1)]
        ratio = DTh.norm(norm)/Jh.norm(norm)
        #DTh.data.zero_()
        #Jh.data.zero_()
        return ratio#**2




### =========================   DEPRECATED   ================================ ###


#### Legacy
class NormalTrainer_old():
    """ Trainer for supervised digit classification in a framework consisting
        of two parts:
            M - model
    """
    def __init__(self, M, loss_type = 'ce', opt = 'adam', lr = 0.0002, cuda = False, log_interval = 100):
        super(NormalTrainer, self).__init__()
        self.M = M
        self.prediction_criterion = F.nll_loss
        optim_betas = (0.9, 0.999)
        if opt == 'adam':
            self.optimizer = optim.Adam(self.M.parameters(), lr=lr, betas=optim_betas)
        elif opt == 'rmsprop':
            #lrD, lrG = 5e-5, 5e-5 # Original WGAN code has 5e-5
            self.optimizer = optim.RMSprop(self.M.parameters(), lr = lr)
        elif opt == 'sgd':
            self.optimizer = optim.SGD(self.M.parameters(), lr = lr, momentum=0.9)


        self.cuda = cuda
        self.log_interval = log_interval

    def train(self, train_loader, test_loader = None, epochs = 10):
        self.M.train()
        losses = []
        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if self.cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                inputs, targets = Variable(inputs), Variable(targets)

                # Init
                self.optimizer.zero_grad()

                # Predict
                pred = self.M(inputs)

                # Loss
                loss = self.prediction_criterion(pred, targets)
                losses.append(loss.data[0])

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                #loss = self.train_batch(inputs, targets)  ### Maybe abstract away differences in refularized traininf in different train_batch methods??

                if batch_idx % self.log_interval == 1:
                    print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.data[0]),
                        end='')
            print()


            if test_loader:
                self.evaluate(test_loader)
    #
    # def train_batch(self, batch, targets):
    #     self.M_optimizer.zero_grad()
    #     self.C_optimizer.zero_grad()
    #     pred_targets = self.C(self.M(batch))
    #     loss        = self.criterion(pred_targets, targets.view(targets.size(0)))
    #     loss.backward()
    #     self.M_optimizer.step()
    #     self.C_optimizer.step()
    #     return loss

    def evaluate(self, test_loader):
        self.M.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.M(data)
            test_loss += self.prediction_criterion(output, target.view(target.size(0))).data[0]
            #print(output)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            #print(pred, target)
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(test_loader) # loss function already averages over batch size
        print('\nEvaluation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))



class HLearningTrainer():
    """ Trainer for
    """
    def __init__(self, M, loss_type = 'ce', opt = 'adam', lr = 0.0002, cuda = False, log_interval = 100):
        super(HLearningTrainer, self).__init__()
        self.M = M
        #self.prediction_criterion = F.nll_loss
        self.prediction_criterion    = nn.NLLLoss()
        self.reconst_criterion = nn.MSELoss()

        optim_betas = (0.9, 0.999)
        if opt == 'adam':
            self.optimizer = optim.Adam(self.M.parameters(), lr=lr, betas=optim_betas)
        elif opt == 'rmsprop':
            #lrD, lrG = 5e-5, 5e-5 # Original WGAN code has 5e-5
            self.optimizer = optim.RMSprop(self.M.parameters(), lr = lr)
        elif opt == 'sgd':
            self.optimizer = optim.SGD(self.M.parameters(), lr = lr, momentum=0.9)

        self.cuda = cuda
        self.log_interval = log_interval


    def train_batch(self, inputs, targets):
        """ inputs, targets already variables """
        # Loss
        pred_loss       = self.prediction_criterion(pred, targets)
        reconst_loss    = self.reconst_criterion(self.M.recons,
                            Variable(inputs.data, requires_grad = False))
        loss = pred_loss + reconst_loss

        # Keep track of losses
        pred_losses.append(pred_loss.data[0])
        reconst_losses.append(reconst_loss.data[0])

        # Backpropagation
        loss.backward()


    def train(self, train_loader, test_loader = None, epochs = 10):
        pred_losses   = []
        reconst_losses = []
        for epoch in range(epochs):
            self.M.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if self.cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                inputs, targets = Variable(inputs), Variable(targets)

                # Init
                self.optimizer.zero_grad()

                # Predict
                pred = self.M(inputs)

                self.train_batch(inputs, targets)

                self.optimizer.step()

                #loss = self.train_batch(inputs, targets)  ### Maybe abstract away differences in refularized traininf in different train_batch methods??

                if batch_idx % self.log_interval == 1:
                    print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tPred Loss: {:.6f}\tRecons Loss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        pred_loss.data[0],
                        reconst_loss.data[0]),
                        end='')
            print()

            if test_loader:
                self.evaluate(test_loader)
    #
    # def train_batch(self, batch, targets):
    #     self.M_optimizer.zero_grad()
    #     self.C_optimizer.zero_grad()
    #     pred_targets = self.C(self.M(batch))
    #     loss        = self.criterion(pred_targets, targets.view(targets.size(0)))
    #     loss.backward()
    #     self.M_optimizer.step()
    #     self.C_optimizer.step()
    #     return loss

    def evaluate(self, test_loader):
        self.M.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.M(data)
            test_loss += self.prediction_criterion(output, target.view(target.size(0))).data[0]
            #print(output)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            #print(pred, target)
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(test_loader) # loss function already averages over batch size
        print('\nEvaluation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
