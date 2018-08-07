#!/usr/bin/env python
'''
    File name: main.py
    Authored by: Matthew Roberts

    Date created: July 13, 2017
    Date last modified: May 22, 2018
    Python Version: 3.6.1
    Pytorch Version: 0.4.0
    Torchvision Version: 0.2.1-py36-1
    numpy Version: 1.12.1

    This module uses Transfer Learning and a pre-trained model (Resnet or Alexnet) to
    classify and categorize a validation and training set based on a select set
    of class names.

    The training data should be located in the datasets/train/ directory, and the
    validation data in datasets/val/
    Image files in these locations should then be collected into subfolders named
    for the classes to be trained for. For this experiment, our classes were:
        -EgyptAnthropomorphicFigurines
        -EgyptZoomorphicFigurines
        -EgyptVessels
        -AsiaAnthropomorphicFigurines
        -AsiaZoomorphicFigurines
        -AsiaVessels

    Written using tutorials and examples from:
        PyTorch Tutorial to Deep Learning coding:
        http://pytorch.org/tutorials/beginner/pytorch_with_examples.html

        PyTorch Transfer Learning with examples and coding instructions:
        http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''

## Model and Data Setup

## Imports

import os
import torch
import shutil
import random
import numpy as np
from PIL import Image
from os.path import basename

import augment
import copy
import imp
import scipy.misc
import time
import transforms
import visdom
import util

## Metric Imports
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import itertools
#import matplotlib.pyplot as plt

## Models
import models.resnet as resnet
import models.alexnet as alexnet
import models.vgg as vgg

import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.trainer as trainer
import torch.utils.trainer.plugins
import torch.utils.data.sampler as sampler
import torchvision.datasets as datasets
import torchvision.models as models

## Setup visdom and pretrained model arch
# The 'run' variable is the name used to checkpoint and store a trained model.
# The notes store details for Visdom which can be retrieved later, and should
# contain details pertinent to a particular training execution. Relevant
# information includes:
#   - Batch Size
#   - Optimization Parameters (Learning Rate, Betas, Weight Decay)
#   - Whether the model has been pre-trained or not
#vis             = visdom.Visdom()
vis             = visdom.Visdom(server='http://ncc.clients.dur.ac.uk', port=8036)
run             = 'May30_Res50'
#notes           = run+': pretrained, batch size 32, frozen layer 2, lr=0.00001, betas=(0.5,0.999), weight_decay=0.00001'
notes           = run+': Resnet50, batch size 32, frozen layer 2, lr=0.00001, weight_decay=0.00001, Random Augments'

# The Deep Learning model to use, located in the 'models' folder.
# Additional models can be downloaded at https://github.com/pytorch/vision/tree/master/torchvision/models/

#model           = torch.nn.DataParallel(resnet.resnet18(pretrained=True)).cuda()
#model           = torch.nn.DataParallel(resnet.resnet34(pretrained=True)).cuda()
model           = torch.nn.DataParallel(resnet.resnet50(pretrained=True)).cuda()
#model           = torch.nn.DataParallel(resnet.resnet101(pretrained=True)).cuda()
#model           = torch.nn.DataParallel(resnet.resnet152(pretrained=True)).cuda()
#model            = torch.nn.DataParallel(alexnet.alexnet(pretrained=True)).cuda()
#model            = torch.nn.DataParallel(vgg.vgg16_bn(pretrained=True)).cuda()


# Observe that all parameters are being optimized
# We chose to use the Adam algorithm as our gradient-descent optimizer, as it is
# currently considered best-in-class
# Relevant Paper: https://arxiv.org/pdf/1412.6980.pdf
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.5, 0.999), weight_decay=0.00001)
optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)

# Currently using this boolean to choose to run cuda functionality.
# Later versions of pytorch should be able to use:
# use_gpu = torch.cuda.is_avilable()
cudnn.benchmark = True
use_gpu = True


## Data loading code
# normalization to be applied to the training and validation tensors.
normalize = augment.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                              std  = [ 0.229, 0.224, 0.225 ])

# Dataset setup
# Choose what sort of data transformations and augmentations you wish to apply to
# the training.

# A set of random transformations which can be applied.
random_Transform_List = [transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
transforms.ColorJitter(brightness=0.5),transforms.Grayscale(num_output_channels=3)]

data_transforms = {
   'train': augment.Compose([
       #augment.ScalePad((1024,1024)),
       #augment.RandomRotate(60),
       #augment.RandomHorizontalFlip(),
       augment.AnisotropicScale((224,224)),
       transforms.RandomApply(random_Transform_List,p=0.5),
       augment.ToTensor(),
       normalize
   ]),
   'val': augment.Compose([
       #augment.ScalePad((1024,1024)),
       augment.AnisotropicScale((224,224)),
       augment.ToTensor(),
       normalize
   ]),
}

data_dir = 'datasets'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

weights = np.zeros(len(class_names))
for i in range(dataset_sizes['train']):
    c = image_datasets['train'].imgs[i][1]
    weights[c] += 1

weights = 1/weights

samples_weight = np.array([weights[t[1]] for t in image_datasets['train'].imgs])
samples_weight = torch.from_numpy(samples_weight)
balanced_sampler = sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

dataloaders = {
   'train': torch.utils.data.DataLoader(image_datasets['train'],
       sampler=balanced_sampler,
       batch_size=32, num_workers=2, pin_memory=True),
   'val': torch.utils.data.DataLoader(image_datasets['val'],
       batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
}

# Display VISDOM notes
vis.text(notes)

######################################################################
# Visdom function to draw an image from the tensor data
# ``tensor_input`` is a tensor of images
# ``name`` is the visdom window name for the image
def draw_tensor(tensor_input, name):
    tmp = tensor_input.data
    for t, m, s in zip(tmp, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
            vis.image(tmp.cpu().numpy(), win=name)

######################################################################
# Training the model
# ------------------
# Based on the sample code provided in the pytorch transfer learning tutorial:
# https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
#
# ``model`` is our chosen model.
# ``optimizer`` is a gradient-descent optimizer
# ``num_epochs`` is the number of iterations we have selected for our training
# For our training, we chose to use Adam (see below) and to not decay the learning rate.
#
#

def train_model(model, optimizer, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        #Each epoch has a training and validation Phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for data in dataloaders[phase]:
                inputs, target = data

                # wrap them in Variable
                if use_gpu:
                    inputs, target = inputs.cuda(), target.cuda()
                    inputs, target = torch.autograd.Variable(inputs), torch.autograd.Variable(target)
                else:
                    inputs, target = Variable(inputs), Variable(target)

                # zero the parameter gradients
                optimizer.zero_grad()

                #forward
                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(outputs, dim=1), target)

                #backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum((preds == target.data).float())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'train':
                # Visdom display of the first image from the current batch training
                draw_tensor(inputs[0], 'img')

            # For each training phase (train and val) print the loss and accuracy values
            # for the epoch iteration
            if phase == 'train':
                with open(run + '_train.txt', 'a') as f:
                    print('{}   Epoch {},    Loss: {:.4f},    Acc: {:.4f}'.format(phase, epoch, epoch_loss, epoch_acc * 100), file=f)  # Python 3.x
            if phase == 'val':
                print('{}   Epoch {},    Loss: {:.4f},    Acc: {:.4f}'.format(phase, epoch, epoch_loss, epoch_acc * 100))
                with open(run + '_val.txt', 'a') as f:
                    print('{}   Epoch {},    Loss: {:.4f},    Acc: {:.4f}'.format(phase, epoch, epoch_loss, epoch_acc * 100), file=f)  # Python 3.x

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                            'epoch' : epoch,
                            'run' : run,
                            'notes' : notes,
                            'params' : model.state_dict(),
                            'best_acc' : best_acc,
                            }, 'chkpts/'+run+'.pth.tar')

    time_elapsed = time.time() -since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {:4f}'.format(best_acc * 100))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model
##



######################################################################
# Train and evaluate the model

model_ft = train_model(model, optimizer_ft, num_epochs=100)

#######################################################################
