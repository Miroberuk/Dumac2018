import augment
import copy
import torch
import torchvision
import matplotlib.pyplot as plt
import models.resnet as resnet
import models.alexnet as alexnet
import models.vgg as vgg
import numpy as np

import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.trainer as trainer
import torch.utils.trainer.plugins
import torch.utils.data.sampler as sampler
import torchvision.datasets as datasets
import torchvision.models as models

## Metric Imports
import sklearn
from sklearn.metrics import precision_recall_fscore_support

import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


## Test the Model
#model           = torch.nn.DataParallel(resnet.resnet18(pretrained=True)).cuda()
#model           = torch.nn.DataParallel(resnet.resnet34(pretrained=True)).cuda()
model           = torch.nn.DataParallel(resnet.resnet50(pretrained=True)).cuda()
#model           = torch.nn.DataParallel(resnet.resnet101(pretrained=True)).cuda()
#model           = torch.nn.DataParallel(resnet.resnet152(pretrained=True)).cuda()
#model            = torch.nn.DataParallel(alexnet.alexnet(pretrained=True)).cuda()
#model            = torch.nn.DataParallel(vgg.vgg16_bn(pretrained=True)).cuda()

# Load a pre-trained model, located in a directory labelled loadModels
chkpt = torch.load('loadModels/Resnet50_Augments_OldDataset.pth.tar')
#model_ft = model.load_state_dict(chkpt)
model.load_state_dict(chkpt['params'])
epoch = chkpt['epoch']
criterion = nn.CrossEntropyLoss()

normalize = augment.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                              std  = [ 0.229, 0.224, 0.225 ])

model.eval()
images_so_far = 0
fig = plt.figure()
test_image_num = 0

ground_truth = np.array([])
predictions = np.array([])

# Load the test dataset
test_dataset = datasets.ImageFolder('test/testSet01/', transform=augment.Compose([
    augment.AnisotropicScale((224,224)),
    #augment.ScalePad((224,224)),
    augment.ToTensor(),
    normalize,]))

# Class Names as identified in the test dataset folder
class_names = test_dataset.classes

test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

dataset_size = len(test_loader)

running_corrects = 0
running_loss = 0
for i, data in enumerate(test_loader):
    inputs, labels = data
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        #inputs, target = inputs.cuda(), labels.cuda()
        #inputs, target = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    outputs = model(inputs)
    #_, preds = torch.max(outputs.data, 1)
    prob, preds = torch.max(outputs.data, 1)
    loss = criterion(outputs, labels)

    ##
    # statistics
    running_loss += loss.data[0] * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

    running_loss += loss.data[0]
    running_corrects += torch.sum(preds == labels.data)
    ##

    for j in range(outputs.size()[0]):
        ground_truth = np.append(ground_truth, labels.data[j])
        predictions = np.append(predictions, preds[j])
        test_image_num += 1

    # for j in range(inputs.size()[0]):
    #     images_so_far += 1
    #     ax = plt.subplot(num_images//3, 3, images_so_far)
    #     ax.axis('off')
    #     ax.set_title('{}'.format(class_names[preds[j]]))
    #     #print('test image{} {}'.format(j,class_names[preds[j]]))
    #     imshow(inputs.cpu().data[j])
    #
    #     if images_so_far == num_images:
            # return

#print(ground_truth)
#print(predictions)

test_loss = running_loss / dataset_size
test_acc = running_corrects / dataset_size

#print("Loss {}".format(running_loss))
#print("Correct {}".format(running_corrects))
print("Test Loss {}".format(test_loss))
print("Test Accuracy {}".format(test_acc))

precisionTestSet = sklearn.metrics.precision_score(ground_truth, predictions, average = 'weighted')
recallTestSet = sklearn.metrics.recall_score(ground_truth, predictions, average = 'weighted')
f1TestSet = sklearn.metrics.f1_score(ground_truth, predictions, average = 'weighted')

print('Precision = {}, Recall = {}, F1 = {}'.format(precisionTestSet, recallTestSet, f1TestSet))

## Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(ground_truth, predictions)
np.set_printoptions(precision=2)

class_names2 = ['AFA','AFZ','AV','EFA','EFZ','EV']

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names2,
                      title='Original Dataset')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
