from __future__ import print_function, division
import torchvision.datasets as dset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import argparse

def data_loader(dataroot, batch_size):
    dataset = dset.CIFAR10(root=dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(299),
                               transforms.CenterCrop(299),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    # dataset = datasets.ImageFolder(root=dataroot,
    #                                transform=transforms.Compose([
    #                                    transforms.Resize(299),
    #                                    transforms.CenterCrop(299),
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True)

    return dataloader


parser = argparse.ArgumentParser()
#  parser.add_argument('--dataset', required=True, help='cifar10 | imagenet | folder | lfw | fake')

parser.add_argument('--dataroot', required = True)
parser.add_argument('--num_epoch', type = int, default=20)
parser.add_argument('--batch_size', type = int, default=64)

config = parser.parse_args()



def train_model(model, criterion, optimizer, scheduler, dataloader, use_gpu, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            count = 0

            # Iterate over data.
            for i, data in enumerate(dataloader, 0):
                count = i
                # get the inputs
                inputs, labels = data
                # print(batch_size)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)


                _, preds = torch.max(outputs[0], 1)
                loss = criterion(outputs[0], labels)


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]

                # print(labels)
                running_corrects += torch.sum(preds == labels)

                print("Epoch: {}, Step: {}, Loss: {}".format(epoch, i, loss.data[0]))

            epoch_loss = running_loss / (count + 1)
            epoch_acc = running_corrects / (count + 1)

            print('{} Loss: {} Acc: {}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



model_ft = models.inception_v3(pretrained=True) #resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)

use_gpu = True
if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       dataloader=data_loader(config.dataroot, config.batch_size), use_gpu=use_gpu, num_epochs=config.num_epoch)
