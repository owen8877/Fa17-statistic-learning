from __future__ import print_function, division

import torch
import torchvision
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform
from sklearn import preprocessing
import os
import time
from PIL import Image
import gc


class DogsDataset(Dataset):
    """Dogs dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.category_set = pd.read_csv(csv_file)#[0:100]
        self.label_onehot_encoder = preprocessing.LabelBinarizer()
        self.label_encoder = preprocessing.LabelEncoder()
        self.labels = self.category_set['breed']
        self.labels_onehot = self.label_onehot_encoder.fit_transform(self.labels)
        self.labels_int = self.label_encoder.fit_transform(self.labels)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.category_set)

    def category_levels(self):
        return len(self.label_onehot_encoder.classes_)

    def decode(self, label_):
        return self.label_encoder.inverse_transform(label_)

    def __getitem__(self, idx):
        img_name = os.path.join(os.getcwd(), self.root_dir, self.category_set.ix[idx, 0]+'.jpg')
        # io.use_plugin(name='matplotlib', kind='imread')
        # image = io.imread(img_name)
        image = Image.open(img_name)
        image = image.convert('RGB')
        label = self.labels_int[idx]

        if self.transform:
            image = self.transform(image)

        # return image, torch.from_numpy(label)
        return image, label


class DogsTestset(Dataset):
    """Dogs testset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.category_set = pd.read_csv(csv_file)#[0:100]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.category_set)

    def __getitem__(self, idx):
        img_name = os.path.join(os.getcwd(), self.root_dir, self.category_set.ix[idx, 0]+'.jpg')
        image = Image.open(img_name)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        gc.collect()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for data in dogs_loader:
            # get the inputs
            inputs, labels = data

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
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            # running_loss += loss.data[0]
            # running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dogs_dataset_size
        epoch_acc = running_corrects / dogs_dataset_size

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # deep copy the model
        # if epoch_acc > best_acc:
            # best_acc = epoch_acc
            # best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # gc.collect()
    return model


def imshow(inp, title=None):
    """显示Tensor类型的图片"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


# def visualize_model(model, num_images=6):
#     images_so_far = 0
#     fig = plt.figure()
#
#     for i, data in enumerate(dogs_loader[-10:-1]):
#         inputs, labels = data
#         if use_gpu:
#             inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#         else:
#             inputs, labels = Variable(inputs), Variable(labels)
#
#         outputs = model(inputs)
#         _, preds = torch.max(outputs.data, 1)
#
#         for j in range(inputs.size()[0]):
#             images_so_far += 1
#             ax = plt.subplot(num_images//2, 2, images_so_far)
#             ax.axis('off')
#             ax.set_title('predicted: {}'.format([preds[j]]))
#             imshow(inputs.cpu().data[j])
#
#             if images_so_far == num_images:
#                 return

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])

dogs_dataset = DogsDataset('data\\dogs-breed\\labels.csv', 'data\\dogs-breed\\train', data_transforms)
dogs_dataset_size = len(dogs_dataset)
dogs_label_levels = dogs_dataset.category_levels()
dogs_loader = torch.utils.data.DataLoader(dogs_dataset, batch_size=16, shuffle=True, num_workers=2)
use_gpu = torch.cuda.is_available() or True

def train_part(model_conv):
    if use_gpu:
        model_conv = model_conv.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=10)
    torch.save(model_conv.state_dict(), 'test.dict')
    return model_conv

def test_part(model_conv):
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])
    ])
    test_file = list(pd.read_csv('data\\dogs-breed\\test_filename.csv', header=None)[0])
    test_dataset = DogsTestset('data\\dogs-breed\\test_filename.csv', 'data\\dogs-breed\\test', test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=2)

    # gives out predictions
    predictions = torch.zeros(len(test_dataset), dogs_label_levels)
    model_conv.train(False)
    model_conv.eval()

    def prob(output):
        return torch.exp(output.data) / torch.sum(torch.exp(output.data), 1)
    for i, data in enumerate(test_loader):
        # get the inputs
        image = data

        # wrap them in Variable
        if use_gpu:
            image = Variable(image.cuda())
        else:
            image = Variable(image)

        outputs = model_conv(image)
        _, preds = torch.max(outputs.data, 1)
        predictions[i, :] = prob(outputs)
        # print('Predict {} to be {}.\n'.format(test_file[i], dogs_dataset.decode(preds)))

    # print(predictions)
    out = pd.DataFrame(data=predictions.numpy(), index=test_file, columns=dogs_dataset.label_encoder.classes_)
    out.to_csv('result.csv', index_label='id')

def main():
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, dogs_label_levels)

    trainb = True
    if trainb:
        model_conv = train_part(model_conv)
    else:
        model_conv.load_state_dict(torch.load('test.dict'))
        model_conv.cuda()

    testb = False
    if testb:
        test_part(model_conv)


if __name__ == '__main__':
    main()