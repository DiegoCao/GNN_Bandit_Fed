#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import collections
import json
import math
import pickle

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, superuser_noniid
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import datasets, transforms

from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar, NLPModel


class DatasetSmall(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset


    def __len__(self):
        return len(self.dataset['x'])

    def __getitem__(self, item):
        xy = self.dataset
        return torch.tensor(xy['x'][item]), torch.tensor(xy['y'][item]), torch.ByteTensor(xy['mask'][item])


def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=transform, target_transform=None,
                                         download=True)
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'superuser':
        with open("./data/superuser/superuser_trainnew.json", "rb") as file:
            dataset_train = json.load(file)
        with open("./data/superuser/superuser_testnew.json", "rb") as file:
            dataset_test = json.load(file)
        with open("./data/superuser/rev_degree.json", "rb") as file:
            rev_d = json.load(file)
        vocab_file = pickle.load(open("./data/superuser/vocab.pck", "rb"))
        vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
        vocab.update(vocab_file['vocab'])
        dict_users = superuser_noniid(dataset_train['user_data'], vocab)
        all_idx = dataset_test['users']
        len = max(dataset_train['num_samples'])
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.model == 'nlp' and args.dataset == 'superuser':
        net_glob = NLPModel(vocab=vocab)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # training
    train_loader = DataLoader(DatasetSmall(dict_users[0]), batch_size=64, shuffle=True)

    net_glob.train()
    # train and update
    optimizer = torch.optim.Adam(net_glob.parameters(), lr=0.001)
    epoch_loss = []
    epoch_correct = []
    loss_func = nn.CrossEntropyLoss()

    for iter in range(args.epochs):
        batch_loss = []
        batch_correct = []
        perplex = []
        state_h, state_c = net_glob.init_state()
        state_h, state_c = state_h.to(args.device), state_c.to(args.device)

        for batch, (x, y, mask) in enumerate(train_loader.dataset):
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = net_glob(x, (state_h, state_c))
            loss = loss_func(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()
            print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t PPL: {:.6f}'.format(
                iter, batch * len(x), len(train_loader.dataset),
                      100. * batch / len(train_loader.dataset), loss.item(), math.exp(loss.item())))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    # plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('./log/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))
