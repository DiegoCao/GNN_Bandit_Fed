#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import pickle
import os
import json
import collections
from torch.utils.tensorboard import SummaryWriter
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, superuser_noniid
from utils.options import args_parser
from models.Update import LocalUpdate, LocalUpdate_nlp
from models.Nets import MLP, CNNMnist, CNNCifar, NLPModel
from models.Dataset import NLPDataset
from models.Fed import FedAvg
from models.test import test_img, test_nlp
from models.Client import degree, sort_degree
from bandit.main import Bandit

CLIENTSELECTION = "CLUSTER"
NUMBERFILE = "1"
CLIENTNUMBER = 5
CLUSTERFILE = "rev_partition"
TESTROUND = 5

if __name__ == '__main__':
    print(CLIENTSELECTION)
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("the device used for training is : ", args.device)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'superuser':
        dataset_train = {"users": [], "user_data": {}}
        with open("./data/superuser/superuser_trainnew.json", "rb") as file:
            jf = json.load(file)
            dataset_train["users"].extend(jf["users"])
            dataset_train["user_data"].update(jf["user_data"])
        dataset_test = {"users": [], "user_data": {}}
        with open("./data/superuser/superuser_testnew.json", "rb") as file:
            jf = json.load(file)
            dataset_test["users"].extend(jf["users"])
            dataset_test["user_data"].update(jf["user_data"])
        vocab_file = pickle.load(open("./data/vocab/superuser_vocab.pck", "rb"))
        vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
        vocab.update(vocab_file['vocab'])
        dict_users = superuser_noniid(dataset_train['user_data'], vocab)
        test_users = superuser_noniid(dataset_test['user_data'], vocab)
        all_idx = dataset_train['users']
        args.num_users = len(all_idx)
        if CLIENTSELECTION == "BANDIT":
            f = np.load("./data/val.npy")
            new_f = np.zeros((CLIENTNUMBER * args.epochs, args.num_users, 16), dtype=np.float32)
            with open("./data/superuser/rev.json") as file:
                rev = json.load(file)
            for i, u in enumerate(all_idx):
                for j in range(CLIENTNUMBER * args.epochs):
                    new_f[j, i] = f[int(rev[u])]
            f = new_f
    elif args.dataset == 'yelp':
        data_files = [f for f in os.listdir("./data/yelp_leaf/train") if f.endswith('.json')]
        dataset_train = {"users": [], "user_data": {}}
        for f in data_files:
            with open("./data/yelp_leaf/train/" + f, "rb") as file:
                jf = json.load(file)
                dataset_train["users"].extend(jf["users"])
                dataset_train["user_data"].update(jf["user_data"])
        data_files = [f for f in os.listdir("./data/yelp_leaf/test") if f.endswith('.json')]
        dataset_test = {"users": [], "user_data": {}}
        for f in data_files:
            with open("./data/yelp_leaf/test/" + f, "rb") as file:
                jf = json.load(file)
                dataset_test["users"].extend(jf["users"])
                dataset_test["user_data"].update(jf["user_data"])
        # vocab_file = pickle.load(open("./data/vocab/superuser_vocab.pck", "rb"))
        vocab_file = pickle.load(open("./data/vocab/yelp_vocab.pck", "rb"))
        vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
        vocab.update(vocab_file['vocab'])
        dict_users = superuser_noniid(dataset_train['user_data'], vocab)
        test_users = superuser_noniid(dataset_test['user_data'], vocab)
        all_idx = dataset_train['users']
        args.num_users = len(all_idx)
        if CLIENTSELECTION == "BANDIT":
            f = np.load("./data/val.npy")
            new_f = np.zeros((CLIENTNUMBER * args.epochs, args.num_users, 16), dtype=np.float32)
            with open("./data/superuser/rev.json") as file:
                rev = json.load(file)
            for i, u in enumerate(all_idx):
                for j in range(CLIENTNUMBER * args.epochs):
                    new_f[j, i] = f[int(rev[u])]
            f = new_f
    else:
        exit('Error: unrecognized dataset')

    if CLIENTSELECTION == "CLUSTER":
        with open("./data/superuser/"+CLUSTERFILE+".json", "rb") as file:
            rev_d = json.load(file)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        img_size = dataset_train[0][0].shape
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'nlp' and args.dataset == 'superuser':
        net_glob = NLPModel(vocab=vocab)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    test_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    write = SummaryWriter('./path/to/log/' + args.dataset + CLIENTSELECTION + '/' + str(CLIENTNUMBER))
    if CLIENTSELECTION == "BANDIT":
        bandit = Bandit(CLIENTNUMBER * args.epochs, args.num_users, 16, f, False)

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    if CLIENTSELECTION == "CLUSTER":
        cluster_ref = set(rev_d.keys())
    lr = 1e-3
    for iter in range(args.epochs):
        loss_locals = []
        acc_locals = []

        if not args.all_clients:
            w_locals = []
        m = CLIENTNUMBER
        if CLIENTSELECTION == "RANDOM":
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        elif CLIENTSELECTION == "CLUSTER":
            idxs_users, cluster_ref = degree(rev_d, m, cluster_ref)
            if cluster_ref.__len__() < m:
                cluster_ref = set(rev_d.keys())
        elif CLIENTSELECTION == "BANDIT":
            idxs_users = []
            idxs_users.append(bandit.get_arm(iter * CLIENTNUMBER))
        # idxs_users = sort_degree(rev_d, m)
        # ma = max(map(int, rev_d.keys()))
        # rev_d[str(min(map(int, rev_d.keys())) - 1)] = rev_d[str(ma)]
        # del rev_d[str(ma)]

        lr = lr * 0.993
        train_losses = []
        for it, idx in enumerate(idxs_users):
            if CLIENTSELECTION == "RANDOM" or CLIENTSELECTION == "BANDIT":
                id = all_idx[idx]
            else:
                id = idx
            local = LocalUpdate_nlp(args=args, dataset=NLPDataset(dict_users), idxs=id, len=len)
            w, loss, acc = local.train(net=copy.deepcopy(net_glob).to(args.device), lr=lr)
            train_losses.append(loss)
            if CLIENTSELECTION == "BANDIT":
                bandit.set_reward(iter * CLIENTNUMBER + it, loss)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            if CLIENTSELECTION == "BANDIT" and it + 1 < CLIENTNUMBER:
                idxs_users.append(bandit.get_arm(iter * CLIENTNUMBER + it + 1))

        # update global weights
        w_glob = FedAvg(w_locals)
        print('The training loss in round ', (iter + 1), ' is ', np.average(train_losses))

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        if (iter + 1) % TESTROUND == 0:
            net = copy.deepcopy(net_glob).to(args.device)
            for it, idx in enumerate(idxs_users):
                if CLIENTSELECTION == "RANDOM" or CLIENTSELECTION == "BANDIT":
                    id = all_idx[0]
                else:
                    id = idx
                local = LocalUpdate_nlp(args=args, dataset=NLPDataset(dict_users), idxs=id, len=len)
                ppl, loss = local.test(copy.deepcopy(net_glob).to(args.device))
                loss_locals.append(copy.deepcopy(loss))
                acc_locals.append(copy.deepcopy(ppl))
                if CLIENTSELECTION == "BANDIT" and idxs_users.__len__() < CLIENTNUMBER:
                    idxs_users.append(bandit.get_arm(iter * CLIENTNUMBER))
            loss_avg = sum(loss_locals) / loss_locals.__len__()
            acc_avg = sum(acc_locals) / acc_locals.__len__()
            loss_train.append(acc_avg)
            write.add_scalar("Word PPL", acc_avg, iter)
            write.add_scalar("Loss", loss_avg, iter)
            print()
            print('Round ' + str(iter + 1) + 'Test Word perplexity ' + str(acc_avg) + '; Test loss ' + str(loss_avg))
    write.close()

    if CLIENTSELECTION == "RANDOM":
        with open("save/random" + NUMBERFILE + ".json", "w") as file:
            file.write(json.dumps(loss_train))
    elif CLIENTSELECTION == "CLUSTER":
        with open("save/partition" + NUMBERFILE + ".json", "w") as file:
            file.write(json.dumps(loss_train))
    elif CLIENTSELECTION == "BANDIT":
        with open("save/bandit" + NUMBERFILE + ".json", "w") as file:
            file.write(json.dumps(loss_train))
    with open("save/" + CLIENTSELECTION + NUMBERFILE + "test.json", "w") as file:
        file.write(json.dumps(test_train))
