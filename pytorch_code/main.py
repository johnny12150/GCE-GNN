#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import logging
import time
from datetime import datetime as dt
from utils import build_graph, Data, split_validation
from model import *
import os
from graph_loader import MultiSessionsGraph
from torch_geometric.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=128, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-4, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--dynamic', type=bool, default=False, help='using a dataset was created for dynamic graph')
opt = parser.parse_args()
print(opt)

today = dt.today()
log_file = os.getcwd()+'/log/'+opt.dataset+'_%s_%s_%s.log' % (str(today.year), str(today.month), str(today.day))
FORMAT = '%(asctime)s %(levelname)s: %(message)s'
datefmt = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(level=logging.INFO, filename=log_file, format=FORMAT, datefmt=datefmt, filemode='w')


def main():
    cur_dir = os.getcwd()
    train_dataset = MultiSessionsGraph(cur_dir + '/datasets/' + opt.dataset, phrase='train', opt=opt)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
    test_dataset = MultiSessionsGraph(cur_dir + '/datasets/' + opt.dataset, phrase='test', opt=opt)
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

    if opt.dataset == 'diginetica':
        n_node = 43040
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'diginetica_users':
        n_node = 57070
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        logging.info('-------------------------------------------------------')
        logging.info('epoch: %s', epoch)
        hit, mrr = train_test(model, train_loader, test_loader, logging)
        # todo add save model
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        logging.info('Best Result:')
        logging.info('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    logging.info('-------------------------------------------------------')
    logging.info("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
