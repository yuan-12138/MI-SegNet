#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:26:43 2020

@author: yuanbi
"""
# import numpy as np
# import matplotlib.pyplot as plt
from Dataset_loader import Pair_Adv_Dataset_loader, Dataset_loader_test, Dataset_loader_ft, Dataset_loader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
# import torch
# from convVAE import *
# import itertools


BATCH_SIZE = 128
NUM_WORKERS = 2

def prepare_test_data(NUM_DEMO, length, files_img, files_label, demo_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):

	test_set = Dataset_loader_test(list(range(0,length.sum())), length, files_img, files_label, NUM_DEMO, demo_path)
	# num_workers denotes how many subprocesses to use for data loading
	testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	print('testloader size:', len(test_set))   
    
	return testloader
    

def prepare_data_pair_adv(NUM_DEMO, length, files_img, files_label, demo_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):

	training_set = Pair_Adv_Dataset_loader(list(range(0,length.sum())), length, files_img, files_label, NUM_DEMO, demo_path)
	# num_workers denotes how many subprocesses to use for data loading
	trainloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	print('trainloader size:', len(training_set))
    
	return trainloader
    
def prepare_data_fine_tuning(NUM_DEMO, length, files_img, files_label, demo_path, val_per=0.9, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    dataSet = Dataset_loader_test(list(range(0,length.sum())), length, files_img, files_label, NUM_DEMO, demo_path)
    nTrain = int(len(dataSet)*(1-val_per))
    nValid = int(len(dataSet)-nTrain)
    
    trainSet,validSet = random_split(dataSet,[nTrain,nValid])
    train_loader = DataLoader(trainSet,batch_size=batch_size,shuffle=True,num_workers=0)
    valid_loader = DataLoader(validSet,batch_size=batch_size,shuffle=True,num_workers=0)
    print('trainloader size:', len(train_loader), 'testloader size:', len(valid_loader))   
    
    return train_loader, valid_loader
    
def prepare_data_fine_tuning_(NUM_DEMO, length, length_test, files_img, files_label, files_img_test, files_label_test, demo_path, val_per=0.9, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    training_set = Dataset_loader_ft(list(range(0,length.sum())), length, files_img, files_label, NUM_DEMO, demo_path, True)
    test_set = Dataset_loader_ft(list(range(0,length_test.sum())), length_test, files_img_test, files_label_test, NUM_DEMO, demo_path, False)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('trainloader size:', len(train_loader), 'testloader size:', len(valid_loader))   
    
    return train_loader, valid_loader

def prepare_data(NUM_DEMO, length, files_img, files_label, demo_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):

	training_set = Dataset_loader(list(range(0,length.sum())), length, files_img, files_label, NUM_DEMO, demo_path)
	# num_workers denotes how many subprocesses to use for data loading
	trainloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	print('trainloader size:', len(training_set))
    
	return trainloader
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
