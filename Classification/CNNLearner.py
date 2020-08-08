import torch
import torchvision


import pandas as pd
import numpy as np
import os

import time
import torch
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms

import copy
from PIL import Image

import datetime

from tqdm import tqdm

import logging





class CNNLearner():


    def __init__(self,model, trainLoader, optimizer, criterion, scheduler=None, validLoader=None, device = 'cuda',
                 logger = None, modelFolder = "./",
                 modelname_prefix = ""):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.trainLoader = trainLoader
        self.validLoader = validLoader

        self.device = 'cuda'

        self.modelFolder = modelFolder
        self.modelname_prefix = modelname_prefix

        self.model = self.model.to(self.device)


        
        

        if logger is None:
            logging.basicConfig(filename='./log.txt',level=logging.DEBUG, format='%(message)s', filemode='w')
            self.logger = logging.getLogger()
        else:
            self.logger = logger



        self.epoch = 0


    def updateOptimizerLR(self, lr):

        numGroups = len(self.optimizer.param_groups)

        current_state = self.optimizer.state_dict()

        if type(lr) == list:
            # Setting different LR for layer groups
            a = 1

            if len(lr) != numGroups:
                raise ValueError("Number of LRs provided doesnt match number of param groups in optimizer")

            for i in range(len(lr)):
                current_state['param_groups'][i]['lr'] = lr[i]
                
        else:
            current_state['param_groups'][i]['lr'] = lr

        self.optimizer.load_state_dict(current_state)

    
    def fit(self, num_epochs, lr, save_best_model = True, save_every_epoch = False):

        # self.scheduler = torch.optim.lr_scheduler

        if save_every_epoch:
            save_best_model = False

        best_metric = 0.0
        best_epoch = 0
        best_model = None

        for epoch in range(num_epochs):
            self.epoch += 1

            epoch_start_time = time.time()

            self.model.train()

            self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            self.logger.info('-' * 10)
            
            self.model.train()
            
            running_loss = 0.0
            running_corrects = 0
            
            
            for data,labels in tqdm(self.trainLoader, miniters = int(len(self.trainLoader)/50), mininterval = 30 ):
                data,labels = data.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(data)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
            
            
            
            epoch_loss = running_loss/len(self.trainLoader)
            epoch_acc  = running_corrects.double()/len(self.trainLoader.dataset)
            
            self.logger.info(f"Epoch Time : {str(datetime.timedelta(seconds = time.time() - epoch_start_time))}")

            self.logger.info(f"Train Loss : {epoch_loss}  Train Acc : {epoch_acc}")

            val_epoch_acc = self.evalModel(self.validLoader)

            if val_epoch_acc > best_metric:
                best_metric = val_epoch_acc
                best_model = copy.deepcopy(self.model.state_dict())
                best_epoch = self.epoch + epoch

            if save_every_epoch:
                self.saveModel(epoch + self.epoch, val_epoch_acc)

            

        if save_best_model:
            self.model.load_state_dict(best_model)
            self.saveModel(best_epoch, best_metric)





    def fit_one_cycle(self, num_epochs, lr,  save_best_model = True, save_every_epoch = False):

        if save_every_epoch:
            save_best_model = False

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr, steps_per_epoch=len(self.trainLoader), epochs=num_epochs)

        best_metric = 0.0
        best_epoch = 0
        best_model = None

        for epoch in range(num_epochs):

            self.epoch += 1

            epoch_start_time = time.time()

            self.model.train()

            self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            self.logger.info('-' * 10)
            
            self.model.train()
            
            running_loss = 0.0
            running_corrects = 0
            
            
            for data,labels in tqdm(self.trainLoader, miniters = int(len(self.trainLoader)/50), mininterval = 30 ):
                data,labels = data.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(data)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
            
            
            
            epoch_loss = running_loss/len(self.trainLoader)
            epoch_acc  = running_corrects.double()/len(self.trainLoader.dataset)
            
            self.logger.info(f"Epoch Time : {str(datetime.timedelta(seconds = time.time() - epoch_start_time))}")

            self.logger.info(f"Train Loss : {epoch_loss}  Train Acc : {epoch_acc}")
            print(f"Train Loss : {epoch_loss}  Train Acc : {epoch_acc}")

            val_epoch_acc = self.evalModel(self.validLoader)

            if val_epoch_acc > best_metric:
                best_metric = val_epoch_acc
                best_model = copy.deepcopy(self.model.state_dict())
                best_epoch = self.epoch + epoch

            if save_every_epoch:
                self.saveModel(epoch + self.epoch, val_epoch_acc)

            

        if save_best_model:
            self.model.load_state_dict(best_model)
            self.saveModel(best_epoch, best_metric)


            

    def saveModel(self, epoch, acc):

        acc = np.round(acc*100)

        filename = f"{self.modelname_prefix}_{epoch}_{acc}.pt"

        path = os.path.join(self.modelFolder, filename)
        torch.save(self.model, path)




    def evalModel(self, validLoader, test_val_subset = False, val_subset_batches = 100):

        self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        val_counter = 0

        with torch.no_grad():

            for data,labels in validLoader:
                data,labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

                val_counter += 1
                if val_counter == val_subset_batches and test_val_subset == True:
                    break
			
        val_epoch_loss = running_loss/len(validLoader)

        if test_val_subset == True:
            val_epoch_acc  = running_corrects.double()/(val_counter*validLoader.batch_size)
        else:
            val_epoch_acc  = running_corrects.double()/(len(validLoader.dataset))

        val_epoch_acc = val_epoch_acc.cpu().numpy()

        self.logger.info(f"Valid Loss : {val_epoch_loss}  Valid Acc : {val_epoch_acc}")
        print(f"Valid Loss : {val_epoch_loss}  Valid Acc : {val_epoch_acc}")

        return val_epoch_acc

		

    def fit(self,num_epochs, lr):

        self.updateOptimizerLR(lr)



    def freezeHead(self):

        ## Freeze all layers
        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = False

        ## Unfreezing the last FC layer        
        for param in list(self.model.children())[-1].parameters():
            param.requires_grad = True


    def unfreezeHead(self):

        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = True




    
			
	

            

