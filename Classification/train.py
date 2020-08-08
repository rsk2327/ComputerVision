import torch
import torchvision

import os
import pandas as pd 
import numpy as np 
import sys
import logging
import time
import copy
import logging
import warnings
warnings.filterwarnings('ignore')



from tqdm import tqdm
from torch.nn import Sigmoid
from torch import Tensor, LongTensor
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.detection.image_list import ImageList 
from torchvision.transforms import Resize,ToTensor, RandomHorizontalFlip, RandomVerticalFlip,Normalize
from torchvision import models
from torch.optim import lr_scheduler

from progressbar import progressbar

from PIL import Image

from utils import createBMIDataset, generateDataList, RaceDataset, freezeLayers, unfreezeLayers, trainModel, saveModel


#### USER INPUTS ######

dataFolder = '/cbica/home/santhosr/RaceDL_BACKUP/Data'
truthFile = '/cbica/home/santhosr/RaceDL_BACKUP/Modeling/TargetFiles/TargetFile_Combined.csv'
modelFolder = '/cbica/home/santhosr/RaceDL_BACKUP/Modeling/PyTorch/Models'
splitfileFolder = '/cbica/home/santhosr/RaceDL_BACKUP/Modeling/PyTorch/SplitFiles'


logFile = '/cbica/home/santhosr/RaceDL_BACKUP/Modeling/PyTorch/log.txt'

numPatients = 2000
imgSize = 256

modelNo = 1
train_batch_size = 12
valid_batch_size = 12

savedModel = None

#######################



logging.basicConfig(filename=logFile,level=logging.DEBUG, format='%(message)s', filemode='w')
logger = logging.getLogger()

logger.info(time.asctime())

logger.info(f"numPatients : {numPatients}")
logger.info(f"imgSize : {imgSize}")

## Looking for existing split file

if f"SplitFile_{modelNo}.csv" in os.listdir(splitfileFolder):
	outDf = pd.read_csv(os.path.join(splitfileFolder, f"SplitFile_{modelNo}.csv" ))
else:
	finalDf = generateDataList(dataFolder, truthFile)
	outDf = createBMIDataset(finalDf, numPatients)

	outDf.to_csv(os.path.join(splitfileFolder, f"SplitFile_{modelNo}.csv" ), index = False, index_label  = False)

logger.info("Dataset created")


trainTransforms = transforms.Compose([Resize( (imgSize, imgSize) ),
						   ToTensor()])
validTransforms = transforms.Compose([Resize( (imgSize, imgSize) ),
						   ToTensor()])


trainDf = outDf[outDf.train ==False]
validDf = outDf[outDf.train ==True]

trainDataset = RaceDataset(dataFolder, trainDf,trainTransforms)
validDataset = RaceDataset(dataFolder, trainDf,validTransforms)


device='cuda'




#Defining dataloaders
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=train_batch_size, 
										  shuffle=True)
validLoader = torch.utils.data.DataLoader(validDataset, batch_size=valid_batch_size, 
										  shuffle=False)


image_datasets = {'train':trainDataset, 'val':validDataset}

dataloaders = {"train":trainLoader, "val":validLoader}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}



#Define the loss functions
criteria = nn.CrossEntropyLoss()


#Defining model architecture. Transfer Learning using Resnet34
net = models.resnet34(pretrained=True)
net.fc = nn.Linear(512,2)
net.to(device)


if savedModel is not None:
	net.load_state_dict(torch.load(os.path.join(modelFolder, savedModel)))
	logger.info(f"Loaded {savedModel}")


logger.info("Model defined")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 1e-3)
#Defining Learning Rate scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)


print("Reached here")


#### TRAINING PROCESS

net = freezeLayers(net)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 1e-3)



net, val_acc = trainModel(model = net, traindl = trainLoader, criterion = criterion, optimizer = optimizer, 
	scheduler = exp_lr_scheduler, logger = logger, num_epochs = 5, validdl = validLoader,
	test_valsubset = False)


saveModel(net, modelFolder, modelNo, 5, val_acc)


