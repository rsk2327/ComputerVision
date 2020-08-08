import torch
import torchvision

import os
import pandas as pd 
import numpy as np 
import sys
import logging
import time
import copy
import warnings
warnings.filterwarnings('ignore')

from progressbar import progressbar
import datetime

from tqdm import tqdm
from torch.nn import Sigmoid
from torch import Tensor, LongTensor
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.detection.image_list import ImageList 


from PIL import Image

device = 'cuda'

def trainModel(model, traindl, criterion, optimizer, scheduler, logger, num_epochs=25, validdl = None, test_valsubset = False):
	
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	
	for epoch in range(num_epochs):

		epoch_start_time = time.time()

		
		logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
		logging.info('-' * 10)
		
		model.train()
		
		running_loss = 0.0
		running_corrects = 0
		
		
		for data,labels in tqdm(traindl, miniters = int(len(traindl)/50), mininterval = 30 ):
			data,labels = data.to(device), labels.to(device)
			
			optimizer.zero_grad()
			
			outputs = model(data)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)
			
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
			running_corrects += torch.sum(preds == labels.data)
			
		scheduler.step()
		
		epoch_loss = running_loss/len(traindl)
		epoch_acc  = running_corrects.double()/len(traindl.dataset)

		logging.info(f"Train Loss : {epoch_loss}  Train Acc : {epoch_acc}")
		
		
		### Validation Testing
		running_loss = 0.0
		running_corrects = 0
		val_counter = 0

		model.eval()
		

		for data,labels in validdl:
			data,labels = data.to(device), labels.to(device)
			outputs = model(data)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)
			
			running_loss += loss.item()
			running_corrects += torch.sum(preds == labels.data)

			val_counter += 1
			if val_counter == 60 and test_valsubset == True:
				break
			
		val_epoch_loss = running_loss/len(validdl)
		if test_valsubset == True:
			val_epoch_acc  = running_corrects.double()/(val_counter*validdl.batch_size)
		else:
			val_epoch_acc  = running_corrects.double()/(len(validdl.dataset))
		
		logging.info(f"Valid Loss : {val_epoch_loss}  Valid Acc : {val_epoch_acc}")

		epoch_end_time = time.time()
		epoch_time = datetime.timedelta(seconds = epoch_end_time - epoch_start_time)
		logger.info(f"Epoch Time : {str(epoch_time)}")


	return model, val_epoch_acc.cpu().numpy()
		
		

def saveModel(net, folder, modelNo, epoch, acc):

	acc = np.round(acc*100)

	filename = f"{modelNo}_{epoch}_{acc}.pt"

	path = os.path.join(folder, filename)
	torch.save(net, path)
		
			



class RaceDataset(torch.utils.data.Dataset):
	def __init__(self, dataFolder, dataList, transforms = None):


		self.data = dataList
		self.dataFolder = dataFolder
		self.transforms = transforms



	def __len__(self):

		return len(self.data)

	def __getitem__(self, index):

		filename = self.data.iloc[index].filename


		img = Image.open(os.path.join(self.dataFolder, filename)).convert('RGB')


		if self.transforms:
			img = self.transforms(img)


		label = self.data.iloc[index].label


		return img,label



def generateDataList(dataFolder, truthFile, includeWhiteDuplicates = True):
	
	fileList = os.listdir(dataFolder)
	df = pd.DataFrame(fileList)
	df.columns = ['filename']
	df['DummyID'] = df.filename.apply(lambda x : int(x.split("_")[0]))

	truthDf = pd.read_csv(truthFile)

	df = df.merge(truthDf[['DummyID','Medview_Race','BMI']], on='DummyID')
	df = df[df.Medview_Race.isin(['African American','White'])]

	df = df[pd.isna(df.BMI)==False]


	dummyCount = df.DummyID.value_counts().reset_index()
	dummyCount.columns = ['DummyID','counts']

	#Selecting all women with 4 images
	finalDf = df[df.DummyID.isin(dummyCount[dummyCount.counts==4]['DummyID'])]

	#Select the white women with duplicate images
	if includeWhiteDuplicates ==True:
		tempDf = df[df.DummyID.isin(dummyCount[dummyCount.counts==8]['DummyID'])]
		tempDf['drop'] = tempDf.filename.apply(lambda x : 'FOR' in x)  #Drop with filenames with FOR PROCESSING in it
		whiteDf=tempDf[tempDf['drop']==False]
		whiteDf.drop('drop',inplace = True, axis = 1)
		finalDf = pd.concat([finalDf,whiteDf], axis =0)
		
		
	return finalDf



def createBMIDataset(df, numPatients, bmi_buckets = [0,20,30,40,50,55,60,100]):

	#Restricting patient base
	
	patientList = df.DummyID.unique()
	df = df[df.DummyID.isin(patientList[:numPatients])]

	
	#Creates equal number of patients from White and AA groups
	white = df[df.Medview_Race=='White']
	AA = df[df.Medview_Race=='African American']


	outputDf = pd.DataFrame()

	for i in range(len(bmi_buckets)-1):
		out = getBMIData(AA,white, bmi_buckets[i], bmi_buckets[i+1])

		outputDf = pd.concat([outputDf, out])
		
	df = df.merge(outputDf, on='DummyID')

	#Shuffling data
	index = list(range(len(df)))
	np.random.shuffle(index)
	df = df.iloc[index]

	raceMap = {'African American':0,'White':1}
	df['label'] = df.Medview_Race.apply(lambda x : raceMap[x])
	
	return df




def getBMIData(aa, white, bmi_min, bmi_max):
	
	aaSub = aa[(aa.BMI>=bmi_min) & (aa.BMI<bmi_max)]
	whiteSub = white[(white.BMI>=bmi_min) & (white.BMI<bmi_max)]

	maxSamples = min(len(whiteSub), len(aaSub))
	
	if maxSamples==0:
		return pd.DataFrame()
	
	
	#Subsetting to keep only maxSamples number of samples in each subset
	aaSub = aaSub.sample(n=maxSamples, replace = False)
	whiteSub = whiteSub.sample(n=maxSamples, replace = False)
	
	#Figuring out the number of samples in train/valid
	numTrain = int(0.8*maxSamples)
	numValid = maxSamples - numTrain
	
	
	
	aaTrain = np.random.choice(aaSub.DummyID, numTrain,replace = False)
	whiteTrain = np.random.choice(whiteSub.DummyID, numTrain,replace = False)
	
	aaValid = np.array(list(set(aaSub.DummyID.values).difference(set(aaTrain))))
	whiteValid = np.array(list(set(whiteSub.DummyID.values).difference(set(whiteTrain))))
	
	d =  pd.concat([
		pd.DataFrame({'DummyID':aaTrain,'train':False}),
		pd.DataFrame({'DummyID':whiteTrain,'train':False}),
		pd.DataFrame({'DummyID':aaValid,'train':True}),
		pd.DataFrame({'DummyID':whiteValid,'train':True})
	])
	
	print("Max Samples : {} numTrain  : {} df len : {}".format(maxSamples, numTrain, len(d)))
	
	return d




def freezeLayers(net):
	
	## Freeze all layers
	for child in net.children():
		for param in child.parameters():
			param.requires_grad = False

	## Unfreezing the last FC layer        
	for param in list(net.children())[-1].parameters():
		param.requires_grad = True
		
	return net
	
	
def unfreezeLayers(net):
	
	## Freeze all layers
	for child in net.children():
		for param in child.parameters():
			param.requires_grad = True
			
	return net