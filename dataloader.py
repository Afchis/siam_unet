import os
import glob
import numpy as np

from PIL import Image
import scipy.ndimage.morphology as morph

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from args import *


def transform(target, search, mask):
	transform = transforms.Compose([
                              transforms.Resize((SEARCH_SIZE, SEARCH_SIZE), interpolation = 0),
                              transforms.ToTensor()
                              ])
	transtarg = transforms.Compose([
                              transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation = 0),
                              transforms.ToTensor()
                              ])
	return transtarg(target), transform(search), transform(mask)


class test_data(Dataset):
	def __init__(self):
		super().__init__()
		self.to_tensor = transforms.ToTensor()

	def list_dir(self, obj):
		return sorted(os.listdir("../siam_unet/data_per/" + obj))

	def transform(self, target, search, mask):
		transform = transforms.Compose([
			transforms.Resize((SEARCH_SIZE, SEARCH_SIZE), interpolation = 0),
			transforms.ToTensor()
			])
		transtarg = transforms.Compose([
			transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation = 0),
			transforms.ToTensor()
			])
		return transtarg(target), transform(search), transform(mask)

	def transform_score_label(self, depth2):
		depth2 = depth2.reshape(1, 1, depth2.size(0), depth2.size(1))
		max_value = depth2.max()
		depth2 = (depth2 == max_value).float()
		score_label = F.max_pool2d(depth2, kernel_size=(16, 16), padding=8, stride=16)
		score_zero = (score_label == 0).float()
		score_label = torch.stack([score_zero, score_label], dim=1).squeeze()
		return score_label

	def get_labels(self, object):
		label1 = (object==0).float()
		depth1 = torch.tensor(morph.distance_transform_edt(np.asarray(label1[0])))
		label2 = (label1==0).float()
		depth2 = torch.tensor(morph.distance_transform_edt(np.asarray(label2[0])))
		depths = depth1 + depth2
		labels = torch.stack([label1, label2], dim=1).squeeze()

		score_label = self.transform_score_label(depth2)
		return labels, depths, score_label

	def __getitem__(self, idx):
		#####
		try:
			names = self.list_dir("imgs/0")
			target = Image.open("../siam_unet/data_per/0.jpg").convert('RGB')
			search = Image.open("../siam_unet/data_per/imgs/0/" + names[idx]).convert('RGB')
			mask = Image.open("../siam_unet/data_per/anno/0/" + names[idx][:5] + '.png').convert('L')
		except IndexError :
			names = self.list_dir("imgs/1")
			target = Image.open("../siam_unet/data_per/1.jpg").convert('RGB')
			search = Image.open("../siam_unet/data_per/imgs/1/" + names[idx-71]).convert('RGB')
			mask = Image.open("../siam_unet/data_per/anno/1/" + names[idx-71][:5] + '.png').convert('L')
		#####
		target, search, mask = self.transform(target, search, mask)
		label, depth, score_label = self.get_labels(mask)
		return target, search, label, depth, score_label

	def __len__(self):
		return len(self.list_dir("imgs/0"))+len(self.list_dir("imgs/1"))


train_dataset = test_data()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False)


if __name__ == '__main__':
	dataset = test_data()
	target, search, label, depth, score_label = dataset[0]
	print(score_label, score_label.shape)
	print(label.shape)

