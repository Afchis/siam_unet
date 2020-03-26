import torch
import torch.nn as nn
import torch.nn.functional as F

from args import *
from model_unet_head import *
from model_unet_paths import *


# SearchModel
class BackboneUNet(nn.Module):
	def __init__(self, input_search):
		super(BackboneUNet, self).__init__()
		self.model = UNetDesigner(input_search)
		self.model.load_state_dict(torch.load('weights/weights.pth'))
		self.adjust = nn.Conv2d(1024, 256, kernel_size=1) 

	def forward(self, x):
		search_cat = self.model(x)
		out = self.adjust(search_cat[4])
		return search_cat, out


# ScoreBranch
class ScoreBranch(nn.Module):
	def __init__(self):
		super(ScoreBranch, self).__init__()
		self.branch = nn.Sequential(
			nn.Conv2d(256, 1024, 1), 
			nn.ReLU(),  
			nn.Conv2d(1024, 2, 1), 
			nn.Sigmoid(),
		)

	def forward(self, x):
		score = self.branch(x)
		max_value = score[0][1].max()
		pos = (score == max_value).nonzero()[0][2:]
		# print('*'*30)
		# print('pos:', pos.shape)
		# print('*'*30)
		return score, pos


# MaskBranch
class MaskBranch(nn.Module):
	def __init__(self):
		super(MaskBranch, self).__init__()
		# self.branch = nn.Sequential(
		# 	nn.Conv2d(256, 256, 1), 
		# 	nn.BatchNorm2d(256), 
		# 	nn.ReLU(), 
		# 	nn.Conv2d(256, 16*16, 1), 
		# )
		self.deconv = nn.ConvTranspose2d(256, 32, 16, 16) # for refine model

	def forward(self, x):
		x = x.reshape(-1, 256, 1, 1)
		out = self.deconv(x)
		return out


# HeadModel
class HeadModel(nn.Module):
	def __init__(self):
		super(HeadModel, self).__init__()
		self.search = BackboneUNet(SEARCH_SIZE)
		self.target = BackboneUNet(TARGET_SIZE)
		self.mask_branch = MaskBranch()
		self.score_branch = ScoreBranch()
		self.up_and_cat = UpAndCat()

		self.up_conv_4 = nn.Sequential(ConvRelu(512+32, 512),
									   ConvRelu(512, 512)
									   )
		self.up_conv_3 = nn.Sequential(ConvRelu(512+256, 256),
									   ConvRelu(256, 256)
									   )
		self.up_conv_2 = nn.Sequential(ConvRelu(256+128, 128),
									   ConvRelu(128, 128)
									   )
		self.up_conv_1 = nn.Sequential(ConvRelu(128+64, 64),
									   ConvRelu(64, 64)
									   )
		self.final = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1),			 
								   nn.Sigmoid()
								   )

	def Correlation_func(self, x, kernel):
		x = x.view(1, -1, x.size(2), x.size(3))  # 1 * (b*c) * k * k
		kernel = kernel.view(-1, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
		out = F.conv2d(x, kernel, groups=x.size(1))
		out = out.view(1, x.size(1), out.size(2), out.size(3))
		return out

	# Forward
	def forward(self, search, target):
		'''
		TODO: 
		    permute for time sequence: (batch, time, channels, input_size, input_size) -->(batch*time, channels, input_size, input_size)
		'''
		search_cats, search = self.search(search)
		_, target = self.target(target)
		corr_feat = self.Correlation_func(search, target)
		score, pos = self.score_branch(corr_feat)

		corr_feat = corr_feat.permute(2, 3, 0, 1)
		out = corr_feat[pos[0]][pos[1]]
		out = self.mask_branch(out)

		out = self.up_and_cat(out, search_cats[3])
		out = self.up_conv_4(out)

		out = self.up_and_cat(out, search_cats[2])
		out = self.up_conv_3(out)

		out = self.up_and_cat(out, search_cats[1])
		out = self.up_conv_2(out)

		out = self.up_and_cat(out, search_cats[0])
		out = self.up_conv_1(out)

		out = self.final(out)

		return score, out


if __name__ == '__main__': 
	search = torch.rand([1, 3, 256, 256])
	target = torch.rand([1, 3, 128, 128])
	model = HeadModel()
	out = model(search, target)
	print(out)
	print(out.shape)