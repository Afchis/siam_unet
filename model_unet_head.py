import torch
import torch.nn as nn

from args import *
from model_unet_paths import *


'''
Model head
'''
class UNetDesigner(nn.Module):    
	def __init__(self, input_size,
				 input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES):
		super(UNetDesigner, self).__init__()
		self.num_classes = NUM_CLASSES
		self.input_size = input_size
		self.input_chennels = input_channels
		self.ch_list = [self.input_chennels, 64, 128, 256, 512, 1024]
		self.input_x2 = int(self.input_size / 2)
		self.input_x4 = int(self.input_size / 4)
		self.input_x8 = int(self.input_size / 8)
		self.input_x16 = int(self.input_size / 8)

		 ##### Down_1 layer ##### input_size = 256                                      # Channels
		self.down1 = nn.Sequential(ConvRelu(self.ch_list[0], self.ch_list[1]),
								   ConvRelu(self.ch_list[1], self.ch_list[1])
								   )
																						# 3  -->64
		self.down1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		 ##### Down_2 layer ##### input_size = 128
		self.down2 = nn.Sequential(ConvRelu(self.ch_list[1], self.ch_list[2]),
								   ConvRelu(self.ch_list[2], self.ch_list[2])
								   )

		self.down2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		 ##### Down_3 layer ##### input_size = 64
		self.down3 = nn.Sequential(ConvRelu(self.ch_list[2], self.ch_list[3]),
			                       ConvRelu(self.ch_list[3], self.ch_list[3])
							       )
																			            # 128-->256
		self.down3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		 ##### Down_4 layer ##### input_size = 32
		self.down4 = nn.Sequential(ConvRelu(self.ch_list[3], self.ch_list[4]),
			                       ConvRelu(self.ch_list[4], self.ch_list[4])
							       )
																			            # 256-->512
		self.down4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

		 ##### Bottom layer ##### input_size = 16
		self.bottom = nn.Sequential(ConvRelu(self.ch_list[4], self.ch_list[5]),
									ConvRelu(self.ch_list[5], self.ch_list[5])
									)
																						# 512-->1028

		 ##### Up_4 layer #####
		self.cat_4 = Cat()
		self.up_conv_4 = nn.Sequential(ConvRelu(self.ch_list[5]+self.ch_list[4], 
												self.ch_list[4]),
									   ConvRelu(self.ch_list[4], self.ch_list[4])
									   )
																						# 1540-->512

		 ##### Up_3 layer #####
		self.up_cat_3 = UpAndCat()
		self.up_conv_3 = nn.Sequential(ConvRelu(self.ch_list[4]+self.ch_list[3], 
												self.ch_list[3]),
							           ConvRelu(self.ch_list[3], self.ch_list[3])
							           )
																						# 768-->256

		 ##### Up_2 layer #####
		self.up_cat_2 = UpAndCat()
		self.up_conv_2 = nn.Sequential(ConvRelu(self.ch_list[3]+self.ch_list[2], 
												self.ch_list[2]),
									   ConvRelu(self.ch_list[2], self.ch_list[2])
									   )
																						# 394-->128

		 ##### Up_1 layer #####
		self.up_cat_1 = UpAndCat()
		self.up_conv_1 = nn.Sequential(ConvRelu(self.ch_list[2]+self.ch_list[1], 
												self.ch_list[1]),
									    ConvRelu(self.ch_list[1], self.ch_list[1])
									    )
																						# 128-->64

	     ##### Final layer #####
		self.final = nn.Sequential(nn.Conv2d(self.ch_list[1], self.num_classes, kernel_size=1),			 
								   nn.Sigmoid())                                                    # 64-->NUM_CLASSES

	def forward(self, x):
		x = x.reshape(-1, self.input_chennels, self.input_size, self.input_size)
		if __name__ == '__main__':
			print('x', x.shape)
		
		down1_feat = self.down1(x)
		pool1 = self.down1_pool(down1_feat)
		if __name__ == '__main__':
			print('pool1', pool1.shape)
		
		down2_feat = self.down2(pool1)
		pool2 = self.down2_pool(down2_feat)
		if __name__ == '__main__':
			print('pool2', pool2.shape)
		
		down3_feat = self.down3(pool2)
		pool3 = self.down3_pool(down3_feat)
		if __name__ == '__main__':
			print('pool3', pool3.shape)
		
		down4_feat = self.down4(pool3)
		if __name__ == '__main__':
			print('down4_feat', down4_feat.shape)
		
		bottom_feat = self.bottom(down4_feat)
		if __name__ == '__main__':
			print('bottom_feat', bottom_feat.shape)
		
		up_feat4 = self.cat_4(bottom_feat, down4_feat)
		up_feat4 = self.up_conv_4(up_feat4)
		if __name__ == '__main__':
			print('up_feat4', up_feat4.shape)
		
		up_feat3 = self.up_cat_3(up_feat4, down3_feat)
		up_feat3 = self.up_conv_3(up_feat3)
		if __name__ == '__main__':
			print('up_feat3', up_feat3.shape)
		
		up_feat2 = self.up_cat_2(up_feat3, down2_feat)
		up_feat2 = self.up_conv_2(up_feat2)
		if __name__ == '__main__':
			print('up_feat2', up_feat2.shape)
		
		up_feat1 = self.up_cat_1(up_feat2, down1_feat)
		up_feat1 = self.up_conv_1(up_feat1)
		if __name__ == '__main__':
			print('up_feat1', up_feat1.shape)
		
		out = self.final(up_feat1)
		if __name__ == '__main__':
			print('out', out.shape)

		return down1_feat, down2_feat, down3_feat, down4_feat, bottom_feat

if __name__ == '__main__':
	tensor = torch.rand([1, 1, 3, 256, 256])
	model = UNetDesigner(256)
	out = model(tensor)