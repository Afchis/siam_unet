import torch

from args import *
from model_siam import *
from dataloader import *
from loss_metric import *


model = HeadModel()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


iter = 0
for epoch in range(15):
	print('*'*10, 'epoch: ', epoch, '*'*10)
	for i, data in enumerate(train_loader):
		target, search, label, depth = data
		target, search, label, depth = target.to(device), search.to(device), label.to(device), depth.to(device)
		pred_mask = model(search, target)
		loss = dice_loss(pred_mask, label, depth)
		if iter % 20 == 0:
			print(loss.item())
			# print(pred_mask[0][1][128])
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		iter += 1

torch.save(model.state_dict(), 'weights/test.pth')
print('WEIGHTS IS SAVED: weights/test.pth')




