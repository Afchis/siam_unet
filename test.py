import torch

from args import *
from model_siam import *
from dataloader import *
from loss_metric import *


to_pil = transforms.ToPILImage()

model = HeadModel()
model.load_state_dict(torch.load('weights/test.pth'))

def save_img(object, i):
    imgs = object[0]
    img = (imgs > 0.5).float()
    img = to_pil(img)
    img.save("../siam_unet/test_output/frame%d.png" % i)
    print('save!!!', i)


for i, data in enumerate(train_loader):
	target, search, label, depth = data
	pred_mask = model(search, target)
	save_img(pred_mask, i)
