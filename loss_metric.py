import torch


########## Mask losses: ##########
def l2_loss(x, y, d):
    y = y.reshape(-1, NUM_CLASSES, INPUT_SIZE, INPUT_SIZE)
    d = d.reshape(-1, 1, INPUT_SIZE, INPUT_SIZE)
    out = (d*(x - y)**2).sum()
    #print(out.item())
    return out

def bce_loss(x, y, d):
    y = y.reshape(x.shape)
    bce_loss =  torch.nn.BCELoss()
    return  bce_loss(x, y)

def dice_loss(x, y, d, smooth = 1.):
    y = y.reshape(-1, 2, 256, 256)
    d = d.reshape(-1, 1, 256, 256)
    intersection = (x * y).sum(dim=2).sum(dim=2)
    x_sum = x.sum(dim=2).sum(dim=2)
    y_sum = y.sum(dim=2).sum(dim=2)
    dice_loss = 1 - ((2*intersection + smooth) / (x_sum + y_sum + smooth))
    #print(dice_loss.mean().item())
    return dice_loss.mean()

def dice_combo_loss(x, y, d, bce_weight=0.5):
    dice_combo_loss = bce_weight * bce_loss(x, y, d) + (1 - bce_weight) * dice_loss(x, y, d)
    return dice_combo_loss

def l2_combo_loss(x, y, d):
    l2_combo_loss = l2_loss(x, y, d) * bce_loss(x, y, d)
    return l2_combo_loss


########## Score losses: ##########
def score_loss(xx, yy):
    bce_loss =  torch.nn.BCELoss()
    return  bce_loss(xx, yy)


########## All losses: ##########
def all_losses(x, y, d, xx, yy, score_weight=0.5):
    all_losses = score_weight * dice_loss(x, y, d) + score_weight * score_loss(xx, yy)
    return  all_losses