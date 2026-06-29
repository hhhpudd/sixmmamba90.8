import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.FAFuse import FAFuse_B
from utils.dataloader import get_loader, test_dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from test_isic import mean_dice_np, mean_iou_np,recall_score,precision_score
import os
import time
from ptflops import get_model_complexity_info
from datetime import datetime
# from torchstat import stat
from thop import profile
import torch
import subprocess
from lib.circle_transform import *

def polar_loss(pred, mask):
    C2P = CartToPolarTensor(radius=112, img_size=224)
    P2C = PolarToCartTensor(radius=112, img_size=224)
#    pred = C2P(pred)
    mask = mask.to(torch.float32)
#    print(mask.shape)
    mask = C2P(mask)
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()



def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#    print(pred.shape)
#    print(mask.shape)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, best_loss, radiuslist, widthlist, scheduler):
    model.train()
    loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter()
    iters = len(train_loader)
    accum = 0
    for i, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # ---- forward ----
        lateral_map_4, lateral_map_3, lateral_map_2, resmap = model(images)
        #lateral_map_4, lateral_map_3, lateral_map_2, polar0, polar1  = model(images)

        # ---- loss function ----
        loss4 = structure_loss(lateral_map_4, gts) # F0 ⭐
        loss3 = structure_loss(lateral_map_3, gts) # Transformer
        loss2 = structure_loss(lateral_map_2, gts) # up2
        loss1 = structure_loss(resmap, gts) # resmap ⭐

        
#        loss5 = polar_loss(polar0, gts)
#        loss6 = polar_loss(polar1, gts)


        #loss = 0.15*loss2+0.15*loss3+0.7*loss4      # right  left  middle
        loss = 0.4 * loss1 + 0.3 * loss4 + 0.2 * loss2 + 0.1 * loss3
#        loss = 0.15*loss2+0.15*loss3+0.5*loss4+0.1*loss5+0.1*loss6      # right  left  middle
        # ---- backward ----
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()
        scheduler.step(epoch + i/iters)
        optimizer.zero_grad()
        #print("lr:"+str(optimizer.param_groups[0]['lr']))

        # ---- recording loss ----
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record4.update(loss4.data, opt.batchsize)
        


        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'.  
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show()))

    save_path = 'snapshots/{}_with_R{}-{}-{}_W{}-{}-{}/'.format(opt.train_save,radiuslist[0],radiuslist[1],radiuslist[2],widthlist[0],widthlist[1],widthlist[2])
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 1 == 0:
        meanloss = test(model, opt.test_path, epoch)
        if meanloss > best_loss:
            print('new best loss: ', meanloss)
            best_loss = meanloss
            torch.save(model.state_dict(), save_path + 'FAFuse-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path + 'FAFuse-%d.pth'% epoch)
            cleancmd = "find snapshots -mindepth 1 -type d -exec sh -c 'cd \"$0\" && ls -t | tail -n +2 | xargs rm' {} \;"
            print("Cleanning.")
            cleanPro = subprocess.Popen(cleancmd, shell = True)
            print("done.")
            cleanPro.wait()
    return best_loss


def test(model, path, epoch):

    model.eval()
    mean_loss = []

    for s in ['test']:
        image_root = '{}/data_{}.npy'.format(path, s)
        gt_root = '{}/mask_{}.npy'.format(path, s)
        test_loader = test_dataset(image_root, gt_root)

        dice_bank = []
        iou_bank = []
        recall_bank = []
        precision_bank = []
        loss_bank = []
        acc_bank = []

        for i in range(test_loader.size):
            image, gt = test_loader.load_data()
            image = image.cuda()

            with torch.no_grad():
                _, _, res, _ = model(image)
            loss = structure_loss(res, torch.tensor(gt).unsqueeze(0).unsqueeze(0).cuda())

            res = res.sigmoid().data.cpu().numpy().squeeze()
            gt = 1*(gt>0.5)            
            res = 1*(res > 0.5)

            dice = mean_dice_np(gt, res)
            iou = mean_iou_np(gt, res)
            recall = recall_score(gt, res)
            precision = precision_score(gt, res)
            acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

            loss_bank.append(loss.item())
            dice_bank.append(dice)
            iou_bank.append(iou)
            recall_bank.append(recall)
            precision_bank.append(precision)
            acc_bank.append(acc)

        print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}, Recall: {:.4f}, Precision: {:.4f}'.
              format(np.mean(dice_bank), np.mean(iou_bank),  np.mean(acc_bank),np.mean(recall_bank),
                     np.mean(precision_bank)))

        mean_loss.append(np.mean(dice_bank)+np.mean(iou_bank))
        writer.add_scalars('Dice/dice', {"Ring_R{}-{}-{}_W{}-{}-{}".format(radiuslist[0],radiuslist[1],radiuslist[2],widthlist[0],widthlist[1],widthlist[2]):np.mean(dice_bank)}, global_step=epoch)

    return mean_loss[0] 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    #parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='data/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='data/', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='FAFuse_R')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument('--radius1', type=int, default=7, help='radius1 of ring attention')
    parser.add_argument('--radius2', type=int, default=5, help='radius2 of ring attention')
    parser.add_argument('--radius3', type=int, default=3, help='radius3 of ring attention')
    parser.add_argument('--width1', type=int, default=2, help='width1 of ring attention')
    parser.add_argument('--width2', type=int, default=2, help='width2 of ring attention')
    parser.add_argument('--width3', type=int, default=3, help='width3 of ring attention')

    opt = parser.parse_args()
    radiuslist = [opt.radius1, opt.radius2, opt.radius3]
    widthlist = [opt.width1, opt.width2, opt.width3]
    time = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')  #
    writer = SummaryWriter('logs/R{}_{}_{}_W{}-{}-{}'.format(radiuslist[0],radiuslist[1],radiuslist[2],widthlist[0],widthlist[1],widthlist[2]) + time + "/")


    # ---- build models ----
    model = FAFuse_B(pretrained=True, ring_radius=radiuslist, ring_width=widthlist).cuda()
    print(model)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
     
    image_root = '{}/data_train.npy'.format(opt.train_path)
    gt_root = '{}/mask_train.npy'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5, verbose=True)
    # warmup+余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)
    best_loss = 0
    for epoch in range(1, opt.epoch + 1):
        
        best_loss = train(train_loader, model, optimizer, epoch, best_loss, radiuslist, widthlist, scheduler)
        scheduler.step()

    # print("End Training on radius{} width{}".format(opt.radius, opt.width))

