#%%
import numpy as np
import time
import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm

from config import Config
from model import CSRNet
from dataset import create_train_dataloader,create_test_dataloader
from utils import denormalize


def masked_multi_soft_cross_entropy(output, target, masks, cls_weight=None, pixel_weight=None):
    """
    Output: B x 7 x h x w
    Target: B x 7 x h x w
    Masks:  B x 3 x h x w
    """
    loss = 0
    # for each attribute (species, sex, age)
    for i in range(3):
        known_mask_i = masks[:,i,:,:].unsqueeze(1).repeat(1,3,1,1)
        
        # select [class0_i, class1_i, background] for this attribute
        target_i = torch.index_select(target, 1, torch.tensor([i*2, i*2+1, 6]))
        output_i = torch.index_select(output, 1, torch.tensor([i*2, i*2+1, 6]))
        
        # print((target_i.detach() - output_i.detach()).sum())
        # print(output_i.shape)
        # print(output_i[0,0:3,0,0])
        
        output_i = -nn.functional.log_softmax(output_i, dim=1)
        # print(output_i[0,0:3,0,0])
        
        loss += torch.mean(torch.mul(output_i[known_mask_i], target_i[known_mask_i]))
    return loss

def masked_multi_mseloss(output, target, masks):
    loss = 0
    # for each attribute (species, sex, age)
    for i in range(3):
        known_mask_i = masks[:,i,:,:].unsqueeze(1).repeat(1,2,1,1)
        target_i = torch.index_select(target, 1, torch.tensor([i*2, i*2+1]))
        output_i = torch.index_select(output, 1, torch.tensor([i*2, i*2+1]))
        loss += nn.MSELoss(size_average=False)(output_i[known_mask_i], target_i[known_mask_i])
    return loss

if __name__=="__main__":
    
    cfg = Config()                                                          # configuration
    model = CSRNet().to(cfg.device)                                         # model
    criterion = nn.MSELoss(size_average=False)                              # objective
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr)              # optimizer
    train_dataloader = create_train_dataloader(cfg.dataset_root, use_flip=True, batch_size=cfg.batch_size)
    test_dataloader  = create_test_dataloader(cfg.dataset_root)             # dataloader

    min_mae = sys.maxsize
    min_mae_epoch = -1
    for epoch in range(1, cfg.epochs):                          # start training
        model.train()
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader)):
            image = data['image'].to(cfg.device)
            gt_densitymap = data['densitymap'].to(cfg.device)
            et_densitymap, et_smap = model(image)                        # forward propagation
            dens_loss = criterion(et_densitymap,gt_densitymap)       # calculate loss
            
            ####
            # Segmentation
            ####
            
            # create the background channel
            BG_THRESH = 0.00001
            target = data['classmaps'].to(cfg.device)
            overall_dens = torch.sum(target[:,:3], dim=1)
            background = (overall_dens < BG_THRESH)
            smap_target = torch.index_select(target, 1, torch.tensor([0,1,3,4,6,7]))
            smap_masks = torch.index_select(target, 1, torch.tensor([2,5,8])) < BG_THRESH
            
            # each class value at each pixel = the fraction of all density at that pixel belonging to that class
            # so, for most of the maps it's binary. there will only be values between 0 and 1 in regions where class densities overlap
            # here we only consider areas with 'known' classifications: p(cls0) + p(cls1) = 1.0
            for i in range(3):
                attr_dens = torch.sum(smap_target[:,i*2:i*2+2], dim=1)
                smap_target[:, i*2:i*2+2] = smap_target[:, i*2:i*2+2] / (attr_dens.unsqueeze(1) + 1e-14)

            for i in range(6):
                smap_target[:,i,:,:][background] = 0

            smap_target = torch.cat((smap_target, background.unsqueeze(1)), 1).to(cfg.device)
            seg_loss = masked_multi_soft_cross_entropy(et_smap, smap_target, smap_masks)
            ####
            
            ####
            # Fine Grained counting loss
            ####
            et_cls_densities = et_densitymap.repeat(1,6,1,1) * et_smap[:,:6,:,:]
            gt_cls_densities = torch.index_select(target, 1, torch.tensor([0,1,3,4,6,7]))
            cls_dens_loss = masked_multi_mseloss(gt_cls_densities, et_cls_densities, smap_masks)
            
            ####
            
            loss = dens_loss + seg_loss + cls_dens_loss
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()                                     # back propagation
            optimizer.step()                                    # update network parameters
            
            # TESTING
            # if i > 20:
            #     break
        cfg.writer.add_scalar('Train_Loss', epoch_loss/len(train_dataloader), epoch)

        model.eval()
        with torch.no_grad():
            epoch_mae = 0.0
            for i, data in enumerate(tqdm(test_dataloader)):
                image = data['image'].to(cfg.device)
                gt_densitymap = data['densitymap'].to(cfg.device)
                et_densitymap = model(image).detach()           # forward propagation
                mae = abs(et_densitymap.data.sum()-gt_densitymap.data.sum())
                epoch_mae += mae.item()
            epoch_mae /= len(test_dataloader)
            if epoch_mae < min_mae:
                min_mae, min_mae_epoch = epoch_mae, epoch
                torch.save(model.state_dict(), os.path.join(cfg.checkpoints,str(epoch)+".pth"))     # save checkpoints
            print('Epoch ', epoch, ' MAE: ', epoch_mae, ' Min MAE: ', min_mae, ' Min Epoch: ', min_mae_epoch)   # print information
            cfg.writer.add_scalar('Val_MAE', epoch_mae, epoch)
            cfg.writer.add_image(str(epoch)+'/Image', denormalize(image[0].cpu()))
            cfg.writer.add_image(str(epoch)+'/Estimate density count:'+ str('%.2f'%(et_densitymap[0].cpu().sum())), torch.clip(et_densitymap[0]/torch.max(et_densitymap[0]), 0.0, 1.0))
            cfg.writer.add_image(str(epoch)+'/Ground Truth count:'+ str('%.2f'%(gt_densitymap[0].cpu().sum())), torch.clip(gt_densitymap[0]/torch.max(gt_densitymap[0]), 0.0, 1.0))
            
# %%
