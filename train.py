#%%
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm

from config import Config
from model import CSRNetFG_multitask, CSRNetFG_naive
from dataset import create_train_dataloader,create_test_dataloader
from utils import denormalize

BG_THRESH = .00001

def masked_multi_soft_cross_entropy(output, target, masks):
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
        target_i = torch.index_select(target, 1, torch.tensor([i*2, i*2+1, 6]).to(target.device))
        output_i = torch.index_select(output, 1, torch.tensor([i*2, i*2+1, 6]).to(target.device))
        
        output_i = -nn.functional.log_softmax(output_i, dim=1)
        mult = torch.mul(output_i, target_i)
        
        # reduce influence of background channel; it is included in all 3 attributes
        mult[:,2,:,:] /= 3
        
        loss += torch.mean(mult[known_mask_i])
    return loss

def masked_multi_mseloss(output, target, masks):
    loss = 0
    # for each attribute (species, sex, age)
    for i in range(3):
        known_mask_i = masks[:,i,:,:].unsqueeze(1).repeat(1,2,1,1)
        target_i = torch.index_select(target, 1, torch.tensor([i*2, i*2+1]).to(target.device))
        output_i = torch.index_select(output, 1, torch.tensor([i*2, i*2+1]).to(target.device))
        loss += nn.MSELoss(size_average=False)(output_i[known_mask_i], target_i[known_mask_i])
    return loss

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=16, help="Batch size. Scale LR linearly with batch size.")
    parser.add_argument("--device", default=0, help="ID of GPU to use.")
    parser.add_argument("--model", default="multitask", help="Which model to use { naive, multitask }.")
    return parser

if __name__ == "__main__":
    args = argument_parser().parse_args()

    import time
    run_id = int(time.time())
    print("Run ID:", run_id)
    
    cfg = Config()
    cfg.batch_size = args.bs
    cfg.lr = cfg.lr * args.bs / 16
    cfg.device = torch.device("cuda:"+str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    
    if args.model == 'multitask':
        model = CSRNetFG_multitask().to(cfg.device)
    elif args.model == 'naive':
        model = CSRNetFG_naive().to(cfg.device)
    else:
        print("Model:", args.model, "not supported")
        exit(1)
        
    mse = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    train_dataloader = create_train_dataloader(cfg.dataset_root, use_flip=True, batch_size=cfg.batch_size)
    test_dataloader  = create_test_dataloader(cfg.dataset_root)

    # track both MAE and fg_MAE for saving checkpoints
    min_mae = sys.maxsize
    min_mae_epoch = -1
    min_fg_mae = sys.maxsize
    min_fg_mae_epoch = -1
    
    for epoch in range(1, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for j, data in enumerate(tqdm(train_dataloader)):
            
            # prep gt data
            image = data['image'].to(cfg.device)
            gt_densitymap = data['densitymap'].to(cfg.device)
            target = data['classmaps'].to(cfg.device)
            gt_cls_densities = torch.index_select(target, 1, torch.tensor([0,1,3,4,6,7]).to(cfg.device))
            
            # create the background channel
            overall_dens = torch.sum(target[:,:3], dim=1)
            background = (overall_dens < BG_THRESH)
            smap_target = torch.index_select(target, 1, torch.tensor([0,1,3,4,6,7]).to(cfg.device)) # will be modified
            smap_masks = torch.index_select(target, 1, torch.tensor([2,5,8]).to(cfg.device)) < BG_THRESH
            
            ####
            # Predict step
            # Get:
            #     et_cls_densities: predicted density maps for each class
            #     et_densitymap: predicted overall density (multitask only)
            #     et_smap: predicted semantic segmentations (multitask only)
            ####
            if args.model == 'multitask':
                et_densitymap, et_smap = model(image)
                et_cls_densities = et_densitymap.repeat(1,6,1,1) * et_smap[:,:6,:,:]
            elif args.model == 'naive':
                et_cls_densities = model(image)
                
            ####
            # Density branch
            ####
            dens_loss = 0
            if args.model == 'multitask':
                dens_loss = mse(et_densitymap, gt_densitymap)
            elif args.model == 'naive':
                # sum classes for each attribute should = overall count
                # make loss = average between the three
                for i in range(3):
                    output_i = torch.index_select(gt_cls_densities, 1, torch.tensor([i*2, i*2+1]).to(target.device))
                    dens_loss += mse(torch.sum(gt_cls_densities, dim=1).unsqueeze(1), gt_densitymap)
                dens_loss /= 3
                
            ####
            # Segmentation branch
            ####
            seg_loss = 0
            if args.model == 'multitask':
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
            cls_dens_loss = masked_multi_mseloss(gt_cls_densities, et_cls_densities, smap_masks)
            ####
            
            loss = dens_loss + seg_loss + cls_dens_loss
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print(dens_loss, seg_loss, cls_dens_loss)
            
            # TESTING
            # if j > 20:
            #     break
        cfg.writer.add_scalar('Train_Loss', epoch_loss/len(train_dataloader), epoch)

        model.eval()
        with torch.no_grad():
            epoch_mae = 0.0
            epoch_cls_maes = np.zeros(6) #[ 0.0 for _ in range(6) ]
            epoch_fg_mae = 0.0
            for i, data in enumerate(tqdm(test_dataloader)):
                image = data['image'].to(cfg.device)
                gt_densitymap = data['densitymap'].to(cfg.device)
                target = data['classmaps'].to(cfg.device)
                smap_masks = torch.index_select(target, 1, torch.tensor([2,2,5,5,8,8]).to(cfg.device)) < BG_THRESH
                gt_cls_densities = torch.index_select(target, 1, torch.tensor([0,1,3,4,6,7]).to(cfg.device))
                
                if args.model == 'multitask':
                    et_densitymap, et_smap = model(image)
                    et_densitymap = et_densitymap.detach()
                    et_smap = et_smap.detach()
                    et_cls_densities = et_densitymap.repeat(1,6,1,1) * et_smap[:,:6,:,:]
                elif args.model == 'naive':
                    et_cls_densities = model(image)
                
                # class-agnostic MAE
                if args.model == 'multitask':
                    mae = abs(et_densitymap.data.sum() - gt_densitymap.data.sum())
                    epoch_mae += mae.item()
                elif args.model == 'naive':
                    # not totally sure how to do this fairly
                    mae = abs(et_cls_densities.data.sum() - 3*gt_densitymap.data.sum()) / 3
                    epoch_mae += mae.item()
                
                # fine-grained multiclass MAE
                fg_maes = np.array([
                    abs(et_cls_densities[:,i,:,:][smap_masks[:,i,:,:]].data.sum() - gt_cls_densities[:,i,:,:][smap_masks[:,i,:,:]].data.sum()).item() for i in range(6)
                ])
                fg_overall_mae = sum(fg_maes) / len(fg_maes)
                
                epoch_cls_maes += fg_maes
                epoch_fg_mae += fg_overall_mae
                
                # print("fg maes", fg_maes)
                # print("class agnostic mae", mae, "fg overall mae", fg_overall_mae)
                
                # TESTING
                # if i > 20:
                #     break
                
            epoch_mae /= len(test_dataloader)
            epoch_cls_maes /= len(test_dataloader)
            epoch_fg_mae /= len(test_dataloader)
            
            if epoch_mae < min_mae:
                min_mae, min_mae_epoch = epoch_mae, epoch
                torch.save(model.state_dict(), os.path.join(cfg.checkpoints,str(run_id) + "_" + str(epoch)+".pth"))
            print('Epoch ', epoch, ' MAE: ', epoch_mae, ' Min MAE: ', min_mae, ' Min Epoch: ', min_mae_epoch)
            
            if epoch_fg_mae < min_fg_mae:
                min_fg_mae, min_fg_mae_epoch = epoch_fg_mae, epoch
                torch.save(model.state_dict(), os.path.join(cfg.checkpoints,str(run_id) + "_" + str(epoch)+".pth"))
            print('Epoch ', epoch, 'FG MAE: ', epoch_fg_mae, ' Min FG MAE: ', min_fg_mae, ' Min FG Epoch: ', min_fg_mae_epoch)
            
            if epoch > 0 and epoch % 25 == 0:
                torch.save(model.state_dict(), os.path.join(cfg.checkpoints,str(run_id) + "_" + str(epoch)+".pth"))
            
            cfg.writer.add_scalar('Val_MAE', epoch_mae, epoch)
            cfg.writer.add_scalar('Val_FG_MAE', epoch_fg_mae, epoch)
            
            for i in range(6):
                cfg.writer.add_scalar(f'Val_cls{i}_MAE', epoch_cls_maes[i], epoch)
            
            cfg.writer.add_image(str(epoch)+'/Image', denormalize(image[0].cpu()))
            
            if args.model == 'multitask':
                cfg.writer.add_image(str(epoch)+'/Estimate density count:'+ str('%.2f'%(et_densitymap[0].cpu().sum())), torch.clip(et_densitymap[0]/torch.max(et_densitymap[0]), 0.0, 1.0))
                cfg.writer.add_image(str(epoch)+'/Ground Truth count:'+ str('%.2f'%(gt_densitymap[0].cpu().sum())), torch.clip(gt_densitymap[0]/torch.max(gt_densitymap[0]), 0.0, 1.0))
            
            for i in range(6):
                gt_map = gt_cls_densities[0][i].unsqueeze(0)
                et_map = et_cls_densities[0][i].unsqueeze(0)
                cfg.writer.add_image(str(epoch)+f'/Cls {i} Estimate density count:'+ str('%.2f'%(et_map.cpu().sum())), torch.clip(et_map/torch.max(et_map), 0.0, 1.0))
                cfg.writer.add_image(str(epoch)+f'/Cls {i} Ground Truth count:'+ str('%.2f'%(gt_map.cpu().sum())), torch.clip(gt_map/torch.max(gt_map), 0.0, 1.0))
            
# %%
