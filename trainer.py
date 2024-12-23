import numpy as np
import os
import monai
from tqdm import tqdm
import wandb

from utils import dice_score,cal_iou,post_process_mask,pre_metric_eval,calc_result

import torch
from torch.utils.data import DataLoader


class SAM2Trainer:
    
    def __init__(
        self,
        args,
        gpu_id:int,
        model:torch.nn.Module,
        trainloader:DataLoader,
        valloader:DataLoader,
        testloader:DataLoader,
        model_save_path:str,
    ):
        self.base_lr = args.base_lr
        self.b_lr = 0
        self.do_warmup = args.do_warmup
        self.warmup_period = args.warmup_period
        self.weight_decay = args.weight_decay
        self.max_epochs = args.max_epochs
        self.use_wandb = args.use_wandb
        self.do_ce_loss = args.do_ce_loss
        self.ce_loss_weight = args.ce_loss_weight
        self.do_iou_loss = args.do_iou_loss
        self.iou_loss_weight = args.iou_loss_weight
        self.min_area = args.min_area
        self.do_filtering = args.do_filtering
        self.kernel_size = args.kernel_size
        self.dilate_iter = args.dilate_iter
        self.iou_thresh = args.iou_thresh

        self.gpu_id = gpu_id
        torch.cuda.set_device(self.gpu_id)
        self.model = model.to(self.gpu_id)

        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.model_save_path = model_save_path
        self.b_lr = 0

        self.seg_loss = None
        self.ce_loss = None
        self.iou_loss = None

        self.optimizer = None

        self.train_loss_per_batch = []
        self.train_dice_per_batch = []
        self.train_loss_per_epoch = []
        self.train_dice_per_epoch = []
        self.train_f1_scores = []
        self.train_ap_metrics = []

        self.best_loss = 1e10
        self.best_dice = -1e10
        self.epochs_run = 0
        self.step = 0
        self.max_steps = self.max_epochs*len(self.trainloader)
        self.epoch_progress_bar = None
        self.batch_progress_bar = None
        self.best_epoch = 0

        self.val_epoch_duration = args.val_epoch_duration
        self.last_val_epoch = 0

        self.val_loss_per_batch = []
        self.val_dice_per_batch = []
        self.val_loss_per_epoch = []
        self.val_dice_per_epoch = []
        self.val_f1_scores = []
        self.val_ap_metrics = []

        self.best_f1 = -1e10 # It is for validation set
        self.best_ap50 = -1e10 # It is for validation set

        self.shift_step = -1

        self.train_dict_id_pred = {}
        self.train_epoch_predictions = {}
        self.train_epoch_ious = {}

        self.val_dict_id_pred = {}
        self.do_kfold = args.do_5_fold_cross_validation
        self.best_val_loss = 1e10
        self.lr_scheduler = []
        self.lr_reducer_factor = args.lr_reducer_factor
        self.scaler = []

    def dice_score_fn(self,pred,gt):
        return dice_score(pred,gt)
    
    def calc_iou_fn(self,pred,gt):
        return cal_iou(pred,gt)
    
    def post_process_mask_fn(self,input_mask,pixel_values,original_sizes,kind):
        return post_process_mask(input_mask,pixel_values,original_sizes,kind)
    
    def pre_metric_eval_fn(self,interpolated_mask,min_area,do_filtering,kernel_size,dilate_iter,pred_mask_iou,gt_bboxes,iou_thresh):
        return pre_metric_eval(interpolated_mask,min_area,do_filtering,kernel_size,dilate_iter,pred_mask_iou,gt_bboxes,iou_thresh)

    def calc_result_fn(self,dict_id_pred):
        return calc_result(dict_id_pred)



    def setup_optimizer(self):

        if self.model.selected_blocks == 0:

            if self.do_warmup:
                self.b_lr = self.base_lr / self.warmup_period
                self.optimizer = torch.optim.AdamW(self.model.sam2_model.sam_mask_decoder.parameters(),eps=1e-08, lr=self.b_lr,betas=(0.9, 0.999), weight_decay=self.weight_decay)
            else:
                self.b_lr = self.base_lr
                self.optimizer = torch.optim.AdamW(self.model.sam2_model.sam_mask_decoder.parameters(),eps=1e-08, lr=self.b_lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)

        elif self.model.selected_blocks == 1:

            img_mask_encdec_params = list(self.model.sam2_model.image_encoder.parameters()) + list(self.model.sam2_model.sam_mask_decoder.parameters())

            if self.do_warmup:
                self.b_lr = self.base_lr / self.warmup_period
                self.optimizer = torch.optim.Adam(img_mask_encdec_params, lr=self.b_lr,eps=1e-08, betas=(0.9, 0.999),weight_decay=self.weight_decay)
            else:
                self.b_lr = self.base_lr
                self.optimizer = torch.optim.AdamW(img_mask_encdec_params, lr=self.b_lr,eps=1e-08, betas=(0.9, 0.999), weight_decay=self.weight_decay)                

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.lr_reducer_factor,
            patience=0,
            cooldown=0
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
    def setup_loss_fns(self):
        self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        if self.do_ce_loss:
            self.ce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        if self.do_iou_loss:
            self.iou_loss = torch.nn.MSELoss(reduction='mean')

    def _run_batch(self, pixel_values, input_boxes,labels,original_sizes,image_ids,gt_bboxes):
        self.optimizer.zero_grad()
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
            pautsam_mask_pred,pautsam_iou_pred = self.model(pixel_values, input_boxes)

            loss = self.seg_loss(pautsam_mask_pred, labels)

            if self.do_ce_loss and self.ce_loss_weight>0.0:
                loss+=self.ce_loss_weight*self.ce_loss(pautsam_mask_pred, labels)

            if self.do_iou_loss:
                iou_gt = self.calc_iou(torch.sigmoid(pautsam_mask_pred) > 0.5, labels.bool())
                loss+=self.iou_loss_weight*self.iou_loss(pautsam_iou_pred,iou_gt)

        dice = self.dice_score_fn((pautsam_mask_pred>0.5).float(), labels)

        self.train_loss_per_batch.append(loss.item())
        self.train_dice_per_batch.append(dice.item())

        for i,sam_out_msk in enumerate(pautsam_mask_pred):
            interpolated_mask = self.post_process_mask_fn(sam_out_msk.detach(),pixel_values[i],original_sizes[i],kind="pred")
            mask,tp,fp,fn,tp_list,fp_list,score_list,gt_count,pred_count,_,_ = \
                self.pre_metric_eval_fn(interpolated_mask,self.min_area,self.do_filtering,self.kernel_size,self.dilate_iter,pautsam_iou_pred[i].item(),gt_bboxes[i],self.iou_thresh)

            self.train_epoch_predictions[image_ids[i].item()] = mask
            self.train_epoch_ious[image_ids[i].item()] = pautsam_iou_pred[i].item()

            self.train_dict_id_pred [image_ids[i].item()] = {"TP" : tp,"FP" : fp,"FN" : fn,\
                "TP_LIST" : tp_list,"FP_LIST" : fp_list,'SCORE_LIST' : score_list,'GT_COUNT' : gt_count,'PRED_COUNT' : pred_count}
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # loss.backward()
        # self.optimizer.step()
        self.optimizer.zero_grad()


    def _run_epoch(self, epoch: int):

        self.batch_progress_bar = tqdm(self.trainloader, total=len(self.trainloader),leave=False)
        for batch in self.batch_progress_bar:
            self._run_batch(batch['pixel_values'].to(self.gpu_id),batch['input_boxes'].to(self.gpu_id),batch['labels'].float().to(self.gpu_id),
                            batch['original_sizes'],batch["image_ids"],batch["gt_bboxes"])
    
            # if self.do_warmup and self.step < self.warmup_period:
            #     lr_ = self.base_lr * ((self.step+1)/self.warmup_period)
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = lr_
            # else:
            #     if self.do_warmup:
            #         self.shift_step = self.step - self.warmup_period
            #         assert self.step >=0, f"Shift Step is {self.shift_step}. smaller than zero"
            #     else:
            #         self.shift_step = self.step
            #     lr_ = self.base_lr*(1.0 - self.shift_step/self.max_steps)**0.9
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = lr_
            # self.step+=1
        _,_,_,_,_,f1,ap = self.calc_result_fn(self.train_dict_id_pred)
        self.train_f1_scores.append(f1)
        self.train_ap_metrics.append(ap)
        self.train_dict_id_pred = {}

        self.train_loss_per_epoch.append(np.mean(self.train_loss_per_batch))
        self.train_dice_per_epoch.append(np.mean(self.train_dice_per_batch))
        self.lr_scheduler.step(np.mean(self.train_loss_per_batch))
        self.train_loss_per_batch= []
        self.traion_dice_per_batch = []
        # if epoch+1<=self.val_epoch_duration:
        #     self.epoch_progress_bar.set_description(f'EPOCH {epoch+1}| Train: Loss:{self.train_loss_per_epoch[epoch]:.3f} Dice:{self.train_dice_per_epoch[epoch]:.3f} F1:{self.train_f1_scores[epoch]:.3f}',refresh=True)
        # else:
        #     self.epoch_progress_bar.set_description(f'EPOCH {epoch+1}| Train: Loss:{self.train_loss_per_epoch[epoch]:.3f} Dice:{self.train_dice_per_epoch[epoch]:.3f} F1:{self.train_f1_scores[epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)]:.3f} Dice:{self.val_dice_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)]:.3f} F1:{self.val_f1_scores[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)]:.3f} AP:{self.val_ap_metrics[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)]:.3f}',refresh=True)
        # if epoch == 0:
        #     self.epoch_progress_bar.set_description(f'EPOCH {epoch+1}| Train: Loss:{self.train_loss_per_epoch[epoch]:.3f} Dice:{self.train_dice_per_epoch[epoch]:.3f} F1:{self.train_f1_scores[epoch]:.3f}',refresh=True)
        # else:
        #     self.epoch_progress_bar.set_description(f'EPOCH {epoch+1}| Train: Loss:{self.train_loss_per_epoch[epoch]:.3f} Dice:{self.train_dice_per_epoch[epoch]:.3f} F1:{self.train_f1_scores[epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f} Dice:{self.val_dice_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f} F1:{self.val_f1_scores[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f} AP:{self.val_ap_metrics[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f}',refresh=True)


        if self.use_wandb:
            wandb.log({"train/loss": self.train_loss_per_epoch[epoch], "train/dice":self.train_dice_per_epoch[epoch], "train/F1":self.train_f1_scores[epoch], "train/AP":self.train_ap_metrics[epoch]},step=epoch+1)
    
            for image_id in self.train_epoch_ious.keys():
                    wandb.log({
                        # f"image_{image_id}/ground_truth": wandb.Image(epoch_labels[image_id], caption="Ground Truth"),
                        f"image_{image_id}/prediction": wandb.Image(self.train_epoch_predictions[image_id], caption="Prediction"),
                        f"image_{image_id}/iou_prediction": self.train_epoch_ious[image_id]
                    }, step=epoch+1)
                
            # Clear the dictionaries for the next epoch
            self.train_epoch_predictions.clear()
            self.train_epoch_ious.clear()




    def _save_checkpoint(self, epoch: int,mode: str):

        if mode == "best":
            # if self.val_f1_scores[int(((epoch+1)/self.val_epoch_duration)-1)] >= self.best_f1:
            # if self.val_f1_scores[int(((epoch+1)/self.val_epoch_duration))] >= self.best_f1 and self.val_loss_per_epoch[epoch]<=self.best_val_loss:
            if self.val_f1_scores[int(epoch/self.val_epoch_duration)] >= self.best_f1 and self.val_loss_per_epoch[epoch]<=self.best_val_loss:

                # self.best_val_loss = self.val_loss_per_epoch[epoch]
                # self.best_f1 = self.val_f1_scores[int(((epoch+1)/self.val_epoch_duration)-1)]
                # self.best_f1 = self.val_f1_scores[int(((epoch+1)/self.val_epoch_duration))]
                self.best_f1 = self.val_f1_scores[int(epoch/self.val_epoch_duration)]

                torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'sam_model_best.pth'))
                self.best_epoch = epoch
                # It will set postfix of epoch_progress_bar as epoch, F1, AP of the best model based on val set 
                # self.epoch_progress_bar.set_postfix_str(f'Epoch {self.best_epoch+1} F1:{self.val_f1_scores[int(((self.best_epoch+1)/self.val_epoch_duration)-1)]:.3f}')            
                # self.epoch_progress_bar.set_postfix_str(f'Epoch {self.best_epoch+1} F1:{self.val_f1_scores[int(((self.best_epoch+1)/self.val_epoch_duration))]:.3f}')            
                self.epoch_progress_bar.set_postfix_str(f'Epoch {self.best_epoch+1} F1:{self.val_f1_scores[int(epoch/self.val_epoch_duration)]:.3f}')            

        elif mode =="last":
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'sam_model_latest.pth'))
            np.array(self.train_loss_per_epoch).dump(open(os.path.join(self.model_save_path,'train_loss.npy'), 'wb'))
            np.array(self.train_dice_per_epoch).dump(open(os.path.join(self.model_save_path,'train_dice.npy'), 'wb'))
            np.array(self.val_loss_per_epoch).dump(open(os.path.join(self.model_save_path,'val_loss.npy'), 'wb'))
            np.array(self.val_dice_per_epoch).dump(open(os.path.join(self.model_save_path,'val_dice.npy'), 'wb'))
            np.array(self.val_f1_scores).dump(open(os.path.join(self.model_save_path,'f1_scores.npy'), 'wb'))
            np.array(self.val_ap_metrics).dump(open(os.path.join(self.model_save_path,'ap50_metrics.npy'), 'wb'))


    def train(self):

        self.setup_optimizer()
        self.setup_loss_fns()
        self.model.get_total_parameters()
        self.model.get_total_trainable_parameters()

        with tqdm(range(self.epochs_run,self.max_epochs),total=len(range(self.epochs_run,self.max_epochs)),leave=True) as self.epoch_progress_bar:
        
            for epoch in self.epoch_progress_bar:
                self.model.train()
                # print(torch.cuda.get_rng_state(device='cuda'))
                # print(np.random.get_state())

                self._run_epoch(epoch)

                # if (epoch+1) % self.val_epoch_duration == 0: # if we want to skip the epoch==0 validation(loss,dice,metrics)
                # if (epoch+1) % self.val_epoch_duration == 0 or epoch == 0: 
                if epoch % self.val_epoch_duration == 0: 

                    self.model.eval()
                    self._validation(epoch,"eval_metric")
                    self._save_checkpoint(epoch, "best")
                else:
                    self.model.eval()
                    self._validation(epoch)

            self._save_checkpoint(epoch, "last")
            # print(f'Best model at epoch {self.best_epoch+1}| Train: Loss:{self.train_loss_per_epoch[self.best_epoch]:.3f} Dice:{self.train_dice_per_epoch[self.best_epoch]:.3f} F1:{self.train_f1_scores[self.best_epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[int(((self.best_epoch+1)/self.val_epoch_duration)-1)]:.3f} Dice:{self.val_dice_per_epoch[int(((self.best_epoch+1)/self.val_epoch_duration)-1)]:.3f} F1:{self.val_f1_scores[int(((self.best_epoch+1)/self.val_epoch_duration)-1)]:.3f} AP:{self.val_ap_metrics[int(((self.best_epoch+1)/self.val_epoch_duration)-1)]:.3f}')   
            # print(f'Best model at epoch {self.best_epoch+1}| Train: Loss:{self.train_loss_per_epoch[self.best_epoch]:.3f} Dice:{self.train_dice_per_epoch[self.best_epoch]:.3f} F1:{self.train_f1_scores[self.best_epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[int(((self.best_epoch+1)/self.val_epoch_duration))]:.3f} Dice:{self.val_dice_per_epoch[int(((self.best_epoch+1)/self.val_epoch_duration))]:.3f} F1:{self.val_f1_scores[int(((self.best_epoch+1)/self.val_epoch_duration))]:.3f} AP:{self.val_ap_metrics[int(((self.best_epoch+1)/self.val_epoch_duration))]:.3f}')   
            # print(f'Best model at epoch {self.best_epoch+1}| Train: Loss:{self.train_loss_per_epoch[self.best_epoch]:.3f} Dice:{self.train_dice_per_epoch[self.best_epoch]:.3f} F1:{self.train_f1_scores[self.best_epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[self.best_epoch]:.3f} Dice:{self.val_dice_per_epoch[self.best_epoch]:.3f} F1:{self.val_f1_scores[int(((self.best_epoch+1)/self.val_epoch_duration))]:.3f} AP:{self.val_ap_metrics[int(((self.best_epoch+1)/self.val_epoch_duration))]:.3f}')   
            print(f'Best model at epoch {self.best_epoch+1}| Train: Loss:{self.train_loss_per_epoch[self.best_epoch]:.3f} Dice:{self.train_dice_per_epoch[self.best_epoch]:.3f} F1:{self.train_f1_scores[self.best_epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[self.best_epoch]:.3f} Dice:{self.val_dice_per_epoch[self.best_epoch]:.3f} F1:{self.val_f1_scores[int(self.best_epoch/self.val_epoch_duration)]:.3f} AP:{self.val_ap_metrics[int(self.best_epoch/self.val_epoch_duration)]:.3f}')   

            if self.do_kfold:
                # return self.val_f1_scores[int(((self.best_epoch+1)/self.val_epoch_duration)-1)]
                # return self.val_f1_scores[int(((self.best_epoch+1)/self.val_epoch_duration))]
                return self.val_f1_scores[int(self.best_epoch/self.val_epoch_duration)]
            
    def _validation(self,epoch:int,mode=None):

        if mode == "eval_metric":
            self.last_val_epoch = epoch


        with torch.no_grad():
            for batch in self.valloader:
                image,labels,input_box = batch['pixel_values'].to(self.gpu_id),batch['labels'].float().to(self.gpu_id),batch['input_boxes'].to(self.gpu_id)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pautsam_mask_pred,pautsam_iou_pred = self.model(image, input_box)   

                    val_loss = self.seg_loss(pautsam_mask_pred, labels)

                    if self.do_ce_loss:
                        val_loss+=self.ce_loss_weight*self.ce_loss(pautsam_mask_pred, labels)

                    if self.do_iou_loss:
                        iou_gt = self.calc_iou_fn(torch.sigmoid(pautsam_mask_pred) > 0.5, labels.bool())
                        val_loss+=self.iou_loss_weight*self.iou_loss(pautsam_iou_pred,iou_gt)
                    
                val_dice = self.dice_score_fn((pautsam_mask_pred>0.5).float(), labels)

                self.val_loss_per_batch.append(val_loss.item())
                self.val_dice_per_batch.append(val_dice.item())

                if mode == "eval_metric":
                    for i,sam_out_msk in enumerate(pautsam_mask_pred):
                        interpolated_mask = self.post_process_mask_fn(sam_out_msk.detach(),image[i],batch['original_sizes'][i],kind="pred")
                        _,tp,fp,fn,tp_list,fp_list,score_list,gt_count,pred_count,_,_ = \
                            self.pre_metric_eval_fn(interpolated_mask,self.min_area,self.do_filtering,self.kernel_size,self.dilate_iter,pautsam_iou_pred[i].item(),batch["gt_bboxes"][i],self.iou_thresh)

                        self.val_dict_id_pred [batch["image_ids"][i].item()] = {"TP" : tp,"FP" : fp,"FN" : fn,\
                            "TP_LIST" : tp_list,"FP_LIST" : fp_list,'SCORE_LIST' : score_list,'GT_COUNT' : gt_count,'PRED_COUNT' : pred_count}

        if mode == "eval_metric":
            _,_,_,_,_,f1,ap = self.calc_result_fn(self.val_dict_id_pred)
            self.val_f1_scores.append(f1)
            self.val_ap_metrics.append(ap)
            self.val_dict_id_pred = {}

        self.val_loss_per_epoch.append(np.mean(self.val_loss_per_batch))
        self.val_dice_per_epoch.append(np.mean(self.val_dice_per_batch))
        self.val_loss_per_batch= []
        self.val_dice_per_batch = []
        # self.val_dict_id_pred = {}
        if self.val_loss_per_epoch[epoch] <=self.best_val_loss:
            self.best_val_loss = self.val_loss_per_epoch[epoch]
        # self.epoch_progress_bar.set_description(f'EPOCH {epoch+1}| Train: Loss:{self.train_loss_per_epoch[epoch]:.3f} Dice:{self.train_dice_per_epoch[epoch]:.3f} F1:{self.train_f1_scores[epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)]:.3f} Dice:{self.val_dice_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)]:.3f} F1:{self.val_f1_scores[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)]:.3f} AP:{self.val_ap_metrics[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)]:.3f}',refresh=True)
        # self.epoch_progress_bar.set_description(f'EPOCH {epoch+1}| Train: Loss:{self.train_loss_per_epoch[epoch]:.3f} Dice:{self.train_dice_per_epoch[epoch]:.3f} F1:{self.train_f1_scores[epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f} Dice:{self.val_dice_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f} F1:{self.val_f1_scores[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f} AP:{self.val_ap_metrics[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f}',refresh=True)

        # if self.use_wandb:
        #     wandb.log({"val/loss": self.val_loss_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)],\
        #         "val/dice":self.val_dice_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)],\
        #             "val/F1":self.val_f1_scores[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)],\
        #                 "val/AP":self.val_ap_metrics[int(((self.last_val_epoch+1)/self.val_epoch_duration)-1)]},step=epoch+1)

        # if self.use_wandb:
        #     wandb.log({"val/loss": self.val_loss_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration))],\
        #         "val/dice":self.val_dice_per_epoch[int(((self.last_val_epoch+1)/self.val_epoch_duration))],\
        #             "val/F1":self.val_f1_scores[int(((self.last_val_epoch+1)/self.val_epoch_duration))],\
        #                 "val/AP":self.val_ap_metrics[int(((self.last_val_epoch+1)/self.val_epoch_duration))]},step=epoch+1)
        if mode =="eval_metric":
            # self.epoch_progress_bar.set_description(f'EPOCH {epoch+1}| Train: Loss:{self.train_loss_per_epoch[epoch]:.3f} Dice:{self.train_dice_per_epoch[epoch]:.3f} F1:{self.train_f1_scores[epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[epoch]:.3f} Dice:{self.val_dice_per_epoch[epoch]:.3f} F1:{self.val_f1_scores[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f} AP:{self.val_ap_metrics[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f}',refresh=True)
            self.epoch_progress_bar.set_description(f'EPOCH {epoch+1}| Train: Loss:{self.train_loss_per_epoch[epoch]:.3f} Dice:{self.train_dice_per_epoch[epoch]:.3f} F1:{self.train_f1_scores[epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[epoch]:.3f} Dice:{self.val_dice_per_epoch[epoch]:.3f} F1:{self.val_f1_scores[int(epoch/self.val_epoch_duration)]:.3f} AP:{self.val_ap_metrics[int(epoch/self.val_epoch_duration)]:.3f}',refresh=True)

            if self.use_wandb:
                wandb.log({"val/loss": self.val_loss_per_epoch[epoch],\
                    "val/dice":self.val_dice_per_epoch[epoch],\
                        # "val/F1":self.val_f1_scores[int(((self.last_val_epoch+1)/self.val_epoch_duration))],\
                        "val/F1":self.val_f1_scores[int(epoch/self.val_epoch_duration)],\

                            # "val/AP":self.val_ap_metrics[int(((self.last_val_epoch+1)/self.val_epoch_duration))]},step=epoch+1)
                            "val/AP":self.val_ap_metrics[int(epoch/self.val_epoch_duration)]},step=epoch+1)
        else:
            # self.epoch_progress_bar.set_description(f'EPOCH {epoch+1}| Train: Loss:{self.train_loss_per_epoch[epoch]:.3f} Dice:{self.train_dice_per_epoch[epoch]:.3f} F1:{self.train_f1_scores[epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[epoch]:.3f} Dice:{self.val_dice_per_epoch[epoch]:.3f} F1:{self.val_f1_scores[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f} AP:{self.val_ap_metrics[int(((self.last_val_epoch+1)/self.val_epoch_duration))]:.3f}',refresh=True)
            self.epoch_progress_bar.set_description(f'EPOCH {epoch+1}| Train: Loss:{self.train_loss_per_epoch[epoch]:.3f} Dice:{self.train_dice_per_epoch[epoch]:.3f} F1:{self.train_f1_scores[epoch]:.3f}| Val: Loss:{self.val_loss_per_epoch[epoch]:.3f} Dice:{self.val_dice_per_epoch[epoch]:.3f} F1:{self.val_f1_scores[int(epoch/self.val_epoch_duration)]:.3f} AP:{self.val_ap_metrics[int(epoch/self.val_epoch_duration)]:.3f}',refresh=True)

            if self.use_wandb:
                wandb.log({"val/loss": self.val_loss_per_epoch[epoch],\
                    "val/dice":self.val_dice_per_epoch[epoch]},step=epoch+1)


