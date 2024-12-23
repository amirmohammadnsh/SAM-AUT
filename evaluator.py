import numpy as np
from tqdm import tqdm
import json

from utils import post_process_mask,pre_metric_eval,get_PIL_image,vizualize_img_msk,calc_result

import torch


@torch.inference_mode()
def evaluate(
    valloader,
    device,
    model,
    do_filtering,
    kernel_size,
    dilate_iter,
    iou_thresh,
    min_area,
    output_save_path,
):
    model.eval()
    dict_id_pred = {}
    for batch in tqdm(valloader, total=len(valloader), desc="Processing"):

        pixel_values, input_boxees = batch['pixel_values'].to(device),batch['input_boxes'].to(device)
        sam_seg_mask,iou_score=model(pixel_values,input_boxees)

        for i,sam_out_msk in enumerate(sam_seg_mask):

            image = get_PIL_image(datasetDirectory = valloader.dataset.img_folder,image_info = valloader.dataset.coco.loadImgs(batch["image_ids"][i].item())[0])

            interpolated_mask = post_process_mask(sam_out_msk,pixel_values[i],batch['reshaped_input_sizes'][i],batch['original_sizes'][i],kind="pred")
            mask,tp,fp,fn,tp_list,fp_list,score_list,gt_count,pred_count,bboxes_list,gt_bboxes_list = pre_metric_eval(interpolated_mask,min_area,do_filtering,kernel_size,dilate_iter,iou_score[i].item(),batch["gt_bboxes"][i],iou_thresh=iou_thresh)
            dict_id_pred [batch["image_ids"][i].item()] = {"TP" : tp,"FP" : fp,"FN" : fn,\
                "TP_LIST" : tp_list,"FP_LIST" : fp_list,'SCORE_LIST' : score_list,'GT_COUNT' : gt_count,'PRED_COUNT' : pred_count}

            vizualize_img_msk(image,gt_bboxes_list,\
                dict_id_pred [batch["image_ids"][i].item()],bboxes_list,mask.astype(np.uint8),\
                    output_save_path,valloader.dataset.coco.loadImgs(batch["image_ids"][i].item())[0])
    

    tp,fp,fn,p,r,f1,ap = calc_result(dict_id_pred)


    print('TP: {:.1f}\t'.format(tp),
        'FP: {:.1f}\t'.format(fp),
        'FN: {:.1f}\t'.format(fn),
        'P: {:.3f}\t'.format(p),
        'R: {:.3f}\t'.format(r),
        'F1: {:.3f}\t'.format(f1),
        'AP: {:.3f}\t'.format(ap))
    
    return f1