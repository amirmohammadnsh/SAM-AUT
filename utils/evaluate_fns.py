import numpy as np
import cv2
from skimage import measure
from collections import defaultdict

import torch
import torch.nn.functional as F

from sam2.utils.amg import remove_small_regions

def compute_precision(truePositive,pred_count):
    return truePositive / (pred_count + 1e-16)
def compute_recall(truePositive,gt_count):
    return truePositive/(gt_count + 1e-16)
def compute_f1_score(precision,recall):
    return 2 * precision * recall / (precision + recall + 1e-16)
def compute_AP(tp_list,score_list,gt_count):
    return ap(tp_list,score_list,gt_count)

def post_process_mask(input_mask,pixel_values,original_sizes,kind="gt"):
    with torch.no_grad():
        interpolated_mask = F.interpolate(input_mask.unsqueeze(0), (pixel_values.shape[1], pixel_values.shape[2]), mode="bilinear", align_corners=False)
        interpolated_mask = F.interpolate(interpolated_mask, (original_sizes[0],original_sizes[1]), mode="bilinear", align_corners=False)

        if kind == "pred":            
            interpolated_mask = torch.sigmoid(interpolated_mask)
            interpolated_mask = interpolated_mask.cpu().float().numpy().squeeze()
            interpolated_mask = (interpolated_mask > 0.5).astype(np.uint8)
            return interpolated_mask
        elif kind == "gt":
            interpolated_mask = interpolated_mask.cpu().numpy().squeeze().astype(np.uint8)
        return interpolated_mask

def ap(tp, conf, count):
    tp = np.array(tp)
    conf = np.array(conf)
    i = np.argsort(-conf)
    tp, conf = tp[i], conf[i]
    n_gt = count
    fpc = (1-tp[i]).cumsum()
    tpc = (tp[i]).cumsum()
    recall_curve = tpc / (n_gt + 1e-16)
    precision_curve = tpc / (tpc + fpc)

    ap = compute_ap(precision_curve, recall_curve)
    return ap

def compute_ap(precision, recall):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def mask2bbox(mask,do_filtering = True,k_size=3,dilate_iter=3):
    if do_filtering:
        kernel = np.ones((k_size,k_size), np.uint8)
        mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    contours = measure.find_contours(mask, 0.5)

    bboxes_list = []
    for contour in contours:
        # Flip coordinates and flatten the contour
        contour = np.flip(contour, axis=1)

        # Calculate bounding box
        x_coords, y_coords = zip(*contour)
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        bbox = [int(xmin), int(ymin), int(xmax ), int(ymax )]
        bboxes_list.append(bbox)
    return bboxes_list

def iou(a,b):

    left1,top1,right1,down1 = a[0], a[1], a[2], a[3]
    left2,top2,right2,down2 = b[0], b[1], b[2], b[3]
    
    area1 = (right1-left1)*(top1-down1)
    area2 = (right2-left2)*(top2-down2)
    area_sum = area1+area2
    
    left = max(left1,left2)
    right = min(right1,right2)
    top = max(top1,top2)
    bottom = min(down1,down2)

    if left>=right or top>=bottom:
        return 0
    else:
        inter = (right-left)*(top-bottom)
        return inter/(area_sum-inter)

def evaluation(pred_bboxes, confidence_score, gt_bboxes, iou_thresh=0.5):
    pred_count = len(pred_bboxes)
    gt_count = len(gt_bboxes)
    tp_list = []
    fp_list = []
    score_list = []
    matched_gt_boxes = set()

    for pred_box in pred_bboxes:
        max_iou = 0
        max_gt_box = None
        for i, gt_box in enumerate(gt_bboxes):
            if i in matched_gt_boxes:
                continue
            iou_val = iou(pred_box, gt_box)
            if iou_val > max_iou:
                max_iou = iou_val
                max_gt_box = i

        if max_iou >= iou_thresh:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt_boxes.add(max_gt_box)
        else:
            tp_list.append(0)
            fp_list.append(1)
        score_list.append(confidence_score)

    true_positive = sum(tp_list)
    false_positive = sum(fp_list)
    false_negative = gt_count - true_positive

    return true_positive, false_positive, false_negative, tp_list, fp_list, score_list, gt_count, pred_count

def pre_metric_eval(interpolated_mask,min_area,do_filtering,kernel_size,dilate_iter,pred_mask_iou,gt_bboxes,iou_thresh):

    mask, _ = remove_small_regions(interpolated_mask, min_area, 'islands')
    mask, _ = remove_small_regions(mask, min_area, 'holes')

    # bboxes_list = mask2bbox(mask.astype(np.uint8),k_size=args.kernel_size,dilate_iter=args.dilate_iter)
    bboxes_list = mask2bbox(mask.astype(np.uint8),do_filtering=do_filtering,k_size=kernel_size,dilate_iter=dilate_iter)

    gt_bboxes_list = [[int(elem) for elem in bbox_list] for bbox_list in gt_bboxes.tolist() if any(bbox_list)]
    tp,fp,fn,tp_list,fp_list,score_list,gt_count,pred_count = evaluation(bboxes_list,pred_mask_iou,gt_bboxes_list,iou_thresh=iou_thresh)

    return mask,tp,fp,fn,tp_list,fp_list,score_list,gt_count,pred_count,bboxes_list,gt_bboxes_list

def calc_result(dict_id_pred):

    dict_id_pred_sorted = {}
    merged_lists = defaultdict(list)
    result = defaultdict(int)    

    dict_id_pred_sorted = dict(sorted(dict_id_pred.items()))

    for value in dict_id_pred_sorted.values():
        result['TP'] += value['TP']
        result['FP'] += value['FP']
        result['FN'] += value['FN']
        result['GT_COUNT'] += value['GT_COUNT']
        result['PRED_COUNT'] += value['PRED_COUNT']
                
        merged_lists['TP_LIST'].extend(value['TP_LIST'])
        merged_lists['FP_LIST'].extend(value['FP_LIST'])
        merged_lists['SCORE_LIST'].extend(value['SCORE_LIST'])

    p = compute_precision(result['TP'],result['PRED_COUNT'])
    r = compute_recall(result['TP'],result['GT_COUNT'])
    f1 = compute_f1_score(p,r)
    ap = compute_AP(merged_lists['TP_LIST'],merged_lists['SCORE_LIST'],result['GT_COUNT'])
    return result['TP'],result['FP'],result['FN'],p,r,f1, ap