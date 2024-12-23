import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import PIL
import os
import operator

def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou

def find_matching_indices(bboxes_list, gt_bboxes_list):
    matching_indices = []

    for bbox in bboxes_list:
        max_iou = -1
        max_index = -1
        
        for i, gt_bbox in enumerate(gt_bboxes_list):
            iou = calculate_iou(bbox, gt_bbox)
            # print(iou)
            if iou > max_iou and iou >= 0.5:
                max_iou = iou
                max_index = i
        
        if max_index != -1:
            matching_indices.append(max_index)
    
    return matching_indices

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 144/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image,alpha=0.5)


def show_box(box, ax,color:str,line_width):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    # ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=1)) 
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=line_width)) 

def get_PIL_image(datasetDirectory,image_info):
    return PIL.Image.open(os.path.join(datasetDirectory, image_info['file_name']))

def vizualize_img_msk(image,gt_bboxes_list,pred_info_dict,bboxes_list,interpolated_mask,output_save_path,image_info):
    dpi = 100
    plt.figure(figsize=(image.size[0]/dpi, image.size[1]/dpi),dpi=dpi)
    plt.imshow(image)
    show_mask(interpolated_mask, plt.gca())
    # print(pred_info_dict)
    if pred_info_dict["FN"] == 0:

        for box in gt_bboxes_list:
            show_box(box,plt.gca(),'green',1)

        tp_indices = [i for i, num in enumerate(pred_info_dict['TP_LIST']) if num == 1]
        if tp_indices !=[]:
            tp_bboxes_list = list(operator.itemgetter(*tp_indices)(bboxes_list))
            if np.array(tp_bboxes_list).ndim == 1:
                show_box(tp_bboxes_list,plt.gca(),'purple',1)
            else:
                for box in tp_bboxes_list:
                    show_box(box,plt.gca(),'purple',1)

        fp_indices = [i for i, num in enumerate(pred_info_dict['FP_LIST']) if num == 1]
        if fp_indices != []:
            fp_bboxes_list = list(operator.itemgetter(*fp_indices)(bboxes_list))
            if np.array(fp_bboxes_list).ndim == 1:
                show_box(fp_bboxes_list,plt.gca(),'red',1)
            else:
                for box in fp_bboxes_list:
                    show_box(box,plt.gca(),'red',1)
    # Case FN !=0
    else:
        if pred_info_dict["TP"] == 0:
            for box in gt_bboxes_list:
                show_box(box,plt.gca(),'blue',1)
            for box in bboxes_list:
                show_box(box,plt.gca(),'red',1)
        # Case TP !=0
        else:
            # print(image_info['file_name'])
            matched_indexes = find_matching_indices(bboxes_list,gt_bboxes_list)
            green_bboxes_list = list(operator.itemgetter(*matched_indexes)(gt_bboxes_list))
            if np.array(green_bboxes_list).ndim == 1:
                show_box(green_bboxes_list,plt.gca(),'green',1)
            else:
                for box in green_bboxes_list:
                    show_box(box,plt.gca(),'green',1)
            blue_bboxes_list = [bbox for i, bbox in enumerate(gt_bboxes_list) if i not in matched_indexes]
            if np.array(blue_bboxes_list).ndim == 1:
                show_box(blue_bboxes_list,plt.gca(),'blue',1)
            else:
                for box in blue_bboxes_list:
                    show_box(box,plt.gca(),'blue',1)

            tp_indices = [i for i, num in enumerate(pred_info_dict['TP_LIST']) if num == 1]
            if tp_indices !=[]:
                tp_bboxes_list = list(operator.itemgetter(*tp_indices)(bboxes_list))
                if np.array(tp_bboxes_list).ndim == 1:
                    show_box(tp_bboxes_list,plt.gca(),'purple',1)
                else:
                    for box in tp_bboxes_list:
                        show_box(box,plt.gca(),'purple',1)

            fp_indices = [i for i, num in enumerate(pred_info_dict['FP_LIST']) if num == 1]
            if fp_indices != []:
                fp_bboxes_list = list(operator.itemgetter(*fp_indices)(bboxes_list))
                if np.array(fp_bboxes_list).ndim == 1:
                    show_box(fp_bboxes_list,plt.gca(),'red',1)
                else:
                    for box in fp_bboxes_list:
                        show_box(box,plt.gca(),'red',1)

    plt.axis('off')
    plt.savefig(os.path.join(output_save_path, image_info['file_name']),bbox_inches='tight', pad_inches=0,dpi=dpi*3)
    plt.close()
