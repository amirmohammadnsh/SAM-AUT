import numpy as np

import torch
import torchvision
from torchvision.transforms import Resize
from torch.nn import functional as F

import os
import PIL
from collections import defaultdict

from sam2.utils.transforms import SAM2Transforms


class PAUTFLAW2(torchvision.datasets.CocoDetection):

    def __init__(self, dataset_root,split="train",filter_empty=True,preprocess=None):

        self.img_folder = os.path.join(dataset_root,split,"images")
        self.ann_dir = os.path.join(dataset_root,split,"annotations","instances_"+split+".json")
        self.preprocess = preprocess
        self.filter_empty = filter_empty

        super(PAUTFLAW2, self).__init__(self.img_folder, self.ann_dir)
        self.ids = list(self.coco.imgs.keys())

        if self.filter_empty:
            self.ids = [ids for ids in self.ids if len(self.coco.getAnnIds(imgIds=ids)) > 0]
    
    def get_list_annotated_images(self):
        return self.ids


    def process_mask(self,mask):

        transform = torch.jit.script(
            torch.nn.Sequential(
                Resize((256, 256)),
            )
        )

        resized_mask_torch = transform(torch.as_tensor(mask).unsqueeze(0)).squeeze(0)

        return resized_mask_torch        
    

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img, _ = super(PAUTFLAW2, self).__getitem__(int(idx))
        image_id = self.ids[idx]

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []
        gt_bboxes = []
        if anns!=[]:
            for ann in anns:
                x, y, w, h = ann['bbox']
                gt_bboxes.append([x, y, x + w, y + h])
                mask = self.coco.annToMask(ann)
                masks.append(mask)
            bboxes.append([0,0,img.size[0],img.size[1]])             

            masks = np.stack(masks, axis=0)
            summed_masks = np.sum(masks, axis=0).astype(np.uint8)
            padded_mask = self.process_mask(summed_masks)
            masks_total = padded_mask.to(torch.int16)



        else:
            # if not self.filter_empty:
            #     bboxes.append([0,0,img.size[0],img.size[1]])
            #     mask = np.full((img.size[1],img.size[0]), False, dtype=bool)
            #     padded_mask = self.process_mask(mask)
            #     masks_total = padded_mask.reshape(1, padded_mask.shape[0] , padded_mask.shape[1])                
            pass

        img_bboxes_masks = defaultdict(dict)        
        if self.preprocess == None:

            img_bboxes_masks['image'] = img
            img_bboxes_masks['bboxes'] = bboxes
            img_bboxes_masks['masks'] = masks_total
            img_bboxes_masks['image_ids'] = image_id

            return img_bboxes_masks
        elif self.preprocess =="Hiera":
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img)

            transform = SAM2Transforms(resolution=1024, mask_threshold=0)
            input_image = transform(img.copy())
            
            original_image_size = img.shape[:2]

            bboxes_torch = torch.as_tensor(np.array(bboxes), dtype=torch.float16)

            box_torch = transform.transform_boxes(bboxes_torch,normalize=True,orig_hw=original_image_size)

            img_bboxes_masks['pixel_values'] = input_image.squeeze(0)
            img_bboxes_masks['original_sizes'] = torch.tensor(original_image_size, dtype=torch.int16)

            img_bboxes_masks['input_boxes'] = box_torch
            img_bboxes_masks['labels'] = masks_total

            img_bboxes_masks['gt_bboxes'] = torch.tensor(gt_bboxes).to(torch.float16)

            img_bboxes_masks['image_ids'] = image_id
            return img_bboxes_masks