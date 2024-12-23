import numpy as np
import torchvision
from torch.nn import functional as F
import os
import PIL
from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide


class PAUTFLAW(torchvision.datasets.CocoDetection):

    def __init__(self, dataset_root,split="train",filter_empty=True,preprocess=None):
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

        self.padded_h_mask = 0
        self.padded_w_mask = 0 

        self.img_folder = os.path.join(dataset_root,split,"images")
        self.ann_dir = os.path.join(dataset_root,split,"annotations","instances_"+split+".json")
        self.preprocess = preprocess
        self.filter_empty = filter_empty

        super(PAUTFLAW, self).__init__(self.img_folder, self.ann_dir)
        self.ids = list(self.coco.imgs.keys())

        if self.filter_empty:
            self.ids = [ids for ids in self.ids if len(self.coco.getAnnIds(imgIds=ids)) > 0]
    
    def get_list_annotated_images(self):
        return self.ids


    def process_mask(self,mask):
        transform = ResizeLongestSide(256)
        resized_mask = transform.apply_image(mask)
        resized_mask_torch = torch.as_tensor(resized_mask)
        h, w = resized_mask_torch.shape[-2:]
        self.padded_h_mask = h
        self.padded_w_mask = w
        padh = 256 - h
        padw = 256 - w
        x = F.pad(resized_mask_torch, (0, padw, 0, padh))
        return x        
    

    def sam_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img, _ = super(PAUTFLAW, self).__getitem__(idx)
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
            masks_total = padded_mask.to(torch.int64)



        else:
            if not self.filter_empty:
                # bboxes.append([0,0,img.size[0],img.size[1]])
                # mask = np.full((img.size[1],img.size[0]), False, dtype=bool)
                # padded_mask = self.process_mask(mask)
                # masks_total = padded_mask.reshape(1, padded_mask.shape[0] , padded_mask.shape[1])                
                pass

        img_bboxes_masks = defaultdict(dict)        
        if self.preprocess == None:

            img_bboxes_masks['image'] = img
            img_bboxes_masks['bboxes'] = bboxes
            img_bboxes_masks['masks'] = masks_total
            img_bboxes_masks['image_ids'] = image_id

            return img_bboxes_masks
        elif self.preprocess =="ViT":
            img = np.array(img)
            transform = ResizeLongestSide(1024)
            input_image = transform.apply_image(img)
            input_image_torch = torch.as_tensor(input_image)
            transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[:, :, :]
            
            input_image = self.sam_preprocess(transformed_image[None,:,:,:])
            original_image_size = img.shape[:2]

            input_size = torch.tensor(transformed_image.shape[-2:])

            box = transform.apply_boxes(np.array(bboxes),original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float64)

            img_bboxes_masks['pixel_values'] = input_image.squeeze(0)
            img_bboxes_masks['reshaped_input_sizes'] = (input_size).to(torch.int64)
            img_bboxes_masks['original_sizes'] = torch.tensor(original_image_size, dtype=torch.int64)
            img_bboxes_masks['reshaped_gt_mask_sizes'] = torch.tensor((self.padded_h_mask,self.padded_w_mask)).to(torch.int64)
            img_bboxes_masks['input_boxes'] = box_torch
            img_bboxes_masks['labels'] = masks_total

            img_bboxes_masks['gt_bboxes'] = torch.tensor(gt_bboxes).to(torch.float64)
            img_bboxes_masks['image_ids'] = image_id
            return img_bboxes_masks