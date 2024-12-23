import os
import random
import argparse
import numpy as np

from dataset import PAUTFLAW2
from utils import collate_fn, AnnotationBalancedKFoldPAUTFLAW,set_seed

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from sam2.build_sam import build_sam2

from PAUTSAM2 import PAUTSAM2
from evaluator import evaluate

from peft import LoraConfig, get_peft_model
import time
def main():

    # torch.cuda.empty_cache()
    # set_seed(42)
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--load_org_SAM', type=int, default=0)
    parser.add_argument('--dataset_path', type=str, \
        default='/path/to/COCO format dataset/', help='path to dataset')
    parser.add_argument('--split', type=str, \
        default='val', help='what split of dataset for evaluation')

    parser.add_argument('--model_cfg', type=str, default='vit_b')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam_vit_b_01ec64.pth')

    parser.add_argument('--task_name', type=str, default='MskDec')
    parser.add_argument('--run_name', type=str, default='MskDec')

    parser.add_argument('--work_dir', type=str, default='./work_dir')

    parser.add_argument('--backbone', type=str, default='ViT')


    parser.add_argument('--finetune_mode', type=int, default=0)
    # 0 means default fine tuning without applying any PEFT or Adaptors
    # 1 means lora fine tuning
    parser.add_argument('--selected_blocks', type=int, default=0)
    # 0 stands for finetuning the Mask Decoder
    # 1 stands for finetuning both Mask Decoder & Image Encoder
    parser.add_argument("--if_encoder_lora_layer", action="store_true", default=False, help="Enable encoder LoRA layer")
    parser.add_argument("--if_decoder_lora_layer", action="store_true", default=False, help="Enable decoder LoRA layer")
    parser.add_argument('--encoder_lora_layer', nargs="+", type=int, default= 0 , help='the depth of blocks to add lora, if [], it will add at each layer')

    # if you pass each of the above lora layers, the value will become True
    # vit_b [0,1,10,11]
    # vit_l [0,1,14,15]
    # vit_h [0,1,30,31]
    parser.add_argument('--lora_rank', type=int, default=4)
    # evaluate
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--do_filtering", action="store_true", default=False, help="Apply morphological operations on the predicted masks")
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--dilate_iter', type=int, default=5)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    parser.add_argument('--min_area', type=float, default=4.5)

    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.empty_cache()
    set_seed(42)
    print("Using GPU "+str(int(os.environ["CUDA_VISIBLE_DEVICES"])+1)+".")

    if args.split in ["train", "val", "test"]:

        val_dataset = PAUTFLAW2(dataset_root=args.dataset_path,split=args.split,preprocess=args.backbone)
        print(f"\nYou selected the option that evalutes model on your {args.split} set.\n")

        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,pin_memory = True,collate_fn=collate_fn)
        output_save_path = os.path.join(args.work_dir,args.task_name,args.run_name,"output_"+args.split)
        os.makedirs(output_save_path, exist_ok=True)




        if args.finetune_mode == 0:
            pass
            # model_load_path = os.path.join(args.work_dir, args.task_name,args.run_name,'sam_model_best.pth')
            # sam_model = sam_model_registry[args.model_cfg](checkpoint=model_load_path)


        elif args.finetune_mode == 1:
            pass
        #     sam_model = sam_model_registry[args.model_cfg](checkpoint=args.checkpoint)        
        #     sam_model = LoRA_Sam(args,sam_model,r=args.lora_rank).sam
        #     model_load_path = os.path.join(args.work_dir, args.task_name,args.run_name,'sam_model_best.pth')
        #     sam_model.load_state_dict(torch.load(model_load_path), strict = False)

        # pautsam_model = PAUTSAM(
        #     image_encoder=sam_model.image_encoder,
        #     mask_decoder=sam_model.mask_decoder,
        #     prompt_encoder=sam_model.prompt_encoder,
        #     selected_blocks=args.selected_blocks
        # )

        # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        # evaluate(
        #     valloader = val_dataloader,
        #     device = int(os.environ["LOCAL_RANK"]),
        #     model = pautsam_model.to(int(os.environ["LOCAL_RANK"])),
        #     do_filtering = args.do_filtering,
        #     kernel_size = args.kernel_size,
        #     dilate_iter = args.dilate_iter,
        #     iou_thresh = args.iou_thresh,
        #     min_area = args.min_area,
        #     output_save_path = output_save_path,
        # )
    elif args.split == "k_fold":
        kfold_json_load_path = os.path.join("k_fold_dataset",args.task_name,"folds_dataset.json")
        kfold_dataset = AnnotationBalancedKFoldPAUTFLAW.load_folds(kfold_json_load_path)  

        for fold in range(5):
            set_seed(42)
            _, val_dataset = kfold_dataset.get_fold(fold)
            print(f"\nValidation on Fold {fold + 1}:")
            print(f"  Validation set size: {len(val_dataset)} images")
            args.run_name = f"fold_{fold+1}"

            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,pin_memory=True,\
                shuffle=True,collate_fn=collate_fn,num_workers=1)

            if args.finetune_mode == 0:
                model_load_path = os.path.join(args.work_dir,args.task_name,args.run_name,'sam_model_best.pth') 
                fineruned_checkpoint = torch.load(model_load_path, map_location="cuda",weights_only=True)
                sam2_model = build_sam2(args.model_cfg,ckpt_path=args.checkpoint)

            elif args.finetune_mode == 1:
                print(args.encoder_lora_layer)
                sam2_model = build_sam2(args.model_cfg,ckpt_path=args.checkpoint)
                lora_config = LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_rank,
                    target_modules=['qkv'],
                    lora_dropout=0,
                    bias="none",
                    modules_to_save=[],
                    layers_to_transform= "null" if args.encoder_lora_layer == 0 else args.encoder_lora_layer
                )
                sam2_model.image_encoder = get_peft_model(sam2_model.image_encoder, lora_config)
                model_load_path = os.path.join(args.work_dir,args.task_name,args.run_name,'sam_model_best.pth')
                fineruned_checkpoint = torch.load(model_load_path, map_location="cuda",weights_only=True)  

            pautsam2_model = PAUTSAM2(
                model=sam2_model,
                selected_blocks=args.selected_blocks
            )
            pautsam2_model.load_state_dict(fineruned_checkpoint, strict=True)

            output_save_path = os.path.join(args.work_dir,args.task_name,args.run_name,"output_val")
            os.makedirs(output_save_path, exist_ok=True)

            f1 = evaluate(
                valloader = val_dataloader,
                device = int(os.environ["LOCAL_RANK"]),
                model = pautsam2_model.to(int(os.environ["LOCAL_RANK"])),
                do_filtering = args.do_filtering,
                kernel_size = args.kernel_size,
                dilate_iter = args.dilate_iter,
                iou_thresh = args.iou_thresh,
                min_area = args.min_area,
                output_save_path = output_save_path,
            )
                
            


            kfold_dataset.fold_results[fold]['f1_score'] = f1

            del val_dataloader
            del sam2_model
            del pautsam2_model

            torch.cuda.empty_cache()

        # kfold_dataset.calculate_overall_f1_score()
        fold_index = max(range(len(kfold_dataset.fold_results)), key=lambda i: (kfold_dataset.fold_results[i]['f1_score'],kfold_dataset.fold_results[i]['num_samples'],-i))
        print(f"Best model is from fold {fold_index+1}.")
        print()
        test_dataset = kfold_dataset.get_test_set()
        set_seed(42)

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,pin_memory=True,\
            shuffle=True,collate_fn=collate_fn,num_workers=1)
        
        output_save_path = os.path.join(args.work_dir,args.task_name,f"fold_{fold_index+1}","output_test")
        os.makedirs(output_save_path, exist_ok=True)

        if args.finetune_mode == 0:
            model_load_path = os.path.join(args.work_dir,args.task_name,f"fold_{fold_index+1}",'sam_model_best.pth') 
            fineruned_checkpoint = torch.load(model_load_path, map_location="cpu",weights_only=True)
            sam2_model = build_sam2(args.model_cfg,ckpt_path=args.checkpoint)

        elif args.finetune_mode == 1:
            print(args.encoder_lora_layer)
            sam2_model = build_sam2(args.model_cfg,ckpt_path=args.checkpoint)
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
                target_modules=['qkv'],
                lora_dropout=0,
                bias="none",
                modules_to_save=[],
                layers_to_transform= "null" if args.encoder_lora_layer == 0 else args.encoder_lora_layer
            )
            sam2_model.image_encoder = get_peft_model(sam2_model.image_encoder, lora_config)
            model_load_path = os.path.join(args.work_dir,args.task_name,f"fold_{fold_index+1}",'sam_model_best.pth')
            fineruned_checkpoint = torch.load(model_load_path, map_location="cpu",weights_only=True)  

        pautsam2_model = PAUTSAM2(
            model=sam2_model,
            selected_blocks=args.selected_blocks
        )
        pautsam2_model.load_state_dict(fineruned_checkpoint, strict=True)
        start = time.time()
        evaluate(
            valloader = test_dataloader,
            device = int(os.environ["LOCAL_RANK"]),
            model = pautsam2_model.to(int(os.environ["LOCAL_RANK"])),
            do_filtering = args.do_filtering,
            kernel_size = args.kernel_size,
            dilate_iter = args.dilate_iter,
            iou_thresh = args.iou_thresh,
            min_area = args.min_area,
            output_save_path = output_save_path,
        )
        end=time.time()
        print(end-start)



if __name__ =="__main__":
    main()
