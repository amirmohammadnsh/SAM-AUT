import argparse
import random
import numpy as np
import os
from tqdm import tqdm

from dataset import PAUTFLAW
from utils import collate_fn,init_wandb,AnnotationBalancedKFoldPAUTFLAW,set_seed
from PAUTSAM import PAUTSAM
from evaluator import evaluate


import torch
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler  
from torch.distributed import init_process_group,destroy_process_group


from segment_anything import sam_model_registry

from trainer import SAMTrainer
from trainerDDP import SAMTrainerDDP

from sam_LoRa import LoRA_Sam


def ddp_setup():

    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def ddp_training(
    args,
):

    ddp_setup()

    train_dataset = PAUTFLAW(dataset_root=args.dataset_path,split="train",\
        filter_empty=args.filter_empty,preprocess=args.backbone)    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,collate_fn=collate_fn,\
        shuffle=False,pin_memory=True,sampler=DistributedSampler(train_dataset),num_workers=16)

    if int(os.environ["LOCAL_RANK"]) == 0:
        model_save_path = os.path.join(args.work_dir, args.task_name)
        os.makedirs(model_save_path, exist_ok=True)
        print("Model is saving in "+model_save_path)

        val_dataset = PAUTFLAW(dataset_root=args.dataset_path,split="val",preprocess=args.backbone)
        val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size,pin_memory=True,\
            shuffle=True,collate_fn=collate_fn)


    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)   
    trainer = SAMTrainerDDP(
        gpu_id = int(os.environ["LOCAL_RANK"]),
        model = sam_model,
        selected_blocks = args.selected_blocks,
        trainloader = train_dataloader,
        valloader = val_dataloader if int(os.environ["LOCAL_RANK"]) == 0 else None,
        testloader = None,
        model_save_path = model_save_path if int(os.environ["LOCAL_RANK"]) == 0 else "",
        lr = args.lr,
        weight_decay = args.weight_decay,
        max_epochs = args.max_epochs,
        val_epoch_duration=args.val_epoch_duration,
        min_area=args.min_area,
        kernel_size=args.kernel_size,
        dilate_iter=args.dilate_iter,
        iou_thresh=args.iou_thresh,
        world_size = int(os.environ["LOCAL_WORLD_SIZE"])    
    )
    trainer.train()
    destroy_process_group()


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, \
        default='/path/to/COCO_format_dataset/', help='path to dataset')
    parser.add_argument('--task_name', type=str, default='MskDec')
    parser.add_argument('--run_name', type=str, default='MskDec')

    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam_vit_b_01ec64.pth')

    
    parser.add_argument("--single_device", action="store_true", default=False, help="use one gpu")
    # means single device
    parser.add_argument('--multi_device', action="store_true", default=False, help="use multi gpu") 
    # means multi device (all available gpus)

    parser.add_argument('--work_dir', type=str, default='./work_dir')
    parser.add_argument('--backbone', type=str, default='ViT')

    # train
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--base_lr', type=float, default=1e-5)
    parser.add_argument('--do_warmup', action='store_true', default=False, help='If activated, warm up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=0,
                        help='Warm up iterations, only valid when warmup is activated')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_reducer_factor', type=float, default=0.9)


    parser.add_argument('--filter_empty', action="store_true", default=False, help="filter images without annots in the dataset")
    # False means include images does not have any annotations, based on PAUTFLAW it will have 338 images in total
    # True means exclude images does not have any annotations, based on PAUTFLAW it will have 95 images in total
    parser.add_argument('--selected_blocks', type=int, default=0)
    # 0 stands for finetuning the Mask Decoder
    # 1 stands for finetuning both Mask Decoder & Image Encoder

    parser.add_argument('--finetune_mode', type=int, default=0)
    # 0 means default fine tuning without applying any PEFT or Adaptors
    # 1 means lora fine tuning
    
    parser.add_argument("--if_encoder_lora_layer", action="store_true", default=False, help="Enable encoder LoRA layer")
    parser.add_argument("--if_decoder_lora_layer", action="store_true", default=False, help="Enable decoder LoRA layer")
    # if you pass each of the above lora layers, the value will become True.
    parser.add_argument('--encoder_lora_layer', nargs="+", type=int, default=0 , help='the depth of blocks to add lora, if [], it will add at each layer')
    # vit_b [0,1,10,11]
    # vit_l [0,1,14,15]
    # vit_h [0,1,30,31]
    parser.add_argument('--lora_rank', type=int, default=4)
    
    # val
    parser.add_argument('--val_batch_size', type=int, default=10)
    parser.add_argument('--val_epoch_duration', type=int, default=5)
    parser.add_argument("--do_filtering", action="store_true", default=False, help="Apply morphological operations on the predicted masks")
    parser.add_argument('--min_area', type=float, default=4.5)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dilate_iter', type=int, default=3)
    parser.add_argument('--iou_thresh', type=float, default=0.5)

    parser.add_argument("--do_ce_loss", action="store_true", default=False, help="Calculate Binary Cross Entropy Loss")
    parser.add_argument("--ce_loss_weight", type=float, default=0)
    parser.add_argument("--do_iou_loss", action="store_true", default=False, help="Calculate IOU Loss")
    parser.add_argument("--iou_loss_weight", type=float, default=0)

    parser.add_argument("--do_5_fold_cross_validation", action="store_true", default=False, help="Apply 5 fold cross validation")
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--train_data_portion", type=float, default=1.0)
    parser.add_argument("--use_binned_stratify", action="store_true", default=False, help="Apply bins for stratification")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="use wandb to monitor training")
    args = parser.parse_args()


    # single gpu mode
    if args.single_device:

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.cuda.empty_cache()
        set_seed(42)
        print("Using GPU "+str(int(os.environ["CUDA_VISIBLE_DEVICES"])+1)+".")

        if args.use_wandb:
            import wandb
            wandb.require("core")
            wandb.login()

        if not args.do_5_fold_cross_validation:


            model_save_path = os.path.join(args.work_dir, args.task_name,args.run_name)
            os.makedirs(model_save_path, exist_ok=True)
            print("Model is saving in "+model_save_path)

            train_dataset = PAUTFLAW(dataset_root=args.dataset_path,split="train",\
                filter_empty=args.filter_empty, preprocess=args.backbone)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,pin_memory=True,\
                shuffle=True,collate_fn=collate_fn,num_workers=4)

            val_dataset = PAUTFLAW(dataset_root=args.dataset_path,split="val",preprocess=args.backbone)
            val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size,pin_memory=True,\
                shuffle=True,collate_fn=collate_fn,num_workers=1)
                
            sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
            pautsam_model = PAUTSAM(
                image_encoder=sam_model.image_encoder,
                mask_decoder=sam_model.mask_decoder,
                prompt_encoder=sam_model.prompt_encoder,
                selected_blocks=args.selected_blocks
            )
            if args.finetune_mode == 1:
                pautsam_model = LoRA_Sam(args,pautsam_model,r=args.lora_rank).sam
            

            if args.use_wandb:
                init_wandb(args)

            SAM_Trainer = SAMTrainer(args,gpu_id=int(os.environ["LOCAL_RANK"]), model = pautsam_model,\
                trainloader = train_dataloader,valloader=val_dataloader, testloader=None,\
                model_save_path=model_save_path)
            SAM_Trainer.train()


        elif args.do_5_fold_cross_validation:
            kfold_dataset = AnnotationBalancedKFoldPAUTFLAW(dataset_root=args.dataset_path,split="k_fold",filter_empty=args.filter_empty,\
                                                            preprocess=args.backbone, task_name=args.task_name,\
                                                                test_size=args.test_split,use_bins=args.use_binned_stratify,n_splits=5,train_data_portion=args.train_data_portion)
            for fold in range(5):
                set_seed(42)
                train_dataset, val_dataset = kfold_dataset.get_fold(fold)
                print(f"\nTraining on Fold {fold + 1}:")
                if args.train_data_portion == 1.0:
                    print(f"  Train set size: {len(train_dataset)} images")
                    print(f"  Validation set size: {len(val_dataset)} images")
                args.run_name = f"fold_{fold+1}"
                model_save_path = os.path.join(args.work_dir, args.task_name,args.run_name)
                os.makedirs(model_save_path, exist_ok=True)
                print("Model is saving in "+model_save_path)

                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,pin_memory=True,\
                    shuffle=True,collate_fn=collate_fn,num_workers=4)

                val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size,pin_memory=True,\
                    shuffle=True,collate_fn=collate_fn,num_workers=1)
                    
                sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

                pautsam_model = PAUTSAM(
                    image_encoder=sam_model.image_encoder,
                    mask_decoder=sam_model.mask_decoder,
                    prompt_encoder=sam_model.prompt_encoder,
                    selected_blocks=args.selected_blocks
                )
                if args.finetune_mode == 1:
                    pautsam_model = LoRA_Sam(args,pautsam_model,r=args.lora_rank).sam
                

                if args.use_wandb:
                    init_wandb(args)

                SAM_Trainer = SAMTrainer(args,gpu_id=int(os.environ["LOCAL_RANK"]), model = pautsam_model,\
                    trainloader = train_dataloader,valloader=val_dataloader, testloader=None,\
                    model_save_path=model_save_path)
                f1 = SAM_Trainer.train()
                kfold_dataset.fold_results[fold]['f1_score'] = f1

                del train_dataloader
                del val_dataloader
                del sam_model
                del pautsam_model
                del SAM_Trainer

                torch.cuda.empty_cache()
                if args.use_wandb:
                    wandb.finish(quiet=True)

            # kfold_dataset.calculate_overall_f1_score()
            fold_index = max(range(len(kfold_dataset.fold_results)), key=lambda i: (kfold_dataset.fold_results[i]['f1_score'],kfold_dataset.fold_results[i]['num_samples'],-i))
            print(f"Best model is from fold {fold_index+1}.")
            print()
            test_dataset = kfold_dataset.get_test_set()
            set_seed(42)

            test_dataloader = DataLoader(test_dataset, batch_size=args.val_batch_size,pin_memory=True,\
                shuffle=True,collate_fn=collate_fn,num_workers=1)
            
            output_save_path = os.path.join(args.work_dir,args.task_name,f"fold_{fold_index+1}","output_test")
            os.makedirs(output_save_path, exist_ok=True)

            if args.finetune_mode == 0:
                model_load_path = os.path.join(args.work_dir,args.task_name,f"fold_{fold_index+1}",'sam_model_best.pth') 
                sam_model = sam_model_registry[args.model_type](checkpoint=model_load_path)

            elif args.finetune_mode == 1:
                sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)        
                sam_model = LoRA_Sam(args,sam_model,r=args.lora_rank).sam
                model_load_path = os.path.join(args.work_dir, args.task_name,f"fold_{fold_index+1}",'sam_model_best.pth')
                sam_model.load_state_dict(torch.load(model_load_path), strict = False)

            pautsam_model = PAUTSAM(
                image_encoder=sam_model.image_encoder,
                mask_decoder=sam_model.mask_decoder,
                prompt_encoder=sam_model.prompt_encoder,
                selected_blocks=args.selected_blocks
            )

            evaluate(
                valloader = test_dataloader,
                device = int(os.environ["LOCAL_RANK"]),
                model = pautsam_model.to(int(os.environ["LOCAL_RANK"])),
                do_filtering = args.do_filtering,
                kernel_size = args.kernel_size,
                dilate_iter = args.dilate_iter,
                iou_thresh = args.iou_thresh,
                min_area = args.min_area,
                output_save_path = output_save_path,
            )

    elif args.multi_device:
        torch.cuda.empty_cache()
        set_seed(42)

        ddp_training(args)

    else:
        print("You may select single-gpu or multi-gpu mode.")

if __name__ =="__main__":
    main()
