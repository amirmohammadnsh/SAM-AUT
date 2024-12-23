import argparse
import random
import numpy as np
import os
from tqdm import tqdm

from dataset import PAUTFLAW2
from utils import collate_fn,init_wandb,AnnotationBalancedKFoldPAUTFLAW,set_seed
from PAUTSAM2 import PAUTSAM2
from evaluator import evaluate


import torch
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler  
from torch.distributed import init_process_group,destroy_process_group

from sam2.build_sam import build_sam2

from trainer import SAM2Trainer
# from trainerDDP import SAMTrainerDDP

from peft import LoraConfig, get_peft_model


def ddp_setup():

    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def ddp_training(
    args,
):

    ddp_setup()

    train_dataset = PAUTFLAW2(dataset_root=args.dataset_path,split="train",\
        filter_empty=args.filter_empty,preprocess=args.backbone)    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,collate_fn=collate_fn,\
        shuffle=False,pin_memory=True,sampler=DistributedSampler(train_dataset),num_workers=16)

    if int(os.environ["LOCAL_RANK"]) == 0:
        model_save_path = os.path.join(args.work_dir, args.task_name)
        os.makedirs(model_save_path, exist_ok=True)
        print("Model is saving in "+model_save_path)

        val_dataset = PAUTFLAW2(dataset_root=args.dataset_path,split="val",preprocess=args.backbone)
        val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size,pin_memory=True,\
            shuffle=True,collate_fn=collate_fn)

    # The following block requires update for SAM2
    # sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)   
    # trainer = SAMTrainerDDP(
    #     gpu_id = int(os.environ["LOCAL_RANK"]),
    #     model = sam_model,
    #     selected_blocks = args.selected_blocks,
    #     trainloader = train_dataloader,
    #     valloader = val_dataloader if int(os.environ["LOCAL_RANK"]) == 0 else None,
    #     testloader = None,
    #     model_save_path = model_save_path if int(os.environ["LOCAL_RANK"]) == 0 else "",
    #     lr = args.lr,
    #     weight_decay = args.weight_decay,
    #     max_epochs = args.max_epochs,
    #     val_epoch_duration=args.val_epoch_duration,
    #     min_area=args.min_area,
    #     kernel_size=args.kernel_size,
    #     dilate_iter=args.dilate_iter,
    #     iou_thresh=args.iou_thresh,
    #     world_size = int(os.environ["LOCAL_WORLD_SIZE"])    
    # )
    # trainer.train()
    # destroy_process_group()


def main():
    


    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, \
        default='/path/to/COCO format dataset/', help='path to dataset')
    parser.add_argument('--task_name', type=str, default='MskDec')
    parser.add_argument('--run_name', type=str, default='MskDec')

    parser.add_argument('--model_cfg', type=str, default='sam2_hiera_t')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam2_hiera_tiny.pt')

    
    parser.add_argument("--single_device", action="store_true", default=False, help="use one gpu")
    # means single device
    parser.add_argument('--multi_device', action="store_true", default=False, help="use multi gpu") 
    # means multi device (all available gpus)

    parser.add_argument('--work_dir', type=str, default='./work_dir')
    parser.add_argument('--backbone', type=str, default='Hiera')

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
        # print(torch.cuda.get_rng_state(device='cuda'))
        # print(np.random.get_state())
        print("Using GPU "+str(int(os.environ["CUDA_VISIBLE_DEVICES"])+1)+".")

        if args.use_wandb:
            import wandb
            wandb.require("core")
            wandb.login(key="a16835308aa4c2498745436187ef4d6520a70850")

        if not args.do_5_fold_cross_validation:


            model_save_path = os.path.join(args.work_dir, args.task_name,args.run_name)
            os.makedirs(model_save_path, exist_ok=True)
            print("Model is saving in "+model_save_path)

            train_dataset = PAUTFLAW2(dataset_root=args.dataset_path,split="train",\
                filter_empty=args.filter_empty, preprocess=args.backbone)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,pin_memory=True,\
                shuffle=True,collate_fn=collate_fn,num_workers=4)

            val_dataset = PAUTFLAW2(dataset_root=args.dataset_path,split="val",preprocess=args.backbone)
            val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size,pin_memory=True,\
                shuffle=True,collate_fn=collate_fn,num_workers=1)
            # requires update                
            # sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
            # pautsam_model = PAUTSAM(
            #     image_encoder=sam_model.image_encoder,
            #     mask_decoder=sam_model.mask_decoder,
            #     prompt_encoder=sam_model.prompt_encoder,
            #     selected_blocks=args.selected_blocks
            # )
            # if args.finetune_mode == 1:
            #     pautsam_model = LoRA_Sam(args,pautsam_model,r=args.lora_rank).sam
            

            if args.use_wandb:
                init_wandb(args)

            # requires update
            # SAM_Trainer = SAMTrainer(args,gpu_id=int(os.environ["LOCAL_RANK"]), model = pautsam_model,\
            #     trainloader = train_dataloader,valloader=val_dataloader, testloader=None,\
            #    model_save_path=model_save_path)
            # SAM_Trainer.train()


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
                    
                sam2_model = build_sam2(args.model_cfg,args.checkpoint)
                # print(torch.cuda.get_rng_state(device='cuda'))
                # print(np.random.get_state())

                # for name, _ in sam2_model.named_modules():
                #     if any(x in name.lower() for x in ['q', 'k', 'v']):
                #         print(name)
                
                pautsam2_model = PAUTSAM2(
                    model=sam2_model,
                    selected_blocks=args.selected_blocks
                )
                if args.finetune_mode == 1:
                    print(args.encoder_lora_layer)
                    lora_config = LoraConfig(
                        r=args.lora_rank,
                        lora_alpha=args.lora_rank,
                        target_modules=['qkv'],
                        lora_dropout=0,
                        bias="none",
                        modules_to_save=[],
                        layers_to_transform= "null" if args.encoder_lora_layer == 0 else args.encoder_lora_layer
                    )
                    pautsam2_model.sam2_model.image_encoder = get_peft_model(pautsam2_model.sam2_model.image_encoder, lora_config)                    
                

                if args.use_wandb:
                    init_wandb(args)
                SAM2_Trainer = SAM2Trainer(args,gpu_id=int(os.environ["LOCAL_RANK"]), model = pautsam2_model,\
                    trainloader = train_dataloader,valloader=val_dataloader, testloader=None,\
                    model_save_path=model_save_path)
                f1 = SAM2_Trainer.train()
                kfold_dataset.fold_results[fold]['f1_score'] = f1

                del train_dataloader
                del val_dataloader
                del sam2_model
                del pautsam2_model
                del SAM2_Trainer

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
                fineruned_checkpoint = torch.load(model_load_path, map_location="cpu")
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
                fineruned_checkpoint = torch.load(model_load_path, map_location="cpu")              
        
            pautsam2_model = PAUTSAM2(
                model=sam2_model,
                selected_blocks=args.selected_blocks
            )
            pautsam2_model.load_state_dict(fineruned_checkpoint, strict=True)
            
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

    elif args.multi_device:
        torch.cuda.empty_cache()
        set_seed(42)

        ddp_training(args)

    else:
        print("You may select single-gpu or multi-gpu mode.")

if __name__ =="__main__":
    main()
