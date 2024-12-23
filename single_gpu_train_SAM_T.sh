#!/bin/bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone main.py --task_name "ViT_T_918_1e-3_3_ND" --dataset_path '/path/to/COCO_format dataset/' --test_split 0.2  --model_type vit_t --checkpoint ./checkpoints/mobile_sam.pt --use_wandb --do_5_fold_cross_validation --val_epoch_duration 1 --single_device --max_epochs 30 --batch_size 16 --base_lr 1e-3 --filter_empty --selected_blocks 1  --finetune_mode 0 --val_batch_size 8 --min_area 4.5 --weight_decay 0.01 --lr_reducer_factor 0.7 --train_data_portion 1.0


