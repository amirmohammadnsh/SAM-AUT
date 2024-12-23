#!/bin/bash
CUDA_VISIBLE_DEVICES=3 torchrun --standalone main.py --task_name "LoRA_Hiera_Large_918_1e-3_1_ND" --dataset_path '/path/to/COCO format dataset/' --test_split 0.2  --model_cfg sam2_hiera_l --checkpoint ./checkpoints/sam2_hiera_large.pt --use_wandb --do_5_fold_cross_validation --val_epoch_duration 1 --single_device --max_epochs 10 --batch_size 4 --base_lr 1e-3 --filter_empty --selected_blocks 1  --finetune_mode 1 --encoder_lora_layer 0 --val_batch_size 32 --min_area 4.5 --lr_reducer_factor 0.9 --weight_decay 0.01 --train_data_portion 1.0

