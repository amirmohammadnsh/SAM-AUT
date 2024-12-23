#!/bin/bash
CUDA_VISIBLE_DEVICES=2 torchrun --standalone main.py --task_name "Hiera_Base+_918_1e-5_2_ND" --dataset_path '/path/to/COCO format dataset/' --test_split 0.2  --model_cfg sam2_hiera_b+ --checkpoint ./checkpoints/sam2_hiera_base_plus.pt --use_wandb --do_5_fold_cross_validation --val_epoch_duration 1 --single_device --max_epochs 25 --batch_size 8 --base_lr 1e-5 --filter_empty --selected_blocks 1  --finetune_mode 0 --val_batch_size 32 --min_area 4.5 --lr_reducer_factor 0.8 --weight_decay 0.01 --train_data_portion 1.0


