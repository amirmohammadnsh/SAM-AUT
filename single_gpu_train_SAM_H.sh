#!/bin/bash

CUDA_VISIBLE_DEVICES=3 torchrun --standalone main.py --task_name "LoRA_ViT_H_918_1e-3_1_ND_Global" --use_wandb --model_type "vit_h" --checkpoint './checkpoints/sam_vit_h_4b8939.pth' --do_5_fold_cross_validation --single_device --max_epochs 10 --batch_size 1 --base_lr 1e-3 --filter_empty --selected_blocks 1  --finetune_mode 1 --val_batch_size 4 --min_area 4.5 --dataset_path '/path/to/COCO_format_dataset/' --test_split 0.2 --val_epoch_duration 1 --if_encoder_lora_layer --encoder_lora_layer 0 1 30 31 --weight_decay 0.01 --lr_reducer_factor 0.9


