#!/bin/bash
CUDA_VISIBLE_DEVICES=2 torchrun --standalone main.py --task_name "LoRA_ViT_L_918_1e-3_1_ND_Not_Global" --model_type "vit_l" --checkpoint './checkpoints/sam_vit_l_0b3195.pth' --use_wandb --do_5_fold_cross_validation --single_device --max_epochs 15 --batch_size 2 --base_lr 1e-3 --filter_empty --selected_blocks 1  --finetune_mode 1 --val_batch_size 8 --min_area 4.5 --dataset_path '/path/to/COCO format dataset/' --test_split 0.2 --val_epoch_duration 1 --if_encoder_lora_layer --encoder_lora_layer 0 1 22 23 --weight_decay 0.01 --lr_reducer_factor 0.9 --train_data_portion 1.0


