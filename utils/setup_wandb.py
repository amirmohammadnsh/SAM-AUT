import wandb
import os

def init_wandb(args):
    wandb.init(project=args.task_name, name=args.run_name,
                        config={
                            "model_type":args.model_cfg,
                            "n_gpu": int(os.environ["LOCAL_WORLD_SIZE"]),
                            "gpu_id(s)": os.environ["CUDA_VISIBLE_DEVICES"],
                            "backbone": args.backbone,
                            "max_epochs": args.max_epochs,
                            "filter_empty":args.filter_empty,
                            "selected_blocks": "MskDec" if args.selected_blocks == 0  else "MskDec_ImgEnc" if args.selected_blocks == 1 else None,
                            "finetune_mode": "Default" if args.finetune_mode == 0 else "LoRA" if args.finetune_mode == 1 else None,
                            "do_decoder_lora_layer": args.if_decoder_lora_layer,
                            "do_encoder_lora_layer": args.if_encoder_lora_layer,
                            "encoder_lora_layer": "null" if args.encoder_lora_layer == 0 else args.encoder_lora_layer,
                            "lora_rank": args.lora_rank if args.finetune_mode == 1 else None,
                            "batch_size": args.batch_size,
                            "val_batch_szie": args.val_batch_size,
                            "val_epoch_duration": args.val_epoch_duration,
                            "do_morphological_operations": args.do_filtering,
                            "min_area": args.min_area,
                            "kernel_size": args.kernel_size if args.do_filtering else None,
                            "dilate_iter": args.dilate_iter if args.do_filtering else None,
                            "iou_thresh": args.iou_thresh,
                            "do_warmup": args.do_warmup,
                            "warmup_period": args.warmup_period if args.do_warmup else 0,
                            "base_lr": args.base_lr,
                            "weight_decay": args.weight_decay,
                            "lr_reducer_factor":args.lr_reducer_factor,
                            "do_ce_loss": args.do_ce_loss,
                            "ce_loss_weight": args.ce_loss_weight,
                            "do_iou_loss":args.do_iou_loss,
                            "iou_loss_weight":args.iou_loss_weight,
                            "train_data_portion":args.train_data_portion,
                            "test_size":args.test_split,
                            "use_bins":args.use_binned_stratify

                        })
