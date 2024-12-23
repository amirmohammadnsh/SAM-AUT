import torch

class PAUTSAM2(torch.nn.Module):
    def __init__(
        self,
        model,
        selected_blocks
    ):
        super().__init__()
        self.sam2_model = model
        self.selected_blocks = selected_blocks
        if self.selected_blocks == 0:
            for param in self.sam2_model.sam_prompt_encoder.parameters():
                param.requires_grad = False

            for param in self.sam2_model.image_encoder.parameters():
                param.requires_grad = False

        elif self.selected_blocks == 1:

            for param in self.sam2_model.sam_prompt_encoder.parameters():
                param.requires_grad = False

    def forward(self, image, box):

        if self.selected_blocks == 0:
            with torch.no_grad():
                _features = self._image_encoder(image)
                img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]

        elif self.selected_blocks == 1:
            _features = self._image_encoder(image)  # (B, 256, 64, 64)
            img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]

        # do not compute gradients for prompt encoder
        with torch.no_grad():

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )
        mask_pred, iou_pred,_,_ = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
            repeat_image = False,
            high_res_features=high_res_features,
        )

        return mask_pred,iou_pred
    
    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])][::-1]
        
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features
    
    def get_total_parameters(self):
        print("Number of total parameters: ",sum(p.numel() for p in self.parameters()),)  

    def get_total_trainable_parameters(self):
        print("Number of trainable parameters: ",sum(p.numel() for p in self.parameters() if p.requires_grad),)       
