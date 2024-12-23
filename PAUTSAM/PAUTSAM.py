import torch

class PAUTSAM(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        selected_blocks,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.selected_blocks = selected_blocks
        
        if self.selected_blocks == 0:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

            for param in self.image_encoder.parameters():
                param.requires_grad = False

        elif self.selected_blocks == 1:

            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

    def forward(self, image, box):

        if self.selected_blocks == 0:
            with torch.no_grad():
                image_embedding = self.image_encoder(image)

        elif self.selected_blocks == 1:
            image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        # do not compute gradients for prompt encoder
        with torch.no_grad():

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )
        mask_pred, iou_pred = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        return mask_pred,iou_pred

    def get_total_parameters(self):
        print("Number of total parameters: ",sum(p.numel() for p in self.parameters()),)  

    def get_total_trainable_parameters(self):
        print("Number of trainable parameters: ",sum(p.numel() for p in self.parameters() if p.requires_grad),)       
