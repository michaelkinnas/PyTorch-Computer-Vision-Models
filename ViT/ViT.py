import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim, image_size):
        super().__init__()

        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, padding=0, stride=patch_size)

        # Class token
        self.cls = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)

        # Positional embedding
        self.num_patches = image_size**2 // patch_size**2
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))

        # Only flatten the feature map dimensions
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
    
    def forward(self, x): 
        x = self.patcher(x)
        x = self.flatten(x)

        # Prepend cls token
        class_token = self.cls.expand(x.shape[0], -1, -1) # Expand the cls token across the batch size (needs the same class for all patch embeddings belonging to the same image)  
        x = x.permute(0, 2, 1)        
        x = torch.cat((class_token, x), dim=1)
        x = self.positional_embedding + x
        return x
    

class ViT(nn.Module):
    def __init__(self, 
                 img_size=224,
                 num_channels=3,
                 patch_size=16,
                 embedding_dim=768,
                 dropout=0.1,
                 mlp_size=3072,
                 n_heads=12, 
                 n_transformer_layers=12,
                 num_classes=1000):
        
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.patch_embedding = PatchEmbedding(in_channels=num_channels, 
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim,
                                              image_size=img_size)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, 
                                                        nhead=n_heads, 
                                                        dim_feedforward=mlp_size, 
                                                        dropout=dropout, 
                                                        activation="gelu", 
                                                        batch_first=True, 
                                                        norm_first=True)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_transformer_layers)

        self.mlp_head = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim),
                                      nn.Linear(in_features=embedding_dim,
                                                out_features=num_classes))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):

        # Create patch embeddings
        x = self.patch_embedding(x)

        # Dropout
        x = self.dropout(x)

        # Pass through transformer
        x = self.transformer_encoder(x)

        #Pass 0th index of x (the class token) throught the feed forward network
        x = self.mlp_head(x[:, 0])

        return x

        
