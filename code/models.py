"""Models for MRI task decoding and subject fingerprinting.""."""
import torch
from einops import rearrange
from torch import nn

# TODO(cyril) : see if EGNNA from Homework2 can be used here (if self-attention is not enough)


class MRIVisionTransformers(nn.Module):
    """MRI Vision Transformers model."""

    def __init__(
        self,
        output_size,
        input_size=400,
        num_heads=4,
        dropout=0.1,
        attention_dropout=0.1,
    ):
        """Initialize the model."""
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # self.transformers_embeddings = nn.Linear(input_size**2, output_size)
        self.self_attention = nn.MultiheadAttention(
            input_size, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.fingerprints = nn.Linear(input_size**2, output_size)
        self.task_decoder = nn.Linear(input_size**2, output_size)

    def forward_self_attention(self, x):
        """Forward pass of the self-attention layer."""
        x, attn_weights = self.self_attention(x, x, x)
        x = nn.Dropout(self.attention_dropout)(x)
        return x, attn_weights

    def forward(self, x):
        """Forward pass of the model."""
        # b, h, w = x.shape
        x, attn_weights = self.forward_self_attention(x)
        x = rearrange(x, "b h w -> b (h w)")
        x_si = self.fingerprints(x)
        x_td = self.task_decoder(x)
        return x_si, x_td, attn_weights


if __name__ == "__main__":
    """Test the model."""
    model = MRIVisionTransformers(output_size=512)
    print(model)
    x = torch.randn(1, 400, 400)
    y = model(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
