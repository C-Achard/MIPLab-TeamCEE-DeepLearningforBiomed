"""Models for MRI task decoding and subject fingerprinting.""."""
import logging

import torch
from einops import rearrange
from torch import nn

# TODO(cyril) : see if EGNNA from Homework2 can be used here (if self-attention is not enough)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


class MRIAttention(nn.Module):
    """MRI Vision Transformers model."""

    def __init__(
        self,
        output_size_subjects,
        output_size_tasks=8,
        input_size=400,
        intermediate_size=512,
        num_heads=4,
        dropout=0.1,
        attention_dropout=0.1,
    ):
        """Initialize the model.

        Args:
            output_size_subjects (int): number of subjects to classify.
            output_size_tasks (int): number of tasks to classify.
            input_size (int): size of the input matrix.
            intermediate_size (int): size of the intermediate linear layers output.
            num_heads (int): number of heads in the multi-head attention.
            dropout (float): dropout rate.
            attention_dropout (float): dropout rate for the attention layers.
        """
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        self.self_attention = nn.MultiheadAttention(
            input_size, num_heads, dropout=attention_dropout, batch_first=True
        )

        self.fingerprints_classifier = nn.Linear(
            input_size**2, output_size_subjects
        )
        self.task_classifier = nn.Linear(input_size**2, output_size_tasks)

    def forward(self, x):
        """Forward pass of the model."""
        ## Attention ##
        x, attn_weights = self.self_attention(x, x, x)
        x = nn.Dropout(self.dropout)(x)
        ## Intermediate linear layers ##
        x = rearrange(x, "b h w -> b (h w)")
        x_si = self.fingerprints_classifier(x)
        x_td = self.task_classifier(x)
        return x_si, x_td, attn_weights


if __name__ == "__main__":
    """Test the model."""
    model = MRIAttention(output_size_subjects=10)
    print(model)
    x = torch.randn(1, 400, 400)
    y = model(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
