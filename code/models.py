"""Models for MRI task decoding and subject fingerprinting.""."""
import logging

import torch
from einops import rearrange
from torch import nn

# TODO(cyril) : see if EGNNA from Homework2 can be used here (if self-attention is not enough)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


class LinearLayer(nn.Module):
    """Linear layer with dropout and layer normalization."""

    def __init__(
        self,
        output_size_subjects,
        output_size_tasks=8,
        input_size=400,
        # intermediate_size=[512],
        intermediate_size=None,
        dropout=0.1,
    ):
        """Initialize the layer.

        Args:
            output_size_subjects (int): number of subjects to classify.
            output_size_tasks (int): number of tasks to classify.
            input_size (int): size of the input matrix.
            intermediate_size (array): size of the intermediate linear layers output.
            dropout (float): dropout rate.
        """
        super().__init__()
        self.interm_layers_finger = nn.ModuleList()
        self.interm_layers_task = nn.ModuleList()
        self.input_size = input_size
        self.intermediate_size_v = intermediate_size
        self.dropout = dropout
        self.norms = nn.ModuleList()
        self.norm_task = nn.LayerNorm(output_size_tasks)
        self.norm_subject = nn.LayerNorm(output_size_subjects)

        if intermediate_size is not None:
            for i, dim in enumerate(intermediate_size):
                if i == 0:
                    self.interm_layers_finger.append(
                        nn.Linear(input_size, dim)
                    )
                    self.interm_layers_task.append(nn.Linear(input_size, dim))
                else:
                    self.interm_layers_finger.append(
                        nn.Linear(intermediate_size[i - 1], dim)
                    )
                    self.interm_layers_task.append(
                        nn.Linear(intermediate_size[i - 1], dim)
                    )

            self.fingerprints_classifier_i = nn.Linear(
                intermediate_size[-1], output_size_subjects
            )
            self.task_classifier_i = nn.Linear(
                intermediate_size[-1], output_size_tasks
            )

            for dim in intermediate_size:
                self.norms.append(nn.LayerNorm(dim))
        else:
            self.fingerprints_classifier = nn.Linear(
                input_size, output_size_subjects
            )
            self.task_classifier = nn.Linear(input_size, output_size_tasks)

    def forward(self, x):
        """Forward pass of the layer."""
        if self.intermediate_size_v is not None:
            i = 0
            for layer_task, layer_finger, norm in zip(
                self.interm_layers_task, self.interm_layers_finger, self.norms
            ):
                if i == 0:
                    x_si = layer_finger(x)
                    x_td = layer_task(x)
                else:
                    x_si = layer_finger(x_si)
                    x_td = layer_task(x_td)

                x_si = norm(x_si)
                x_si = nn.Dropout(p=self.dropout)(x_si)

                x_td = norm(x_td)
                x_td = nn.Dropout(p=self.dropout)(x_td)

                i = 1

            # Classification layers
            x_si = self.fingerprints_classifier_i(x_si)
            x_td = self.task_classifier_i(x_td)

        else:
            x_si = self.fingerprints_classifier(x)
            x_td = self.task_classifier(x)

        x_si = self.norm_subject(x_si)
        x_si = nn.Dropout(p=self.dropout)(x_si)

        x_td = self.norm_task(x_td)
        x_td = nn.Dropout(p=self.dropout)(x_td)

        # return an attention weight empty
        return x_si, x_td, []


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
