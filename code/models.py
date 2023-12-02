"""Models for MRI task decoding and subject fingerprinting.""."""
import logging
import math

import torch
from einops import rearrange
from torch import nn

# TODO(cyril) : see if EGNNA from Homework2 can be used here (if self-attention is not enough)
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)


class LinearLayer(nn.Module):
    """Linear layer with dropout and layer normalization."""

    def __init__(
        self,
        output_size_subjects,
        output_size_tasks=9,
        input_size=400,
        intermediate_size=None,
        dropout=0.1,
        layer_norm=True,
    ):
        """Initialize the layer.

        Args:
            output_size_subjects (int): number of subjects to classify.
            output_size_tasks (int): number of tasks to classify.
            input_size (int): size of the input matrix.
            intermediate_size (array): size of the intermediate linear layers output.
            dropout (float): dropout rate.
            layer_norm (boolean): layer norm or batch norm.
        """
        super().__init__()
        self.input_size = input_size
        self.dropout = dropout

        if layer_norm:
            self.norm_task = nn.LayerNorm(output_size_tasks)
            self.norm_subject = nn.LayerNorm(output_size_subjects)
        else:
            self.norm_task = nn.BatchNorm1d(output_size_tasks)
            self.norm_subject = nn.BatchNorm1d(output_size_subjects)

        # If multiple layers
        self.interm_layers_finger = nn.ModuleList()
        self.interm_layers_task = nn.ModuleList()
        self.intermediate_size_v = intermediate_size
        self.norms = nn.ModuleList()

        if intermediate_size is not None:
            for i, dim in enumerate(intermediate_size):
                if i == 0:
                    self.interm_layers_finger.append(
                        nn.Linear(input_size**2, dim)
                    )
                    self.interm_layers_task.append(
                        nn.Linear(input_size**2, dim)
                    )

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
                input_size**2, output_size_subjects
            )
            self.task_classifier = nn.Linear(
                input_size**2, output_size_tasks
            )

    def forward(self, x):
        """Forward pass of the layer."""
        x = rearrange(x, "b h w -> b (h w)")
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
                x_si = nn.Dropout(self.dropout)(x_si)

                x_td = norm(x_td)
                x_td = nn.Dropout(self.dropout)(x_td)

                i = 1

            # Classification layers
            x_si = self.fingerprints_classifier_i(x_si)
            x_td = self.task_classifier_i(x_td)

        else:
            # If no intermediate layers
            x_si = self.fingerprints_classifier(x)
            x_td = self.task_classifier(x)

        # return an attention weight empty
        return x_si, x_td, []


class LinearLayerShared(nn.Module):
    """Shared linear layer with dropout and layer normalization."""

    def __init__(
        self,
        output_size_subjects,
        output_size_tasks=9,
        input_size=400,
        intermediate_size=None,
        dropout=0.1,
        layer_norm=True,
    ):
        """Initialize the layer.

        Args:
            output_size_subjects (int): number of subjects to classify.
            output_size_tasks (int): number of tasks to classify.
            input_size (int): size of the input matrix.
            intermediate_size (array): size of the intermediate linear layers output.
            dropout (float): dropout rate.
            layer_norm (boolean): layer norm or batch norm.
        """
        super().__init__()
        self.input_size = input_size

        if layer_norm:
            self.norm_task = nn.LayerNorm(output_size_tasks)
            self.norm_subject = nn.LayerNorm(output_size_subjects)
        else:
            self.norm_task = nn.BatchNorm1d(output_size_tasks)
            self.norm_subject = nn.BatchNorm1d(output_size_subjects)

        # If multiple layers
        self.interm_layers = nn.ModuleList()
        self.intermediate_size_v = intermediate_size
        self.dropout = dropout
        self.norms = nn.ModuleList()

        if intermediate_size is not None:
            for i, dim in enumerate(intermediate_size):
                if i == 0:
                    self.interm_layers.append(nn.Linear(input_size**2, dim))
                else:
                    self.interm_layers.append(
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
                input_size**2, output_size_subjects
            )
            self.task_classifier = nn.Linear(
                input_size**2, output_size_tasks
            )

    def forward(self, x):
        """Forward pass of the shared layer."""
        x = rearrange(x, "b h w -> b (h w)")
        if self.intermediate_size_v is not None:
            for layer, norm in zip(self.interm_layers, self.norms):
                x = layer(x)
                x = norm(x)
                x = nn.Dropout(self.dropout)(x)

            # Classification layers
            x_si = self.fingerprints_classifier_i(x)
            x_td = self.task_classifier_i(x)
        else:
            # If no intermediate layers
            x_si = self.fingerprints_classifier(x)
            x_td = self.task_classifier(x)

        # return an attention weight empty
        return x_si, x_td, []


class DotProductAttention(nn.Module):
    """Dot Product Attention module."""

    def __init__(self, dropout_p=0.0):
        """Initialize the module.

        Args:
            dropout_p (float): dropout rate.
        """
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        """Forward pass of the module.

        Args:
            query (torch.Tensor): query matrix of size (batch, L, d_k).
            key (torch.Tensor): key matrix of size (batch, S, d_k).
            value (torch.Tensor): value matrix of size (batch, S, d_v).

        Returns:
            torch.Tensor: output matrix of size (batch, L, d_v).
        torch.Tensor: attention weights of size (batch, L, S).
        """
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout_p, train=True)

        return attn_weight @ value, attn_weight


class MRIAttention(nn.Module):
    """MRI Self-Attention model."""

    def __init__(
        self,
        output_size_subjects,
        output_size_tasks=8,
        input_size=400,
        num_heads=4,
        attention_dropout=0.1,
        dropout=0.1,
        intermediate_size=512,
        add_and_norm=True,
    ):
        """Initialize the model.

        Args:
            output_size_subjects (int): number of subjects to classify.
            output_size_tasks (int): number of tasks to classify.
            input_size (int): size of the input matrix.
            num_heads (int): number of heads in the multi-head attention.
            dropout (float): dropout rate for the intermediate layers.
            attention_dropout (float): dropout rate for the attention layers. (on attention weights)
            intermediate_size (int): size of the intermediate linear layers output.
            add_and_norm (bool): whether to add and normalize the attention output.
        """
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.multihead_attention = nn.MultiheadAttention(
            input_size, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.intermediate = nn.Linear(input_size**2, intermediate_size)
        self.intermediate_norm = nn.BatchNorm1d(intermediate_size)
        self.fingerprints = nn.Linear(intermediate_size, output_size_subjects)
        self.task_decoder = nn.Linear(intermediate_size, output_size_tasks)

        self.add_and_norm = add_and_norm
        self.norm = nn.BatchNorm1d(input_size**2)

        self.output_size_subjects = output_size_subjects
        self.output_size_tasks = output_size_tasks

        self._deeplift_mode = None

    def forward(self, x):
        """Forward pass of the model."""
        if self._deeplift_mode is not None:
            return self.forward_deeplift(x)
        ## Attention ##
        x_att, attn_weights = self.multihead_attention(x, x, x)
        if self.add_and_norm:
            x = x + x_att
            x = rearrange(x, "b h w -> b (h w)")
            x = self.norm(x)
        else:
            x = x_att
            x = rearrange(x, "b h w -> b (h w)")

        logger.debug(f"multihead_attention: {x.shape}")
        ## Intermediate layers ##
        x = self.intermediate(x)
        x = self.intermediate_norm(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(self.dropout)(x)

        ## Classification layers ##
        x_si = self.fingerprints(x)
        x_td = self.task_decoder(x)
        return x_si, x_td, attn_weights

    def forward_deeplift(self, x):
        """Forward pass of the model in deeplift mode.

        Uses the atention weights computed before to assess contributions separately for fingerprints and tasks.
        """
        print("deeplift mode")
        print("x in", x.shape)
        x, _ = self.multihead_attention(x, x, x)
        x = nn.Dropout(self.attention_dropout)(x)
        x = rearrange(x, "b h w -> b (h w)")
        x = self.intermediate(x)
        x = nn.ReLU()(x)
        x = self.intermediate_norm(x)
        x = nn.Dropout(self.dropout)(x)
        if self._deeplift_mode == "si":
            x = self.fingerprints(x)
        elif self._deeplift_mode == "td":
            x = self.task_decoder(x)
        else:
            raise ValueError("Invalid deeplift mode.")
        print("x out", x.shape)
        return x


class EGNNA(nn.Module):
    """Custom self-attention layer.

    Reference : Exploiting Edge Features for Graph Neural Networks, Gong and Cheng, 2019

    Args:
        in_features (int): number of input node features.
        out_features (int): number of output node features.
        activation (nn.Module or callable): activation function to apply. (optional)
        attention_activation (nn.Module or callable): activation function to apply to the attention scores. (default : LeakyReLU(0.2))
    """

    def __init__(
        self,
        in_features=400,
        out_features=400,
        activation=None,
        attention_activation=None,
        alpha=0.2,
    ):
        """Initialize the attention-based graph convolutional layer.

        Args:
            in_features (int): number of input node features.
            out_features (int): number of output node features.
            num_nodes (int): max number of nodes in the graph. (default: 28)
            activation (nn.Module or callable): activation function to apply. (optional)
            attention_activation (nn.Module or callable): activation function to apply to the attention scores. (default : LeakyReLU(0.2))
            alpha (float): alpha value for the LeakyReLU activation of attention score. (default: 0.2)
        """
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        # self.S = nn.Linear(2 * out_features, in_features, bias=False)
        self.S = nn.Linear(in_features, in_features, bias=False)
        self.activation = activation if activation is not None else lambda x: x
        self.att_activation = (
            attention_activation
            if attention_activation is not None
            else nn.LeakyReLU(alpha)
        )
        self.softmax = nn.Softmax(dim=2)
        self.instance_norm = nn.LayerNorm(out_features)

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight.weight)
        nn.init.kaiming_uniform_(self.a.weight)

    def forward(self, x):
        """Perform graph convolution operation with edge features.

        Uses edge features as "channels" for the attention scores.

        Args:
            x (Tensor): Input matrix of size (batch, in_features, in_features).

        Returns:
            Tensor: Output matrix of size (batch, out_features, out_features).
            Tensor: Attention scores of size (batch, out_features, out_features).
        """
        logger.debug(f"weight: {self.weight.weight.shape}")
        support = self.weight(x)
        logger.debug(f"support: {support.shape}")

        # cat_features = torch.cat([support, support], dim=2)
        # logger.debug(f"cat_features: {cat_features.shape}")
        att_score = self.att_activation(self.S(support))
        att_score = self.softmax(att_score)

        logger.debug(f"att_score: {att_score.shape}")

        alpha_channels = att_score * support
        logger.debug(f"alpha_channels: {alpha_channels.shape}")

        alpha_channels = self.instance_norm(alpha_channels)
        out = torch.bmm(alpha_channels, support)
        return self.activation(out), att_score


class MRICustomAttention(nn.Module):
    """MRI Self-Attention model."""

    def __init__(
        self,
        output_size_subjects,
        output_size_tasks=8,
        input_size=400,
        attention_dropout=0.1,
        intermediate_size=512,
        intermediate_dropout=0.1,
        add_and_norm=True,
    ):
        """Initialize the model.

        Args:
            output_size_subjects (int): number of subjects to classify.
            output_size_tasks (int): number of tasks to classify.
            input_size (int): size of the input matrix.
            attention_dropout (float): dropout rate for the attention layers.
            intermediate_size (int): size of the intermediate linear layers output.
            intermediate_dropout (float): dropout rate for the intermediate layers.
            add_and_norm (bool): whether to add and normalize the attention output.
        """
        super().__init__()
        self.input_size = input_size
        self.attention_dropout = attention_dropout
        self.intermediate_dropout = intermediate_dropout
        self.attention = EGNNA(input_size, input_size)
        self.intermediate = nn.Linear(input_size**2, intermediate_size)
        self.intermediate_norm = nn.BatchNorm1d(intermediate_size)
        self.fingerprints = nn.Linear(intermediate_size, output_size_subjects)
        self.task_decoder = nn.Linear(intermediate_size, output_size_tasks)

        self.add_and_norm = add_and_norm
        self.norm = nn.BatchNorm1d(input_size**2)

        self.output_size_subjects = output_size_subjects
        self.output_size_tasks = output_size_tasks

        self._deeplift_mode = None

    def forward(self, x):
        """Forward pass of the model."""
        if self._deeplift_mode is not None:
            return self.forward_deeplift(x)
        ## Attention ##
        x_att, attn_weights = self.attention(x)
        x_att = nn.Dropout(self.attention_dropout)(x_att)

        if self.add_and_norm:
            x = x + x_att
            x = rearrange(x, "b h w -> b (h w)")
            x = self.norm(x)
        else:
            x = x_att
            x = rearrange(x, "b h w -> b (h w)")

        x = self.intermediate(x)
        x = nn.ReLU()(x)
        x = self.intermediate_norm(x)
        x = nn.Dropout(self.intermediate_dropout)(x)
        ## Classification layers ##
        x_si = self.fingerprints(x)
        x_td = self.task_decoder(x)
        return x_si, x_td, attn_weights

    def forward_deeplift(self, x):
        """Forward pass of the model in deeplift mode.

        Uses the atention weights computed before to assess contributions separately for fingerprints and tasks.
        """
        print("deeplift mode")
        print("x in", x.shape)
        x = rearrange(x, "b h w -> b (h w)")
        x = self.intermediate(x)
        x = nn.ReLU()(x)
        x = self.intermediate_norm(x)
        x = nn.Dropout(self.intermediate_dropout)(x)
        if self._deeplift_mode == "si":
            x = self.fingerprints(x)
        elif self._deeplift_mode == "td":
            x = self.task_decoder(x)
        else:
            raise ValueError("Invalid deeplift mode.")
        print("x out", x.shape)
        return x


if __name__ == "__main__":
    """Test the model."""
    model = MRIAttention(output_size_subjects=10)
    print(model)
    x = torch.randn(1, 400, 400)
    y = model(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
