#Several SqueezeFormer components where copied/ adapted from https://github.com/upskyy/Squeezeformer/

import torch
from torch.nn import functional as F
from torch import nn
from typing import Tuple, Union, Optional
import typing
from torch import Tensor
import math
import numpy as np
import timm
import json
from transformers.models.speech_to_text import Speech2TextConfig, Speech2TextForConditionalGeneration
from transformers.models.speech_to_text.modeling_speech_to_text import shift_tokens_right, Speech2TextDecoder
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding
from timm.layers.norm_act import BatchNormAct2d


class Decoder(nn.Module):
    def __init__(self, decoder_config):
        super(Decoder, self).__init__()
        
        self.config = decoder_config
        self.decoder = Speech2TextDecoder(decoder_config) 
        self.lm_head = nn.Linear(decoder_config.d_model, decoder_config.vocab_size, bias=False)
        
        self.decoder_start_token_id = decoder_config.decoder_start_token_id
        self.decoder_pad_token_id = decoder_config.pad_token_id #used for early stopping
        self.decoder_end_token_id= decoder_config.eos_token_id
        
    def forward(self,x, labels=None, attention_mask = None, encoder_attention_mask = None):
        
        if labels is not None:
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
            
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                       encoder_hidden_states=x, 
                                       attention_mask = attention_mask,
                                       encoder_attention_mask = encoder_attention_mask)
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        return lm_logits
            
    def generate(self, x, max_new_tokens=33, encoder_attention_mask=None):

        decoder_input_ids = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.long).fill_(self.decoder_start_token_id)
        for i in range(max_new_tokens-1):  
            decoder_outputs = self.decoder(input_ids=decoder_input_ids,encoder_hidden_states=x, encoder_attention_mask=encoder_attention_mask)
            logits = self.lm_head(decoder_outputs.last_hidden_state)
            decoder_input_ids = torch.cat([decoder_input_ids,logits.argmax(2)[:,-1:]],dim=1)

            if torch.all((decoder_input_ids==self.decoder_end_token_id).sum(-1) > 0):
                break
                
        return decoder_input_ids


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
   
    def __init__(self, dim=512, expansion=4, dropout=0.1, use_glu=False):
        super().__init__()
        hidden = dim * expansion
        self.use_glu = use_glu
        if use_glu:
            self.fc1 = tf.keras.layers.Dense(hidden * 2)
        else:
            self.fc1 = tf.keras.layers.Dense(hidden)
        self.act = tf.keras.activations.swish
        self.fc2 = tf.keras.layers.Dense(dim)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        if self.use_glu:
            x1, gate = tf.split(self.fc1(x), num_or_size_splits=2, axis=-1)
            x = x1 * tf.nn.sigmoid(gate)
        else:
            x = self.act(self.fc1(x))
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: False
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class DepthwiseConv2d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 2
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 2-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: int = 2,
        padding: int = 0,
    ) -> None:
        super(DepthwiseConv2d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConvModule(nn.Module):
    """
    Convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by squeezeformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)

    # mask_pad = mask.bool().unsqueeze(1)
    def forward(self, x, mask_pad):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        Reference for masking : https://github.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/built-in/audio/Wenet_Conformer_for_Pytorch/wenet/transformer/convolution.py#L26
        """
        # mask batch padding
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        # torch.Size([4, 128, 384])
        x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
        mask_bn = mask_pad.view(-1)
        x_bn[mask_bn] = self.bn(x_bn[mask_bn])
        x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        '''
        x = self.bn(x)
        '''
        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x



def make_scale(encoder_dim):
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None,None,:])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None,None,:])
    return scale, bias

class SqueezeformerBlock(nn.Module):
    """
    SqueezeformerBlock is a simpler block structure similar to the standard Transformer block,
    where the MHA and convolution modules are each directly followed by a single feed forward module.

    Args:
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by squeezeformer block.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super(SqueezeformerBlock, self).__init__()
        
        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)
        
        '''
        self.mhsa = MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,)
        encoder_dim = 144
        num_attention_heads = 4
        attention_dropout_p = 0.1
        self.mhsa_whisper = WhisperAttention(embed_dim = encoder_dim,\
                                       num_heads = num_attention_heads,\
                                       dropout = attention_dropout_p,\
                                       is_decoder = False,\
                                       bias = True)
        '''       
        
        
       self.mhsa_masa = WindowDecomposedMaSA1D(
    dim=encoder_dim,
    num_heads=num_attention_heads,
    window_size=64,          # tune: 32/64/128 based on seq lengths
    gammas=torch.linspace(0.85, 0.95, steps=num_attention_heads),  # per-head decay
    qkv_bias=True,
    attn_drop=attention_dropout_p,
    proj_drop=attention_dropout_p,
    use_lce=True,
    lce_kernel=5,
)
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        
        self.ff_mhsa = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
        
        
        # Attention_mask = (bsz, self.num_heads, q_len, kv_seq_len)   
            
            
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)
        
        '''
        self.mhsa = self.encoder.blocks[0].mhsa
        self.ln_mhsa = self.encoder.blocks[0].ln_mhsa
        self.ln_ff_mhsa = self.encoder.blocks[0].ln_ff_mhsa
        self.ln_conv = self.encoder.blocks[0].ln_conv
        self.ln_ff_conv = self.encoder.blocks[0].ln_ff_conv
        self.ff_mhsa = self.encoder.blocks[0].ff_mhsa
        self.ln_mhsa = self.encoder.blocks[0].ln_mhsa
        self.conv = self.encoder.blocks[0].conv
        self.ff_conv = self.encoder.blocks[0].ff_conv
        '''

    def forward(self, x, cos, sin, mask):
        # --- inside SqueezeformerBlock.forward ---
        # inputs: x (B, T, C), cos, sin, mask (B, T)
        mask_pad = (mask).long().bool().unsqueeze(1)                 # (B, 1, T)
        mask_pad = ~(mask_pad.permute(0, 2, 1) * mask_pad)           # (B, T, T) for Llama (kept for compatibility)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape
        
        residual = x
        x = x * self.scale_mhsa.to(x.dtype) + self.bias_mhsa.to(x.dtype)
        
        # old LlamaAttention call:
        # x = residual + self.mhsa_llama(x, cos, sin, attention_mask = mask_pad.unsqueeze(1) )[0]
        
        # new MaSA call (uses 1D mask only):
        x = residual + self.mhsa_masa(x, attention_mask=mask.bool())  # (B, T, C)
        
        # Skip pad #1 (unchanged)
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x)
        residual = x
        x = x * self.scale_ff_mhsa.to(x.dtype) + self.bias_ff_mhsa.to(x.dtype)
        x = residual + self.ff_mhsa(x)
        x = self.ln_ff_mhsa(x)
        
        # Unskip pad #1 (unchanged)
        x_skip[mask_flat] = x[0].to(x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)
        
        # rest of block (conv + FF) unchanged ...

        residual = x
        # torch.Size([16, 384, 128])
        x = x * self.scale_conv.to(x.dtype) + self.bias_conv.to(x.dtype)
        x = residual + self.conv(x, mask_pad = mask.bool().unsqueeze(1))
        # Skip pad #2
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        
        x = self.ln_conv(x)
        
        
        residual = x
        x = x * self.scale_ff_conv.to(x.dtype) + self.bias_ff_conv.to(x.dtype)
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)
        
        # Unskip pad #2
        x_skip[mask_flat] = x[0].to(x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)  
        
        
        return x


class SqueezeformerEncoder(nn.Module):
    """
    Squeezeformer encoder first processes the input with a convolution subsampling layer and then
    with a number of squeezeformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_layers (int, optional): Number of squeezeformer blocks
        reduce_layer_index (int, optional): The layer index to reduce sequence length
        recover_layer_index (int, optional): The layer index to recover sequence length
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths
    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by squeezeformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        reduce_layer_index: int = 7,
        recover_layer_index: int = 15,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.recover_tensor = None

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                )
            )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor):
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """

        for idx, block in enumerate(self.blocks):
            x = block(x, cos, sin, mask)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self,
                 n_landmarks,out_dim, conv_ch = 3):
        super().__init__()   

        self.in_channels = in_channels = 32 * math.ceil(n_landmarks / 2)
        self.stem_linear = nn.Linear(in_channels,out_dim,bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)
        self.conv_stem = nn.Conv2d(conv_ch, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU,drop_layer=None)
        
    def forward(self, data, mask):


        xc = data.permute(0,3,1,2)
        xc = self.conv_stem(xc)
        xc = self.bn_conv(xc)
        xc = xc.permute(0,2,3,1)
        xc = xc.reshape(*data.shape[:2], -1)
        
        m = mask.to(torch.bool)  
        x = self.stem_linear(xc)
        
        # Batchnorm without pads
        bs,slen,nfeat = x.shape
        x = x.view(-1, nfeat)
        x_bn = x[mask.view(-1)==1].unsqueeze(0)
        x_bn = self.stem_bn(x_bn.permute(0,2,1)).permute(0,2,1)
        x[mask.view(-1)==1] = x_bn[0]
        x = x.view(bs,slen,nfeat)
        # Padding mask
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        
        return x


class LCE1D(nn.Module):
    """Depthwise + pointwise 1D conv on channels-last sequences via Conv1d."""
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.act = nn.GELU()
        nn.init.kaiming_normal_(self.dw.weight, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.pw.weight, a=math.sqrt(5))

    def forward(self, x_bt_c: torch.Tensor):  # (B, T, C)
        x = x_bt_c.transpose(1, 2)           # (B, C, T)
        x = self.pw(self.act(self.dw(x)))
        return x.transpose(1, 2)             # (B, T, C)

def _seq_window_partition(x: torch.Tensor, ws: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition (B, T, C) into non-overlapping windows -> (B*nw, ws, C), with right padding."""
    B, T, C = x.shape
    pad_t = (ws - T % ws) % ws
    if pad_t > 0:
        x = F.pad(x, (0, 0, 0, pad_t))  # pad along T
    nw = (T + pad_t) // ws
    xw = x.view(B, nw, ws, C).reshape(B * nw, ws, C)
    return xw, (T, pad_t)

def _seq_window_reverse(xw: torch.Tensor, ws: int, B: int, T: int, pad_t: int) -> torch.Tensor:
    """Reverse windows (B*nw, ws, C) to (B, T, C), cropping right padding."""
    C = xw.shape[-1]
    nw = xw.shape[0] // B
    x = xw.view(B, nw, ws, C).reshape(B, nw * ws, C)
    if pad_t > 0:
        x = x[:, :T, :]
    return x

def _make_1d_distance(ws: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(ws, device=device)
    return torch.abs(idx[:, None] - idx[None, :]).float()

class WindowDecomposedMaSA1D(nn.Module):
    """
    1D Windowed Decomposed Manhattan Self-Attention for sequences (B, T, C).
    - Per-head decay gamma^{|i-j|} inside each window.
    - Softmax over masked logits.
    - Optional LCE1D on V as residual: out += LCE(V).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 64,
        gammas: Union[float, int, torch.Tensor] = 0.9,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_lce: bool = True,
        lce_kernel: int = 5,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        if isinstance(gammas, (float, int)):
            g = torch.tensor([float(gammas)] * num_heads, dtype=torch.float32)
        else:
            g = torch.as_tensor(gammas, dtype=torch.float32)
            assert g.shape[0] == num_heads
        self.register_buffer("gammas", g)

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # distance buffer (created lazily per-device/size at forward)
        self.register_buffer("dist_ws", torch.empty(0), persistent=False)

        self.use_lce = use_lce
        if use_lce:
            self.lce = LCE1D(dim, kernel_size=lce_kernel)

    def _get_dist(self, ws: int, device: torch.device):
        if (self.dist_ws.numel() == 0) or (self.dist_ws.shape[0] != ws) or (self.dist_ws.device != device):
            self.dist_ws = _make_1d_distance(ws, device)
        return self.dist_ws

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, C)
        attention_mask: (B, T) bool, True for valid tokens, False for pad
        """
        B, T, C = x.shape
        ws = self.window_size
        device = x.device

        xw, (T_orig, pad_t) = _seq_window_partition(x, ws)  # (B*nw, ws, C)
        Bnw = xw.shape[0]

        qkv = self.qkv(xw)  # (Bnw, ws, 3C)
        qkv = qkv.view(Bnw, ws, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]  # (Bnw, heads, ws, hd)

        dist = self._get_dist(ws, device)             # (ws, ws)
        gammas = self.gammas.to(device).view(1, self.num_heads, 1, 1)  # (1, heads, 1, 1)
        DW = gammas ** dist                            # (1, heads, ws, ws)
        # expand to (Bnw, heads, ws, ws)
        DW = DW.expand(Bnw, self.num_heads, ws, ws)

        # logits
        attn_logits = torch.matmul(q, k.transpose(-2, -1))  # (Bnw, heads, ws, ws)
        attn_logits = attn_logits * DW

        # build per-window key mask (mask keys where padded)
        if attention_mask is not None:
            mask = attention_mask
            if pad_t > 0:
                mask = F.pad(mask, (0, pad_t), value=False)
            nw = mask.shape[1] // ws
            mk = mask.view(B, nw, ws).reshape(Bnw, ws)                    # (Bnw, ws)
            # expand to (Bnw, heads, ws, ws) to mask keys
            mk_keys = (~mk).unsqueeze(1).unsqueeze(2).expand(Bnw, self.num_heads, ws, ws)
            attn_logits = attn_logits.masked_fill(mk_keys, torch.finfo(attn_logits.dtype).min)

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # (Bnw, heads, ws, hd)

        out = out.transpose(1, 2).contiguous().view(Bnw, ws, C)  # merge heads

        # LCE on V (original), residual add
        if self.use_lce:
            v_merge = v.transpose(1, 2).contiguous().view(Bnw, ws, C)  # (Bnw, ws, C)
            out = out + self.lce(v_merge)

        out = self.proj(out)
        out = self.proj_drop(out)

        # zero out padded query positions (optional safety)
        if attention_mask is not None:
            mq = mask.view(B, nw, ws).reshape(Bnw, ws)  # (Bnw, ws)
            out = out * mq.unsqueeze(-1).to(out.dtype)

        y = _seq_window_reverse(out, ws, B, T_orig, pad_t)  # (B, T, C)
        return y




def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q = q.unsqueeze(1)
    k = k.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.squeeze(1)
    k_embed = k_embed.squeeze(1)
    return q_embed, k_embed



def get_lm_type(lm):
    if 'left_hand' in lm:
        t = 1
    elif 'right_hand' in lm:
        t = 2    
    elif 'face' in lm:
        t = 3  
    elif 'pose' in lm:
        t = 4  
    return t

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.max_phrase = cfg.max_phrase
        
        with open(cfg.data_folder + 'inference_args.json', "r") as f:
            columns = json.load(f)['selected_columns']
        
        self.xyz_landmarks = np.array(columns)
        landmarks = np.array([item[2:] for item in self.xyz_landmarks[:len(self.xyz_landmarks)//3]])
        self.landmark_types = np.array([get_lm_type(lm) for lm in landmarks])
        self.feature_extractor = FeatureExtractor(n_landmarks=cfg.n_landmarks,out_dim=cfg.encoder_config.encoder_dim)
        self.feature_extractor_lhand = FeatureExtractor(n_landmarks=(self.landmark_types==1).sum(),out_dim=cfg.encoder_config.encoder_dim//4)
        self.feature_extractor_rhand = FeatureExtractor(n_landmarks=(self.landmark_types==2).sum(),out_dim=cfg.encoder_config.encoder_dim//4)
        self.feature_extractor_face = FeatureExtractor(n_landmarks=(self.landmark_types==3).sum(),out_dim=cfg.encoder_config.encoder_dim//4)
        self.feature_extractor_pose = FeatureExtractor(n_landmarks=(self.landmark_types==4).sum(),out_dim=cfg.encoder_config.encoder_dim//4)
       
        rotary_emb = LlamaRotaryEmbedding(cfg.encoder_config.encoder_dim//cfg.encoder_config.num_attention_heads, max_position_embeddings=cfg.max_len)
        self.cos = torch.nn.parameter.Parameter(rotary_emb.cos_cached, requires_grad=False)#[:, :, :seq_len, ...]#.to(dtype=x.dtype)
        self.sin = torch.nn.parameter.Parameter(rotary_emb.sin_cached, requires_grad=False)#[:, :, :seq_len, ...]#.to(dtype=x.dtype)

        self.encoder = SqueezeformerEncoder(
                      input_dim=cfg.encoder_config.input_dim,
                      encoder_dim=cfg.encoder_config.encoder_dim,
                      num_layers=cfg.encoder_config.num_layers,
                      num_attention_heads= cfg.encoder_config.num_attention_heads,
                      feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
                      conv_expansion_factor= cfg.encoder_config.conv_expansion_factor,
                      input_dropout_p= cfg.encoder_config.input_dropout_p,
                      feed_forward_dropout_p= cfg.encoder_config.feed_forward_dropout_p,
                      attention_dropout_p= cfg.encoder_config.attention_dropout_p,
                      conv_dropout_p= cfg.encoder_config.conv_dropout_p,
                      conv_kernel_size= cfg.encoder_config.conv_kernel_size,
                     )
        
        self.decoder = Decoder(cfg.transformer_config)
        self.decoder2 = Decoder(cfg.transformer_config) 
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.ce_ignore_index, label_smoothing = cfg.label_smoothing)
        self.aux_loss_fn = nn.BCEWithLogitsLoss()
        self.aux_fc = nn.Linear(cfg.encoder_config.encoder_dim,1)
        self.aux_loss_weight = cfg.aux_loss_weight
        self.return_aux_logits = cfg.return_aux_logits
        self.bwd_loss_weight = cfg.bwd_loss_weight
        
        self.val_mode = cfg.val_mode
        self.decoder_mask_aug = cfg.decoder_mask_aug
        print('n_params:',count_parameters(self))

    def forward(self, batch, debug = False):

        x = batch['input'] # bs, seq_len, n_landmarks, 3
        xp = batch['input'].clone()
        labels = batch['token_ids']
        mask = batch['input_mask'].long()
        label_mask = batch['attention_mask']

        
        # normalise the parts
        for ii in range(4):
            pidx = self.landmark_types==1+ii            
            x_mean = xp[:,:,pidx].mean(2).unsqueeze(2)
            x_std = xp[:,:,pidx].std(2, unbiased=False).unsqueeze(2)
            xp[:,:,pidx] = (xp[:,:,pidx] - x_mean) / x_std
        xp[torch.isnan(xp)] = 0.
        xp[torch.isinf(xp)] = 0.
        dropped_mask = x[:,:,:,:2].sum(-1)==0.
        xp[dropped_mask] = 0.

        x_lhand = self.feature_extractor_lhand(xp[:,:,self.landmark_types==1], mask)
        x_rhand = self.feature_extractor_rhand(xp[:,:,self.landmark_types==2], mask)
        x_face = self.feature_extractor_face(xp[:,:,self.landmark_types==3], mask)
        x_pose = self.feature_extractor_pose(xp[:,:,self.landmark_types==4], mask)
        
        x1 = torch.cat([x_lhand,x_rhand,x_face,x_pose],dim=-1)
        x = self.feature_extractor(x, mask)
        x = x + x1
        x = self.encoder(x, self.cos, self.sin, mask)
        aux_logits = self.aux_fc(x[:,0])
        if debug:
            return x
        
        decoder_labels = labels.clone()
        if self.training:
            m = torch.rand(labels.shape) < self.decoder_mask_aug
            decoder_labels[m] = 62

        logits = self.decoder(x,
                                  labels=decoder_labels, 
                                  encoder_attention_mask=mask.long(),
    #                               attention_mask=label_mask,
                                 )
        
        x_bwd = torch.flip(x, [1])
        mask_bwd = torch.flip(mask, [1])
        lbl_bwd = [dlbl[msk==1] for dlbl,msk in zip(labels.clone(), label_mask)]
        lbl_bwd = [torch.cat((torch.flip(i[:-1], [0]), i[-1:])) for  i in lbl_bwd]
        lbl_bwd = torch.nn.utils.rnn.pad_sequence(lbl_bwd, batch_first=True)
        decoder_lbl_bwd = lbl_bwd.clone()
        if self.training:
            m = torch.rand(lbl_bwd.shape) < self.decoder_mask_aug
            decoder_lbl_bwd[m] = 62
        logits_bwd = self.decoder2(x_bwd,
                                  labels=decoder_lbl_bwd, 
                                  encoder_attention_mask=mask_bwd.long(),
    #                               attention_mask=label_mask,
                                 )
        
        loss_ce = self.loss_fn(logits.view(-1, self.decoder.config.vocab_size), labels.view(-1))   
        loss_ce_bwd = self.loss_fn(logits_bwd.view(-1, self.decoder.config.vocab_size), lbl_bwd.view(-1))   
        loss_aux = self.aux_loss_fn(aux_logits,batch['score'].clamp(0,1)[:,None])
        loss = (1- self.aux_loss_weight) * (loss_ce * (1-self.bwd_loss_weight) + self.bwd_loss_weight * loss_ce_bwd) \
                + self.aux_loss_weight * loss_aux
        
        output = {'loss':loss}
        output['loss_aux'] = loss_aux
        output['loss_ce'] = loss_ce    
        output['loss_ce_bwd'] = loss_ce_bwd     
        
        if not self.training:
            generated_ids_padded = torch.ones((x.shape[0],self.max_phrase), dtype=torch.long, device=x.device) * 59
            
            if self.val_mode == 'padded':
                generated_ids = self.decoder.generate(x,max_new_tokens=self.max_phrase + 1, encoder_attention_mask=mask.long())
                    
            elif self.val_mode == 'cutted':
                generated_ids = torch.ones((x.shape[0],self.max_phrase+1), dtype=torch.long, device=x.device) * 59
                mask_lens = mask.sum(1)
                for lidx in mask_lens.unique():
                    liddx = lidx == mask_lens
                    preds = self.decoder.generate(x[liddx, :lidx],max_new_tokens=self.max_phrase + 1)
                    generated_ids[liddx, :preds.shape[1]] = preds
                    
            cutoffs = (generated_ids==self.decoder.decoder_end_token_id).float().argmax(1).clamp(0,self.max_phrase)
            for i, c in enumerate(cutoffs):
                generated_ids_padded[i,:c] = generated_ids[i,:c]
            output['generated_ids'] = generated_ids_padded
            output['seq_len'] = batch['seq_len']    
            if self.return_aux_logits:
                output['aux_logits'] = aux_logits
        return output
