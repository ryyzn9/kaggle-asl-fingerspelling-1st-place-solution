#Several tensorflow SqueezeFormer components where copied/ adapted from https://github.com/kssteven418/Squeezeformer

# tensorflow_squeezeformer_masa.py
# TensorFlow/Keras rewrite of Squeezeformer with WindowDecomposedMaSA1D attention

import json
import math
import numpy as np
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ----------------- Utilities -----------------
def count_parameters(model: keras.Model):
    return np.sum([np.prod(v.shape) for v in model.trainable_variables])


def make_scale(encoder_dim: int):
    scale = tf.Variable(tf.ones((1, 1, encoder_dim), dtype=tf.float32), trainable=True, name="scale")
    bias = tf.Variable(tf.zeros((1, 1, encoder_dim), dtype=tf.float32), trainable=True, name="bias")
    return scale, bias


# ----------------- Activations & small layers -----------------
class Swish(layers.Layer):
    def call(self, x):
        return x * tf.nn.sigmoid(x)


class GLU(layers.Layer):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def call(self, x):
        a, b = tf.split(x, 2, axis=self.axis)
        return a * tf.nn.sigmoid(b)


# ----------------- Feed Forward -----------------
class FeedForwardModule(layers.Layer):
    def __init__(self, encoder_dim=512, expansion_factor=4, dropout_p=0.1, use_glu=False):
        super().__init__()
        hidden = encoder_dim * expansion_factor
        in_features = hidden * 2 if use_glu else hidden
        self.use_glu = use_glu
        self.fc1 = layers.Dense(in_features)
        self.act = layers.Activation("swish")  # or layers.Activation(tf.nn.silu)
        self.fc2 = layers.Dense(hidden if use_glu else hidden)  # intermediate
        self.proj = layers.Dense(encoder_dim)
        self.do = layers.Dropout(dropout_p)

    def call(self, x, training=False):
        if self.use_glu:
            x = self.fc1(x)
            a, gate = tf.split(x, 2, axis=-1)
            x = a * tf.nn.sigmoid(gate)
        else:
            x = self.act(self.fc1(x))
        x = self.do(x, training=training)
        x = self.fc2(x)
        x = self.do(x, training=training)
        x = self.proj(x)
        x = self.do(x, training=training)
        return x


# ----------------- Masked BatchNorm helper -----------------
class MaskedBatchNorm1D(layers.Layer):
    """
    Apply BatchNormalization only on valid rows of a 2D tensor (N, C) selected by mask (N,).
    Scatters normalized values back into the original tensor shape.
    """
    def __init__(self, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.bn = layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    def call(self, x_flat_nc, mask_flat, training=False):
        idx = tf.where(mask_flat)  # (M, 1)
        gathered = tf.gather(x_flat_nc, idx[:, 0])
        if tf.shape(gathered)[0] == 0:
            return x_flat_nc
        normed = self.bn(gathered, training=training)
        x_out = tf.tensor_scatter_nd_update(x_flat_nc, tf.expand_dims(idx[:, 0], -1), normed)
        return x_out


# ----------------- Convolution blocks -----------------
class DepthwiseConv1D(layers.Layer):
    def __init__(self, channels, kernel_size, stride=1, padding="same"):
        super().__init__()
        self.dw = layers.DepthwiseConv1D(kernel_size, strides=stride, padding=padding, depth_multiplier=1, use_bias=False)

    def call(self, x):
        return self.dw(x)


class PointwiseConv1D(layers.Layer):
    def __init__(self, out_channels, stride=1, padding="valid", use_bias=True):
        super().__init__()
        self.pw = layers.Conv1D(out_channels, kernel_size=1, strides=stride, padding=padding, use_bias=use_bias)

    def call(self, x):
        return self.pw(x)


class ConvModule(layers.Layer):
    """
    Keras port of the Squeezeformer ConvModule with mask-aware BatchNorm as in the PyTorch code.
    Input/Output: (B, T, C)
    """
    def __init__(self, in_channels: int, kernel_size: int = 31, expansion_factor: int = 2, dropout_p: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        assert expansion_factor == 2

        self.pw1 = PointwiseConv1D(in_channels * expansion_factor)
        self.glu = GLU(axis=-1)
        self.dw = DepthwiseConv1D(in_channels, kernel_size=kernel_size, stride=1, padding="same")
        self.bn = MaskedBatchNorm1D(momentum=0.9, epsilon=1e-5)
        self.act = Swish()
        self.pw2 = PointwiseConv1D(in_channels)
        self.do = layers.Dropout(dropout_p)

    def call(self, x_bt_c, mask_pad_b1t, training=False):
        # x: (B, T, C) -> conv path uses channels-last consistently
        B = tf.shape(x_bt_c)[0]
        T = tf.shape(x_bt_c)[1]
        C = tf.shape(x_bt_c)[2]

        # mask_pad_b1t: (B, 1, T) True for valid
        x = self.pw1(x_bt_c)
        x = self.glu(x)
        x = self.dw(x)
        # Mask-aware BN: flatten time, select valid, BN, scatter back
        x_bn = tf.reshape(x, (-1, tf.shape(x)[-1]))  # (B*T, C)
        mask_flat = tf.reshape(mask_pad_b1t, (-1,))  # (B*T,)
        x_bn = self.bn(x_bn, mask_flat, training=training)
        x = tf.reshape(x_bn, (B, T, -1))
        x = self.act(x)
        x = self.pw2(x)
        x = self.do(x, training=training)
        # zero-out pads
        x = tf.where(tf.expand_dims(mask_pad_b1t[:, 0, :], -1), x, tf.zeros_like(x))
        return x


# ----------------- LCE 1D -----------------
class LCE1D(layers.Layer):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        pad = "same"
        self.dw = layers.DepthwiseConv1D(kernel_size, padding=pad, depth_multiplier=1, use_bias=False)
        self.pw = layers.Conv1D(channels, kernel_size=1, use_bias=True)
        self.act = layers.Activation("gelu")

    def call(self, x_bt_c):
        x = self.dw(x_bt_c)
        x = self.act(x)
        x = self.pw(x)
        return x


# ----------------- 1D Window Decomposed MaSA -----------------
def _seq_window_partition_tf(x_bt_c, ws: int):
    # x: (B, T, C) -> (B*nw, ws, C), returns partitioned and metadata
    B = tf.shape(x_bt_c)[0]
    T = tf.shape(x_bt_c)[1]
    C = tf.shape(x_bt_c)[2]
    pad_t = tf.math.floormod(-T, ws)
    x_padded = tf.pad(x_bt_c, paddings=[[0, 0], [0, pad_t], [0, 0]])
    nw = (T + pad_t) // ws
    x_reshaped = tf.reshape(x_padded, (B, nw, ws, C))
    xw = tf.reshape(x_reshaped, (B * nw, ws, C))
    return xw, (T, pad_t, nw, B, C)


def _seq_window_reverse_tf(xw, ws: int, meta):
    T, pad_t, nw, B, C = meta
    x = tf.reshape(xw, (B, nw, ws, C))
    x = tf.reshape(x, (B, nw * ws, C))
    if pad_t > 0:
        x = x[:, :T, :]
    return x


class WindowDecomposedMaSA1D(layers.Layer):
    """
    1D windowed Manhattan Self-Attention with per-head decay gamma^{|i-j|} and optional LCE(V).
    Inputs:
      x: (B, T, C)
      attention_mask: (B, T) boolean, True for valid tokens
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 64,
        gammas: Union[float, np.ndarray] = 0.9,
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
        self.window_size = window_size
        # per-head gammas
        if isinstance(gammas, float) or isinstance(gammas, int):
            g = np.array([float(gammas)] * num_heads, dtype=np.float32)
        else:
            g = np.array(gammas, dtype=np.float32)
            assert g.shape[0] == num_heads
        self.gammas = tf.constant(g, dtype=tf.float32)  # (heads,)
        # layers
        self.qkv = layers.Dense(3 * dim, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)
        self.use_lce = use_lce
        if use_lce:
            self.lce = LCE1D(dim, kernel_size=lce_kernel)

    def call(self, x_bt_c, attention_mask=None, training=False):
        B = tf.shape(x_bt_c)[0]
        T = tf.shape(x_bt_c)[1]
        C = tf.shape(x_bt_c)[2]
        ws = self.window_size

        xw, meta = _seq_window_partition_tf(x_bt_c, ws)  # (Bnw, ws, C)
        Bnw = tf.shape(xw)[0]

        # qkv
        qkv = self.qkv(xw)  # (Bnw, ws, 3C)
        qkv = tf.reshape(qkv, (Bnw, ws, 3, self.num_heads, self.head_dim))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q / tf.math.sqrt(tf.cast(self.head_dim, q.dtype))  # scale

        # decay matrix DW: (1, heads, ws, ws)
        idx = tf.cast(tf.range(ws), tf.float32)
        dist = tf.abs(tf.expand_dims(idx, 1) - tf.expand_dims(idx, 0))  # (ws, ws)
        gam = tf.reshape(self.gammas, (1, self.num_heads, 1, 1))
        DW = tf.pow(gam, dist)  # broadcast to (1, heads, ws, ws)
        DW = tf.repeat(DW, repeats=Bnw, axis=0)  # (Bnw, heads, ws, ws)

        # logits and mask
        attn_logits = tf.matmul(q, k, transpose_b=True)  # (Bnw, heads, ws, ws)
        attn_logits = attn_logits * DW

        if attention_mask is not None:
            # pad to multiple of ws
            pad_t = tf.math.floormod(-T, ws)
            mask = attention_mask
            if pad_t > 0:
                mask = tf.pad(mask, paddings=[[0, 0], [0, pad_t]], constant_values=False)
            nw = tf.shape(mask)[1] // ws
            mk = tf.reshape(mask, (B, nw, ws))
            mk = tf.reshape(mk, (Bnw, ws))  # (Bnw, ws), True for valid
            # mask invalid keys (False) with large negative
            mk_keys = tf.logical_not(tf.expand_dims(tf.expand_dims(mk, 1), 2))  # (Bnw, 1, 1, ws)
            mk_keys = tf.broadcast_to(mk_keys, tf.shape(attn_logits))
            neg_inf = tf.constant(-1e9, dtype=attn_logits.dtype)
            attn_logits = tf.where(mk_keys, neg_inf, attn_logits)

        attn = tf.nn.softmax(attn_logits, axis=-1)
        attn = self.attn_drop(attn, training=training)
        out = tf.matmul(attn, v)  # (Bnw, heads, ws, head_dim)
        out = tf.transpose(out, perm=[0, 2, 1, 3])  # (Bnw, ws, heads, hd)
        out = tf.reshape(out, (Bnw, ws, self.dim))  # merge heads

        if self.use_lce:
            v_merge = tf.transpose(v, perm=[0, 2, 1, 3])  # (Bnw, ws, heads, hd)
            v_merge = tf.reshape(v_merge, (Bnw, ws, self.dim))
            out = out + self.lce(v_merge)

        out = self.proj(out)
        out = self.proj_drop(out, training=training)

        # zero-out padded queries (optional safety)
        if attention_mask is not None:
            mq = tf.reshape(tf.reshape(mask, (B, nw, ws)), (Bnw, ws))
            out = out * tf.cast(tf.expand_dims(mq, -1), out.dtype)

        y = _seq_window_reverse_tf(out, ws, meta)  # (B, T, C)
        return y


# ----------------- Feature extractor -----------------
class FeatureExtractor(layers.Layer):
    """
    Matches the PyTorch stem: Conv2D on (B, T, L, 3), BN+SiLU, flatten landmarks, then Linear to out_dim,
    followed by masked BN over valid time steps and zeroing padded positions.
    """
    def __init__(self, n_landmarks, out_dim, conv_ch=3):
        super().__init__()
        in_channels = 32 * math.ceil(n_landmarks / 2)
        self.conv_stem = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 2), padding="same", use_bias=False)
        self.bn_conv = layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.act_conv = layers.Activation(tf.nn.silu)
        self.stem_linear = layers.Dense(out_dim, use_bias=False)
        self.masked_bn = MaskedBatchNorm1D(momentum=0.95, epsilon=1e-5)

    def call(self, data_btlc, mask_bt, training=False):
        # data: (B, T, L, 3), mask: (B, T) True valid
        x = self.conv_stem(data_btlc)                  # (B, T, L/2, 32)
        x = self.bn_conv(x, training=training)
        x = self.act_conv(x)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        Lp = tf.shape(x)[2]
        Cc = tf.shape(x)[3]
        x = tf.reshape(x, (B, T, Lp * Cc))             # (B, T, in_channels)
        x = self.stem_linear(x)                        # (B, T, out_dim)

        # masked BN over time positions
        x_flat = tf.reshape(x, (-1, tf.shape(x)[-1]))  # (B*T, C)
        mask_flat = tf.reshape(mask_bt, (-1,))
        x_flat = self.masked_bn(x_flat, mask_flat, training=training)
        x = tf.reshape(x_flat, (B, T, tf.shape(x)[-1]))
        # zero padded positions
        x = tf.where(tf.expand_dims(mask_bt, -1), x, tf.zeros_like(x))
        return x


# ----------------- Squeezeformer Block -----------------
class SqueezeformerBlock(layers.Layer):
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
        masa_window_size: int = 64,
        gammas_low_high: Tuple[float, float] = (0.85, 0.95),
    ):
        super().__init__()
        # learnable scales/biases
        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)

        # attention
        gammas = np.linspace(gammas_low_high[0], gammas_low_high[1], num_attention_heads).astype(np.float32)
        self.mhsa_masa = WindowDecomposedMaSA1D(
            dim=encoder_dim,
            num_heads=num_attention_heads,
            window_size=masa_window_size,
            gammas=gammas,
            qkv_bias=True,
            attn_drop=attention_dropout_p,
            proj_drop=attention_dropout_p,
            use_lce=True,
            lce_kernel=5,
        )
        self.ln_mhsa = layers.LayerNormalization(epsilon=1e-5)

        # FF after attention
        self.ff_mhsa = FeedForwardModule(encoder_dim=encoder_dim, expansion_factor=feed_forward_expansion_factor, dropout_p=feed_forward_dropout_p)
        self.ln_ff_mhsa = layers.LayerNormalization(epsilon=1e-5)

        # Conv module
        self.conv = ConvModule(in_channels=encoder_dim, kernel_size=conv_kernel_size, expansion_factor=conv_expansion_factor, dropout_p=conv_dropout_p)
        self.ln_conv = layers.LayerNormalization(epsilon=1e-5)

        # FF after conv
        self.ff_conv = FeedForwardModule(encoder_dim=encoder_dim, expansion_factor=feed_forward_expansion_factor, dropout_p=feed_forward_dropout_p)
        self.ln_ff_conv = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x_bt_c, mask_bt, training=False):
        # mask shapes
        mask_flat = tf.reshape(mask_bt, (-1,))

        # Attention block
        residual = x_bt_c
        x = x_bt_c * tf.cast(self.scale_mhsa, x_bt_c.dtype) + tf.cast(self.bias_mhsa, x_bt_c.dtype)
        x = residual + self.mhsa_masa(x, attention_mask=mask_bt, training=training)

        # pad-skip before LN/FF (compact valid tokens)
        B = tf.shape(x)[0]; T = tf.shape(x)[1]; C = tf.shape(x)[2]
        x_skip = tf.reshape(x, (-1, C))                          # (B*T, C)
        x_compact = tf.gather(x_skip, tf.where(mask_flat)[:, 0]) # (N, C)
        x_compact = self.ln_mhsa(x_compact, training=training)
        comp2 = x_compact * tf.cast(self.scale_ff_mhsa, x_compact.dtype) + tf.cast(self.bias_ff_mhsa, x_compact.dtype)
        comp2 = x_compact + self.ff_mhsa(comp2, training=training)
        comp2 = self.ln_ff_mhsa(comp2, training=training)
        # scatter back
        x_skip = tf.tensor_scatter_nd_update(x_skip, tf.expand_dims(tf.where(mask_flat)[:, 0], -1), comp2)
        x = tf.reshape(x_skip, (B, T, C))

        # Conv block
        residual = x
        x = x * tf.cast(self.scale_conv, x.dtype) + tf.cast(self.bias_conv, x.dtype)
        x = residual + self.conv(x, mask_pad_b1t=tf.expand_dims(mask_bt, 1), training=training)

        # pad-skip before LN/FF (conv branch)
        x_skip = tf.reshape(x, (-1, C))
        x_compact = tf.gather(x_skip, tf.where(mask_flat)[:, 0])
        x_compact = self.ln_conv(x_compact, training=training)
        comp2 = x_compact * tf.cast(self.scale_ff_conv, x_compact.dtype) + tf.cast(self.bias_ff_conv, x_compact.dtype)
        comp2 = x_compact + self.ff_conv(comp2, training=training)
        comp2 = self.ln_ff_conv(comp2, training=training)
        x_skip = tf.tensor_scatter_nd_update(x_skip, tf.expand_dims(tf.where(mask_flat)[:, 0], -1), comp2)
        x = tf.reshape(x_skip, (B, T, C))

        return x


# ----------------- Encoder -----------------
class SqueezeformerEncoder(keras.Model):
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        masa_window_size: int = 64,
    ):
        super().__init__()
        self.blocks = []
        for _ in range(num_layers):
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
                    masa_window_size=masa_window_size,
                )
            )

    def call(self, x_bt_c, mask_bt, training=False):
        for blk in self.blocks:
            x_bt_c = blk(x_bt_c, mask_bt, training=training)
        return x_bt_c


# ----------------- Simple Keras Decoder surrogate -----------------
class Decoder(keras.Model):
    """
    Minimal surrogate for Speech2TextDecoder: teacher-forced forward and greedy generate.
    Config expected to provide: d_model, vocab_size, pad_token_id, decoder_start_token_id, eos_token_id.
    """
    def __init__(self, decoder_config):
        super().__init__()
        self.config = decoder_config
        self.d_model = decoder_config.d_model
        self.vocab = decoder_config.vocab_size
        self.pad_token_id = decoder_config.pad_token_id
        self.decoder_start_token_id = decoder_config.decoder_start_token_id
        self.eos_token_id = decoder_config.eos_token_id

        self.embed = layers.Embedding(self.vocab, self.d_model)
        self.dec_block = keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-5),
                layers.Dense(self.d_model, activation="swish"),
                layers.Dense(self.d_model),
            ]
        )
        self.lm_head = layers.Dense(self.vocab, use_bias=False)

    def call(self, x_bt_c, labels=None, attention_mask=None, encoder_attention_mask=None, training=False):
        B = tf.shape(x_bt_c)[0]
        if labels is not None:
            # shift right
            start = tf.fill((B, 1), self.decoder_start_token_id)
            dec_in = tf.concat([start, labels[:, :-1]], axis=1)
        else:
            dec_in = tf.fill((B, 1), self.decoder_start_token_id)

        y = self.embed(dec_in)
        # naive conditioning on encoder via concatenation + dense (illustrative)
        enc_summary = tf.reduce_mean(x_bt_c, axis=1, keepdims=True)
        enc_summary = tf.tile(enc_summary, [1, tf.shape(y)[1], 1])
        h = self.dec_block(y + enc_summary, training=training)
        logits = self.lm_head(h)
        return logits

    def generate(self, x_bt_c, max_new_tokens=33, encoder_attention_mask=None):
        B = tf.shape(x_bt_c)[0]
        dec = tf.fill((B, 1), self.decoder_start_token_id)
        for _ in range(max_new_tokens - 1):
            logits = self(x_bt_c, labels=dec, training=False)
            next_tok = tf.argmax(logits[:, -1], axis=-1, output_type=tf.int32)
            next_tok = tf.expand_dims(next_tok, 1)
            dec = tf.concat([dec, next_tok], axis=1)
            # early stop if all ended
            ended = tf.reduce_any(tf.equal(dec, self.eos_token_id), axis=1)
            if tf.reduce_all(ended):
                break
        return dec


# ----------------- Top-level Net skeleton -----------------
class Net(keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.max_phrase = cfg.max_phrase

        with open(cfg.data_folder + "inference_args.json", "r") as f:
            columns = json.load(f)["selected_columns"]

        self.xyz_landmarks = np.array(columns)
        landmarks = np.array([item[2:] for item in self.xyz_landmarks[: len(self.xyz_landmarks) // 3]])
        self.landmark_types = np.array([self.get_lm_type(lm) for lm in landmarks])

        enc_dim = cfg.encoder_config.encoder_dim
        self.feature_extractor = FeatureExtractor(n_landmarks=cfg.n_landmarks, out_dim=enc_dim)
        quarter = enc_dim // 4
        self.feature_extractor_lhand = FeatureExtractor(n_landmarks=(self.landmark_types == 1).sum(), out_dim=quarter)
        self.feature_extractor_rhand = FeatureExtractor(n_landmarks=(self.landmark_types == 2).sum(), out_dim=quarter)
        self.feature_extractor_face = FeatureExtractor(n_landmarks=(self.landmark_types == 3).sum(), out_dim=quarter)
        self.feature_extractor_pose = FeatureExtractor(n_landmarks=(self.landmark_types == 4).sum(), out_dim=quarter)

        self.encoder = SqueezeformerEncoder(
            input_dim=cfg.encoder_config.input_dim,
            encoder_dim=enc_dim,
            num_layers=cfg.encoder_config.num_layers,
            num_attention_heads=cfg.encoder_config.num_attention_heads,
            feed_forward_expansion_factor=cfg.encoder_config.feed_forward_expansion_factor,
            conv_expansion_factor=cfg.encoder_config.conv_expansion_factor,
            input_dropout_p=cfg.encoder_config.input_dropout_p,
            feed_forward_dropout_p=cfg.encoder_config.feed_forward_dropout_p,
            attention_dropout_p=cfg.encoder_config.attention_dropout_p,
            conv_dropout_p=cfg.encoder_config.conv_dropout_p,
            conv_kernel_size=cfg.encoder_config.conv_kernel_size,
            masa_window_size=cfg.encoder_config.get("masa_window_size", 64),
        )

        self.decoder = Decoder(cfg.transformer_config)
        self.decoder2 = Decoder(cfg.transformer_config)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.aux_fc = layers.Dense(1)
        self.aux_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.aux_loss_weight = cfg.aux_loss_weight
        self.return_aux_logits = cfg.return_aux_logits
        self.bwd_loss_weight = cfg.bwd_loss_weight

        self.val_mode = cfg.val_mode
        self.decoder_mask_aug = cfg.decoder_mask_aug

        print("n_params:", count_parameters(self))

    @staticmethod
    def get_lm_type(lm):
        if "left_hand" in lm:
            return 1
        if "right_hand" in lm:
            return 2
        if "face" in lm:
            return 3
        if "pose" in lm:
            return 4
        return 0

    def call(self, batch, training=False, debug=False):
        x_in = batch["input"]          # (B, T, n_landmarks, 3)
        labels = batch["token_ids"]    # (B, L_out)
        mask_bt = tf.cast(batch["input_mask"], tf.bool)  # (B, T)
        label_mask = batch["attention_mask"]

        # Normalize parts per group like PyTorch (omitted for brevity); assume x_in already normalized

        # Specialized extractors
        xp = tf.identity(x_in)
        x_lhand = self.feature_extractor_lhand(xp[:, :, self.landmark_types == 1], mask_bt, training=training)
        x_rhand = self.feature_extractor_rhand(xp[:, :, self.landmark_types == 2], mask_bt, training=training)
        x_face = self.feature_extractor_face(xp[:, :, self.landmark_types == 3], mask_bt, training=training)
        x_pose = self.feature_extractor_pose(xp[:, :, self.landmark_types == 4], mask_bt, training=training)

        x1 = tf.concat([x_lhand, x_rhand, x_face, x_pose], axis=-1)
        x = self.feature_extractor(x_in, mask_bt, training=training)
        x = x + x1

        x = self.encoder(x, mask_bt, training=training)
        aux_logits = self.aux_fc(x[:, 0])

        if debug:
            return x

        # label masking augmentation
        decoder_labels = tf.identity(labels)
        if training and self.decoder_mask_aug > 0:
            m = tf.random.uniform(tf.shape(labels), 0, 1) < self.decoder_mask_aug
            decoder_labels = tf.where(m, tf.fill(tf.shape(labels), 62), decoder_labels)

        logits = self.decoder(x, labels=decoder_labels, encoder_attention_mask=tf.cast(mask_bt, tf.int32), training=training)

        # backward pass sequence
        x_bwd = tf.reverse(x, axis=[1])
        mask_bwd = tf.reverse(mask_bt, axis=[1])
        # Construct lbl_bwd similar to PyTorch snippet
        # For simplicity, just reverse labels with EOS handling omitted
        lbl_bwd = tf.reverse(labels, axis=[1])
        decoder_lbl_bwd = tf.identity(lbl_bwd)
        if training and self.decoder_mask_aug > 0:
            m = tf.random.uniform(tf.shape(lbl_bwd), 0, 1) < self.decoder_mask_aug
            decoder_lbl_bwd = tf.where(m, tf.fill(tf.shape(lbl_bwd), 62), decoder_lbl_bwd)

        logits_bwd = self.decoder2(x_bwd, labels=decoder_lbl_bwd, encoder_attention_mask=tf.cast(mask_bwd, tf.int32), training=training)

        # losses
        loss_ce = tf.reduce_mean(self.loss_fn(labels, logits))
        loss_ce_bwd = tf.reduce_mean(self.loss_fn(lbl_bwd, logits_bwd))
        loss_aux = self.aux_loss_fn(tf.clip_by_value(batch["score"], 0.0, 1.0)[:, None], aux_logits)
        loss = (1 - self.aux_loss_weight) * (loss_ce * (1 - self.bwd_loss_weight) + self.bwd_loss_weight * loss_ce_bwd) + self.aux_loss_weight * loss_aux

        output = {"loss": loss, "loss_aux": loss_aux, "loss_ce": loss_ce, "loss_ce_bwd": loss_ce_bwd}

        if not training:
            max_phrase = self.max_phrase
            if self.val_mode == "padded":
                generated_ids = self.decoder.generate(x, max_new_tokens=max_phrase + 1, encoder_attention_mask=tf.cast(mask_bt, tf.int32))
            else:
                generated_ids = self.decoder.generate(x, max_new_tokens=max_phrase + 1, encoder_attention_mask=tf.cast(mask_bt, tf.int32))
            # pad/cut to max_phrase
            pad_token = 59
            B = tf.shape(x)[0]
            generated_ids_padded = tf.fill((B, max_phrase), pad_token)
            cutoffs = tf.argmax(tf.cast(tf.equal(generated_ids, self.decoder.eos_token_id), tf.float32), axis=1)
            cutoffs = tf.clip_by_value(cutoffs, 0, max_phrase)
            # simple assign up to cutoffs (vectorized approximation)
            gen_trim = generated_ids[:, :max_phrase]
            generated_ids_padded = tf.where(tf.sequence_mask(cutoffs, maxlen=max_phrase), gen_trim, generated_ids_padded)
            output["generated_ids"] = generated_ids_padded
            output["seq_len"] = batch["seq_len"]
            if self.return_aux_logits:
                output["aux_logits"] = aux_logits

        return output


# ----------------- Example usage -----------------
if __name__ == "__main__":
    # Minimal smoke test with dummy cfg
    class Cfg:
        class Enc:
            input_dim = 80
            encoder_dim = 128
            num_layers = 2
            num_attention_heads = 4
            feed_forward_expansion_factor = 4
            conv_expansion_factor = 2
            input_dropout_p = 0.0
            feed_forward_dropout_p = 0.0
            attention_dropout_p = 0.0
            conv_dropout_p = 0.0
            conv_kernel_size = 15
            masa_window_size = 32
        class Dec:
            d_model = 128
            vocab_size = 64
            pad_token_id = 0
            decoder_start_token_id = 1
            eos_token_id = 2
        encoder_config = Enc()
        transformer_config = Dec()
        max_phrase = 16
        data_folder = "./"
        n_landmarks = 21
        ce_ignore_index = -100
        label_smoothing = 0.0
        aux_loss_weight = 0.1
        return_aux_logits = True
        bwd_loss_weight = 0.5
        val_mode = "padded"
        decoder_mask_aug = 0.0

    cfg = Cfg()
    net = Net(cfg)

    B, T, L, Cc = 2, 64, 21, 3
    batch = {
        "input": tf.random.normal((B, T, L, Cc)),
        "token_ids": tf.random.uniform((B, 12), minval=3, maxval=63, dtype=tf.int32),
        "input_mask": tf.cast(tf.sequence_mask([T, T - 4], maxlen=T), tf.int32),
        "attention_mask": tf.cast(tf.sequence_mask([12, 12], maxlen=12), tf.int32),
        "score": tf.random.uniform((B,), minval=0.0, maxval=1.0),
        "seq_len": tf.constant([T, T - 4]),
    }
    out = net(batch, training=True)
    print("Loss:", out["loss"].numpy())
