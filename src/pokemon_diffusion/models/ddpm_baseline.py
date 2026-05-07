import math
import tensorflow as tf
from tensorflow.keras import layers


class SinusoidalPositionEmbeddings(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, time):
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)

        freqs = tf.exp(
            tf.range(half_dim, dtype=tf.float32) * -emb_scale
        )

        time = tf.cast(time, tf.float32)
        emb = time[:, None] * freqs[None, :]

        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


class ResidualBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()

        self.norm1 = layers.GroupNormalization(groups=8)
        self.act1 = layers.Activation("swish")
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, padding="same")

        self.time_mlp = tf.keras.Sequential([
            layers.Activation("swish"),
            layers.Dense(out_channels),
        ])

        self.norm2 = layers.GroupNormalization(groups=8)
        self.act2 = layers.Activation("swish")
        self.dropout = layers.Dropout(dropout)
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, padding="same")

        if in_channels != out_channels:
            self.residual_conv = layers.Conv2D(out_channels, kernel_size=1)
        else:
            self.residual_conv = layers.Lambda(lambda x: x)

    def call(self, x, time_emb, training=False):
        residual = self.residual_conv(x)

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb[:, None, None, :]

        h = h + time_emb

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h, training=training)
        h = self.conv2(h)

        return h + residual


class AttentionBlock(layers.Layer):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = layers.GroupNormalization(groups=8)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=channels // num_heads,
        )

    def call(self, x):
        residual = x

        h = self.norm(x)

        batch_size = tf.shape(h)[0]
        height = tf.shape(h)[1]
        width = tf.shape(h)[2]
        channels = tf.shape(h)[3]

        h = tf.reshape(h, [batch_size, height * width, channels])
        h = self.attn(h, h)
        h = tf.reshape(h, [batch_size, height, width, channels])

        return h + residual


class SimpleUNetTF(tf.keras.Model):
    def __init__(
        self,
        image_size=28,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mult=(1, 2, 4),
        time_emb_dim=128,
        num_res_blocks=2,
        attention_resolutions=(7,),
        dropout=0.1,
    ):
        super().__init__()

        self.image_size = image_size
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions

        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)

        self.time_mlp = tf.keras.Sequential([
            layers.Dense(time_emb_dim * 4),
            layers.Activation("swish"),
            layers.Dense(time_emb_dim),
        ])

        self.conv_in = layers.Conv2D(
            base_channels,
            kernel_size=3,
            padding="same",
        )

        channels = [base_channels * mult for mult in channel_mult]

        self.encoder = []
        self.encoder_downs = []

        in_ch = base_channels
        resolution = image_size

        for level, out_ch in enumerate(channels):
            blocks = []

            for _ in range(num_res_blocks):
                blocks.append(
                    ResidualBlock(
                        in_ch,
                        out_ch,
                        time_emb_dim,
                        dropout,
                    )
                )
                in_ch = out_ch

            if resolution in attention_resolutions:
                blocks.append(AttentionBlock(out_ch))

            self.encoder.append(blocks)

            if level < len(channels) - 1:
                self.encoder_downs.append(
                    layers.Conv2D(
                        out_ch,
                        kernel_size=3,
                        strides=2,
                        padding="same",
                    )
                )
                resolution //= 2

        mid_ch = channels[-1]

        self.middle = [
            ResidualBlock(mid_ch, mid_ch, time_emb_dim, dropout),
            AttentionBlock(mid_ch),
            ResidualBlock(mid_ch, mid_ch, time_emb_dim, dropout),
        ]

        self.decoder = []
        self.decoder_ups = []

        in_ch = mid_ch

        for level, out_ch in enumerate(reversed(channels)):
            blocks = []

            for i in range(num_res_blocks + 1):
                skip_ch = out_ch if i == 0 else 0

                blocks.append(
                    ResidualBlock(
                        in_ch + skip_ch,
                        out_ch,
                        time_emb_dim,
                        dropout,
                    )
                )
                in_ch = out_ch

            if resolution in attention_resolutions:
                blocks.append(AttentionBlock(out_ch))

            self.decoder.append(blocks)

            if level < len(channels) - 1:
                self.decoder_ups.append(
                    layers.Conv2DTranspose(
                        out_ch,
                        kernel_size=4,
                        strides=2,
                        padding="same",
                    )
                )
                resolution *= 2

        self.norm_out = layers.GroupNormalization(groups=8)
        self.act_out = layers.Activation("swish")
        self.conv_out = layers.Conv2D(
            out_channels,
            kernel_size=3,
            padding="same",
        )

    def call(self, inputs, training=False):
        x, time = inputs

        time_emb = self.time_embedding(time)
        time_emb = self.time_mlp(time_emb)

        h = self.conv_in(x)

        encoder_outputs = [h]

        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, time_emb, training=training)
                else:
                    h = block(h)

            encoder_outputs.append(h)

            if level < len(self.encoder_downs):
                h = self.encoder_downs[level](h)

        for block in self.middle:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb, training=training)
            else:
                h = block(h)

        for level, blocks in enumerate(self.decoder):
            skip = encoder_outputs.pop()

            for i, block in enumerate(blocks):
                if isinstance(block, ResidualBlock):
                    if i == 0:
                        h = tf.concat([h, skip], axis=-1)
                    h = block(h, time_emb, training=training)
                else:
                    h = block(h)

            if level < len(self.decoder_ups):
                h = self.decoder_ups[level](h)

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)

        return h