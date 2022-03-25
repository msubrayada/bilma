from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Layer, Dense, concatenate, Input, add, Dropout, LayerNormalization, MultiHeadAttention, Embedding
import tensorflow as tf
import numpy as np


class EncoderBlock(Layer):
    def __init__(self, patch_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.p_d = patch_dim
        self.n_h = num_heads
        self.f_d = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=patch_dim)
        self.ffn = Sequential(
            #[Conv1D(ff_dim, kernel_size=1, activation=tf.nn.gelu), 
            # Conv1D(patch_dim, kernel_size=1),]
            [Dense(ff_dim, activation=tf.nn.gelu), 
             Dense(patch_dim),]
        )
        #self.layernorm0 = LayerNormalization(epsilon=1e-6)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def get_config(self):
        config = super(EncoderBlock, self).get_config()
        config.update({"patch_dim":self.p_d, "num_heads":self.n_h, "ff_dim":self.f_d, "rate":self.rate})
        return config

    def call(self, inputs, training=False):
        #inputs = self.layernorm0(inputs)
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(add([inputs, attn_output]))
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(add([out1, ffn_output]))
    

class DecoderBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.e_d = embed_dim
        self.n_h = num_heads
        self.f_d = ff_dim
        self.rate = rate
        
        self.att1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            #[Conv1D(ff_dim, kernel_size=1, activation=tf.nn.gelu), 
            # Conv1D(embed_dim, kernel_size=1),]
            [Dense(ff_dim, activation=tf.nn.gelu), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
        
    def get_config(self):
        config = super(DecoderBlock, self).get_config()
        config.update({"embed_dim":self.e_d, "num_heads":self.n_h, "ff_dim":self.f_d, "rate":self.rate})
        return config

    def call(self, inputs, encoder_output, look_ahead_mask, padding_mask, training=None):
        y, attn_output1 = self.att1(inputs, inputs, attention_mask=look_ahead_mask, return_attention_scores=True)
        y = self.dropout1(y, training=training)
        y = add([inputs, y])                
        out1 = self.layernorm1(y)
        
        y, attn_encoder = self.att2(out1, encoder_output, attention_mask=padding_mask, return_attention_scores=True)
        y = self.dropout2(y, training=training)
        y = add([out1, y])                
        out2 = self.layernorm1(y)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        final_output =  self.layernorm2(out2 + ffn_output)
        
        return final_output, attn_output1, attn_encoder


class Encoder(Layer):
    def __init__(self, n, embed_dim, max_length, num_heads, ff_dim, rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.n = n        
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.n_h = num_heads
        self.f_d = ff_dim
        self.rate = rate
        self._layers = [EncoderBlock(embed_dim, num_heads, ff_dim, rate=0.1) for _ in range(n)]
        self.pe = positional_encoding(self.max_length, self.embed_dim)
        
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"n": self.n, "embed_dim":self.embed_dim, "max_length": self.max_length, "num_heads":self.n_h, "ff_dim":self.f_d, "rate":self.rate})
        return config
    
    def call(self, x, training=False):
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x = x + self.pe[:, :tf.shape(x)[1], :]
        for layer in self._layers:
            x = layer(x, training)
        return x

    
class Decoder(Layer):
    def __init__(self, n, embed_dim, max_length, num_heads, ff_dim, rate=0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.n = n
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.n_h = num_heads
        self.f_d = ff_dim
        self.rate = rate
        self._layers = [DecoderBlock(embed_dim, num_heads, ff_dim, rate=0.1) for _ in range(n)]
        self.pe = positional_encoding(self.max_length, self.embed_dim)
    
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"n": self.n, "embed_dim":self.embed_dim, "max_length": self.max_length, "num_heads":self.n_h, "ff_dim":self.f_d, "rate":self.rate})
        return config
    
    def call(self, x, encoder_output, look_ahead_mask, padding_mask, training):      
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x = x + self.pe[:, :tf.shape(x)[1], :]
        
        for layer in self._layers:
            x, self_att, enc_att = layer(x, encoder_output, look_ahead_mask, padding_mask, training)

        return x




# =========================================
#   M A S K S 
# =========================================
def create_padding_mask(seq):
    """
    For self-attention
    seq shape(bs, max_length, emb_dim)
    output shape (bs, max_length, max_length)
    """
    mask = tf.cast(tf.not_equal(seq, 0), tf.bool)
    mask = tf.reduce_any(mask, 2)
    mask = tf.repeat(mask, seq.shape[1], 0)
    mask = tf.reshape(mask, (-1,seq.shape[1], seq.shape[1]))
    return tf.cast(mask, tf.float32)


def create_cross_padding_mask(seq, target_seq):
    """
    For cross-attention
    seq shape(bs, k, image_features)
    target_seq(bs, max_length, emb_dim)
    output shape (bs, max_length, k)
    """
    mask = tf.cast(tf.not_equal(target_seq, 0), tf.bool)
    mask = tf.reduce_any(mask, 2)
    mask = tf.repeat(mask, seq.shape[1], 0)
    mask = tf.reshape(mask, (-1, tf.shape(seq)[1], tf.shape(target_seq)[1]))
    mask = tf.transpose(mask, [0, 2, 1])
    return mask


def create_look_ahead_mask(seq):
    """
    seq shape(bs, max_length, emb_dim)
    output 2D matrix of shape (bs, max_length, max_length) with ones on the diagonal and below.
    """
    size = seq.shape[1]
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = tf.expand_dims(mask, 0)
    mask = tf.repeat(mask, tf.shape(seq)[0], 0)
    return mask


def create_masks(seq, target_seq):
    decoder_mask = create_padding_mask(target_seq)
    decoder_mask *= create_look_ahead_mask(target_seq)
    cross_att_mask = create_cross_padding_mask(seq, target_seq)
    return decoder_mask, cross_att_mask
        
    
def create_masks_looking_ahead(seq, target_seq):
    decoder_mask = create_padding_mask(target_seq)
    cross_att_mask = create_cross_padding_mask(seq, target_seq)
    return decoder_mask, cross_att_mask
    
# =========================================
#   P O S I T I O N A L   E N C O D I N G
# =========================================
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

@tf.autograph.experimental.do_not_convert
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({"num_patches": self.num_patches, "projection_dim":self.projection_dim})
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded