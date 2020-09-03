from typing import Optional
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LayerNormalization


class Transformer(tf.keras.Model):
    def __init__(self, 
                 d_model:                   int = 256, 
                 nhead:                     int = 8, 
                 num_encoder_layers:        int = 6,
                 num_decoder_layers:        int = 6, 
                 dim_feedforward:           int = 2048, 
                 dropout:                   float = 0.1,
                 activation:                str = 'relu', 
                 normalize_before:          bool = False,
                 return_intermediate_dec:   bool = False, 
                 **kwargs):

        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead

        enc_norm = LayerNormalization(epsilon=1e-5, name='norm_pre') if normalize_before else None
        self.encoder = TransformerEncoder(d_model, nhead, dim_feedforward,
                                          dropout, activation, normalize_before, enc_norm,
                                          num_encoder_layers, name='encoder')

        dec_norm = LayerNormalization(epsilon=1e-5, name='norm')
        self.decoder = TransformerDecoder(d_model, nhead, dim_feedforward,
                                          dropout, activation, normalize_before, dec_norm,
                                          num_decoder_layers, name='decoder',
                                          return_intermediate=return_intermediate_dec)

    def call(self, src, mask, query_embed, pos_embed, training=False):
        # flatten (N, H, W, C) to (HW, N, C)
        batch_size, h, w, c = src.shape
        # batch_size, rows, cols = [tf.shape(src)[i] for i in range(3)]

        src = tf.reshape(src, [batch_size, -1, self.d_model])
        src = tf.transpose(src, [1, 0, 2])

        pos_embed = tf.reshape(pos_embed, [batch_size, -1, self.d_model])
        pos_embed = tf.transpose(pos_embed, [1, 0, 2])

        query_embed = tf.expand_dims(query_embed, axis=1)
        query_embed = tf.tile(query_embed, [1, batch_size, 1])

        mask = tf.reshape(mask, [batch_size, -1])

        tgt = tf.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos_embed=pos_embed, training=training)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos_embed=pos_embed, query_embed=query_embed, training=training)

        hs = tf.transpose(hs, [0, 2, 1, 3])
        memory = tf.transpose(memory, [1, 0, 2])
        memory = tf.reshape(memory, [batch_size, rows, cols, self.d_model])

        return hs, memory


class TransformerEncoder(tf.keras.Model):
    def __init__(self, 
                 d_model:               int = 256, 
                 nhead:                 int = 8, 
                 dim_feedforward:       int = 2048,
                 dropout:               float = 0.1, 
                 activation:            str = 'relu', 
                 normalize_before:      bool = False, 
                 norm:                  tf.keras.layers.Layer = None,
                 num_encoder_layers:    int = 6, 
                 **kwargs):

        super().__init__(**kwargs)
        self.enc_layers = [TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before,
                                                   name='layers/%d'%i)
                           for i in range(num_encoder_layers)]
        
        self.norm = norm

    def call(self, 
             src, 
             mask:                  Optional[tf.Tensor] = None, 
             src_key_padding_mask:  Optional[tf.Tensor] = None,
             pos:                   Optional[tf.Tensor] = None, 
             training:              bool                = False):
        
        output = src
        for layer in self.enc_layers:
            output = layer(output, 
                           src_mask=mask, 
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos, training=training)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerDecoder(tf.keras.Model):
    def __init__(self, 
                 d_model:               int                     = 256, 
                 nhead:                 int                     = 8, 
                 dim_feedforward:       int                     = 2048,
                 dropout:               float                   = 0.1, 
                 activation:            str                     = 'relu', 
                 normalize_before:      bool                    = False, 
                 norm:                  tf.keras.layers.Layer   = None,
                 num_decoder_layers:    int                     = 6, 
                 return_intermediate:   bool                    = False,
                 **kwargs):

        super().__init__(**kwargs)
        self.dec_layers = [DecoderLayer(d_model, nhead, dim_feedforward,
                                        dropout, activation, normalize_before,
                                        name='layers/%d'%i)
                           for i in range(num_decoder_layers)]

        self.norm = norm
        self.return_intermediate = return_intermediate

    def call(self, target, memory, target_mask=None, memory_mask=None,
             target_key_padding_mask=None, memory_key_padding_mask=None,
             pos_encoding=None, query_encoding=None, training=False):

        x = target
        intermediate = []

        for layer in self.dec_layers:
            x = layer(x, memory,
                      target_mask=target_mask,
                      memory_mask=memory_mask,
                      target_key_padding_mask=target_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      pos_encoding=pos_encoding,
                      query_encoding=query_encoding)

            if self.return_intermediate:
                if self.norm:
                    intermediate.append(self.norm(x))
                else:
                    intermediate.append(x)

        if self.return_intermediate:
            return tf.stack(intermediate, axis=0)

        if self.norm:
            x = self.norm(x)

        return x


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 d_model:           int     = 256, 
                 nhead:             int     = 8, 
                 dim_feedforward:   int     = 2048,
                 dropout:           float   = 0.1, 
                 activation:        str     = 'relu', 
                 normalize_before:  bool    = False,
                 **kwargs):

        super().__init__(**kwargs)
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, name='self_attn')
        # Implementation of Feedforward model
        self.linear1 = Dense(dim_feedforward, name='linear1')
        self.dropout = Dropout(dropout)
        self.linear2 = Dense(d_model, name='linear2')

        self.norm1 = LayerNormalization(epsilon=1e-5, name='norm1')
        self.norm2 = LayerNormalization(epsilon=1e-5, name='norm2')
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = Activation(activation, name='activation')
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[tf.Tensor]):
        return tensor if pos is None else tensor + pos

    def call_post(self, 
                  src, 
                  src_mask:             Optional[tf.Tensor] = None, 
                  src_key_padding_mask: Optional[tf.Tensor] = None,
                  pos:                  Optional[tf.Tensor] = None, 
                  training:             bool = False):
        
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        if pos is None:
            query = key = src
        else:
            query = key = src + pos

        attn_src = self.self_attn((query, key, src), attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask,
                                     need_weights=False)
        src += self.dropout(attn_src, training=training)
        src = self.norm1(src)

        x = self.linear1(src)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        x = self.linear2(x)
        src += self.dropout(x, training=training)
        src = self.norm2(src)
        
        return src

    def call_pre(self, 
                 src, 
                 src_mask:              Optional[tf.Tensor] = None, 
                 src_key_padding_mask:  Optional[tf.Tensor] = None,
                 pos:                   Optional[tf.Tensor] = None, 
                 training:              bool = False):
        raise Exception('pre_norm_call not implemented yet')

    def call(self, 
             src, 
             src_mask:              Optional[tf.Tensor] = None, 
             src_key_padding_mask:  Optional[tf.Tensor] = None,
             pos:                   Optional[tf.Tensor] = None, 
             training:              bool = False):

        if self.normalize_before:
            return self.pre_norm_call(src, src_mask, src_key_padding_mask, pos, training)
        return self.post_norm_call(src, src_mask, src_key_padding_mask, pos, training)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', normalize_before=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout,
                                            name='self_attn')
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout,
                                                 name='multihead_attn')

        self.dropout = Dropout(dropout)
        self.activation = Activation(activation)

        self.linear1 = Dense(dim_feedforward, name='linear1')
        self.linear2 = Dense(d_model, name='linear2')

        self.norm1 = LayerNormalization(epsilon=1e-5, name='norm1')
        self.norm2 = LayerNormalization(epsilon=1e-5, name='norm2')
        self.norm3 = LayerNormalization(epsilon=1e-5, name='norm3')

        self.normalize_before = normalize_before

    def call(self, target, memory, target_mask=None, memory_mask=None,
             target_key_padding_mask=None, memory_key_padding_mask=None,
             pos_encoding=None, query_encoding=None, training=False):
        if self.normalize_before:
            return self.pre_norm_call(target, memory, target_mask, memory_mask,
                                      target_key_padding_mask, memory_key_padding_mask,
                                      pos_encoding, query_encoding, training=training)
        return self.post_norm_call(target, memory, target_mask, memory_mask,
                                   target_key_padding_mask, memory_key_padding_mask,
                                   pos_encoding, query_encoding, training=training)

    def pre_norm_call(self, target, memory, target_mask=None, memory_mask=None,
                       target_key_padding_mask=None, memory_key_padding_mask=None,
                       pos_encoding=None, query_encoding=None, training=False):
        raise Exception('pre_norm_call not implemented yet')

    def post_norm_call(self, target, memory, target_mask=None, memory_mask=None,
                       target_key_padding_mask=None, memory_key_padding_mask=None,
                       pos_encoding=None, query_encoding=None, training=False):

        query_tgt = key_tgt = target + query_encoding
        attn_target = self.self_attn((query_tgt, key_tgt, target), attn_mask=target_mask,
                                    key_padding_mask=target_key_padding_mask,
                                    need_weights=False)
        target += self.dropout(attn_target, training=training)
        target = self.norm1(target)

        query_tgt = target + query_encoding
        key_mem = memory + pos_encoding
        
        attn_target2 = self.multihead_attn((query_tgt, key_mem, memory), attn_mask=memory_mask,
                                           key_padding_mask=memory_key_padding_mask,
                                           need_weights=False)
        target += self.dropout(attn_target2, training=training)
        target = self.norm2(target)

        x = self.linear1(target)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        x = self.linear2(x)
        target += self.dropout(x, training=training)
        target = self.norm3(target)
        
        return target


class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    Implementation of Multi Head Attention in Transformer.
    Original code is https://github.com/Leonardo-Blanger/detr_tensorflow/blob/master/models/transformer.py
    '''
    def __init__(self, 
                 d_model:   int, 
                 nhead:     int, 
                 dropout:   float = 0., 
                 **kwargs):

        super().__init__(**kwargs)

        self.d_model = d_model
        self.nhead = nhead

        assert d_model % nhead == 0
        self.head_dim = d_model // nhead

        self.dropout = Dropout(rate=dropout)

    def build(self, input_shape):
        in_dim = sum([input_shape[:3]])
        self.in_proj_weight = tf.Variable(
            tf.zeros((in_dim, self.d_model), dtype=tf.float32), name='in_proj_weight')
        self.in_proj_bias = tf.Variable(tf.zeros((in_dim,), dtype=tf.float32),
                                        name='in_proj_bias')

        self.out_proj_weight = tf.Variable(
            tf.zeros((self.d_model, self.d_model), dtype=tf.float32), name='out_proj/kernel')
        self.out_proj_bias = tf.Variable(
            tf.zeros((self.d_model,), dtype=tf.float32), name='out_proj/bias')

    def call(self, 
             query:             tf.Tensor, 
             key:               tf.Tensor,
             value:             tf.Tensor,
             attn_mask:         Optional[tf.Tensor] = None, 
             key_padding_mask:  Optional[tf.Tensor] = None,
             need_weights:      bool = True, 
             training:          bool = False):

        batch_size = tf.shape(query)[1]
        target_len = tf.shape(query)[0]
        source_len = tf.shape(key)[0]

        W = self.in_proj_weight[:self.d_model, :]
        b = self.in_proj_bias[:self.d_model]
        WQ = tf.matmul(query, W, transpose_b=True) + b

        W = self.in_proj_weight[self.d_model:2*self.d_model, :]
        b = self.in_proj_bias[self.d_model:2*self.d_model]
        WK = tf.matmul(key, W, transpose_b=True) + b

        W = self.in_proj_weight[2*self.d_model:, :]
        b = self.in_proj_bias[2*self.d_model:]
        WV = tf.matmul(value, W, transpose_b=True) + b

        WQ *= float(self.head_dim) ** -0.5
        WQ = tf.reshape(WQ, [target_len, batch_size * self.nhead, self.head_dim])
        WQ = tf.transpose(WQ, [1, 0, 2])
        
        WK = tf.reshape(WK, [source_len, batch_size * self.nhead, self.head_dim])
        WK = tf.transpose(WK, [1, 0, 2])

        WV = tf.reshape(WV, [source_len, batch_size * self.nhead, self.head_dim])
        WV = tf.transpose(WV, [1, 0, 2])
        
        attn_output_weights = tf.matmul(WQ, WK, transpose_b=True)

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = tf.reshape(attn_output_weights,
                                [batch_size, self.nhead, target_len, source_len])

            key_padding_mask = tf.expand_dims(key_padding_mask, 1)
            key_padding_mask = tf.expand_dims(key_padding_mask, 2)
            key_padding_mask = tf.tile(key_padding_mask, [1, self.nhead, target_len, 1])

            attn_output_weights = tf.where(key_padding_mask,
                                           tf.zeros_like(attn_output_weights) + float('-inf'),
                                           attn_output_weights)
            attn_output_weights = tf.reshape(attn_output_weights,
                                [batch_size * self.nhead, target_len, source_len])

        attn_output_weights = tf.nn.softmax(attn_output_weights, axis=-1)
        attn_output_weights = self.dropout(attn_output_weights, training=training)

        attn_output = tf.matmul(attn_output_weights, WV)
        attn_output = tf.transpose(attn_output, [1, 0, 2])
        attn_output = tf.reshape(attn_output, [target_len, batch_size, self.d_model])
        attn_output = tf.matmul(attn_output, self.out_proj_weight,
                                transpose_b=True) + self.out_proj_bias

        if need_weights:
            attn_output_weights = tf.reshape(attn_output_weights,
                            [batch_size, self.nhead, target_len, source_len])
            # Retrun the average weight over the heads
            avg_weights = tf.reduce_mean(attn_output_weights, axis=1)
            return attn_output, avg_weights
        
        return attn_output