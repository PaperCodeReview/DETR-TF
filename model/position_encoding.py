import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding


class PositionEmbeddingSine(tf.keras.Model):
    def __init__(self, 
                 num_pos_feats: int     = 64, 
                 temperature:   int     = 10000, 
                 normalize:     bool    = False, 
                 scale:         float   = None, 
                 **kwargs):

        super(PositionEmbeddingSine, self).__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def call(self, inputs):
        x = inputs['img']
        mask = inputs['mask']
        assert mask is not None
        not_mask = tf.cast(~mask, tf.float32)
        y_embed = tf.cumsum(not_mask, axis=1)
        x_embed = tf.cumsum(not_mask, axis=2)
        if self.normalize:
            eps = K.epsilon()
            y_embed = y_embed / (y_embed[:,-1:,:]+eps) * self.scale
            x_embed = x_embed / (x_embed[:,:,-1:]+eps) * self.scale

        dim_t = tf.range(self.num_pos_feats, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., tf.newaxis] / dim_t
        pos_y = y_embed[..., tf.newaxis] / dim_t

        pos_x = tf.stack((tf.sin(pos_x[...,0::2]), tf.cos(pos_x[...,1::2])), axis=-1)
        pos_y = tf.stack((tf.sin(pos_y[...,0::2]), tf.cos(pos_y[...,1::2])), axis=-1)

        shape = [tf.shape(pos_x)[i] for i in range(3)] + [-1]
        pos_x = tf.reshape(pos_x, shape)
        pos_y = tf.reshape(pos_y, shape)
        pos = tf.concat((pos_y, pos_x), axis=-1)
        return pos


class PositionEmbeddingLearned(tf.keras.Model):
    def __init__(self, num_pos_feats: int = 256, **kwargs):
        super(PositionEmbeddingLearned, self).__init__(**kwargs)
        self.row_embed = Embedding(50, num_pos_feats)
        self.col_embed = Embedding(50, num_pos_feats)

    def call(self, inputs):
        x = inputs['img']
        h, w = x.shape[1:-1]
        i = tf.range(w)
        j = tf.range(h)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = tf.concat((
            tf.repeat(tf.expand_dims(x_emb, axis=0), repeats=[h], axis=0),
            tf.repeat(tf.expand_dims(y_emb, axis=1), repeats=[w], axis=1)
        ), axis=-1)
        pos = tf.repeat(tf.expand_dims(pos, axis=0), repeats=[x.shape[0]], axis=0)
        return pos
        

def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding