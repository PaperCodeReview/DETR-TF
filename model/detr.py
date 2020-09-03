from typing import Dict

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Embedding

from model.backbone import build_backbone
from model.transformer import build_transformer


class DETR(tf.keras.Model):
    """ This is the DETR module that performs object detection """
    def __init__(self, 
                 backbone:      tf.keras.Model, 
                 transformer:   tf.keras.Model, 
                 num_classes:   int, 
                 num_queries:   int, 
                 aux_loss:      bool = False, 
                 **kwargs):
                 
        super(DETR, self).__init__(**kwargs)
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = Dense(num_classes+1, name='class_embed')
        self.bbox_embed = MLP(hidden_dim, 4, 3, name='bbox_embed')
        self.query_embed = Embedding(num_queries, hidden_dim, name='query_embed')
        self.query_embed.build((num_queries, hidden_dim))
        self.input_proj = Conv2D(hidden_dim, 1, name='input_proj')
        self.backbone = backbone
        self.aux_loss = aux_loss

    def call(self, samples: Dict):
        features, pos = self.backbone(samples)
        src, mask = features[-1][1]['img'], features[-1][1]['mask']
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weights[0], pos[-1][1])


class PostProcess(tf.keras.Model):
    def call(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logtis'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = tf.nn.softmax(out_logits, axis=-1)
        scores, labels = prob[...,:-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes


class MLP(tf.keras.layers.Layer):
    def __init__(self, 
                 hidden_dim:    int, 
                 output_dim:    int, 
                 num_layers:    int, 
                 **kwargs):
                 
        self(MLP, self).__init__(**kwargs)
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = [Dense(k) for k in h+[output_dim]]

    def call(self, x):
        for i, layer in enumerate(self.layers):
            x = tf.nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = DETR(
        backbone,
        transformer,
        num_classes=args.classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss
    )
    if args.masks:
        raise NotImplementedError('Segmentation model is not implemented yet.')

