from typing import Dict
import tensorflow as tf

from model.position_encoding import build_position_encoding

class FrozenBatchNorm2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FrozenBatchNorm2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=[input_shape[-1]], initializer="ones", trainable=False
        )
        self.bias = self.add_weight(
            name="bias", shape=[input_shape[-1]], initializer="zeros", trainable=False
        )
        self.running_mean = self.add_weight(
            name="running_mean", shape=[input_shape[-1]], initializer="zeros", trainable=False
        )
        self.running_var = self.add_weight(
            name="running_var", shape=[input_shape[-1]], initializer="ones", trainable=False
        )

    def call(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = tf.reshape(self.weight, (1, 1, 1, -1))
        b = tf.reshape(self.bias, (1, 1, 1, -1))
        rv = tf.reshape(self.running_var, (1, 1, 1, -1))
        rm = tf.reshape(self.running_mean, (1, 1, 1, -1))
        eps = 1e-5
        scale = w * tf.math.rsqrt(rv + eps)
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(tf.keras.Model):
    def __init__(self, 
                 backbone:              tf.keras.Model, 
                 train_backbone:        bool, 
                 num_channels:          int, 
                 return_interm_layers:  bool, 
                 **kwargs):

        super(BackboneBase, self).__init__(**kwargs)
        for layer in backbone.layers:
            if not train_backbone:
                layer.trainable = False
        self.body = backbone
        self.num_channels = num_channels

    def call(self, inputs: Dict):
        xs = self.body(inputs['img'])
        out = {}
        for name, x in xs.items():
            m = inputs['mask']
            assert m is not None
            m = tf.cast(m, tf.float32)
            mask = tf.cast(tf.image.resize(m, x.shape[1:-1], method='nearest'), tf.bool)
            out[name] = {'img': x, 'mask': mask}
        return out

    
class Backbone(BackboneBase):
    def __init__(self, 
                 name:                  str, 
                 train_backbone:        bool, 
                 return_interm_layers:  bool, 
                 dilation:              bool, 
                 **kwargs):

        if name == 'resnet50':
            from model.resnet import ResNet50Backbone as b
        elif name == 'resnet101':
            from model.resnet import ResNet101Backbone as b
        
        backbone = b(return_interm_layers=return_interm_layers,
                     replace_stride_with_dilation=[False, False, dilation])
        num_channels = 512 if name in ['resnet18', 'resnet34'] else 2048
        super(Backbone, self).__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(tf.keras.Model):
    def __init__(self, 
                 backbone:              tf.keras.Model, 
                 position_embedding:    tf.keras.Model, 
                 **kwargs):

        super(Joiner, self).__init__(**kwargs)
        self.backbone = backbone
        self.position_embedding = position_embedding
        
    def call(self, inputs):
        xs = self.backbone(inputs)
        out = []
        pos = []
        for name, x in xs.items():
            out.append((name, x))
            pos.append((name, tf.cast(self.position_embedding(x), tf.float32)))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model