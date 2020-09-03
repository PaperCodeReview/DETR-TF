import tensorflow as tf


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = tf.unstack(x)
    b = [(x_c - .5 * w), (y_c - .5 * h),
         (x_c + .5 * w), (y_c + .5 * h)]
    return tf.stack(b, axis=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = tf.unstack(x)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return tf.stack(b, axis=-1)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = tf.reduce_max(boxes1[:, None, :2], boxes2[:, :2])
    rb = tf.reduce_min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = tf.clip_by_value(rb - lt,
                          clip_value_min=0,
                          clip_value_max=tf.reduce_max(rb - lt))
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union