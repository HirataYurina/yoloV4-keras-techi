from functools import wraps

import keras as keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from nets.darknet53 import csp_darknet_body
from utils.utils import compose, diou_nms
from config.configs import CONFIG


# --------------------------------------------------
# convolution without BN and activation
# --------------------------------------------------
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------
# Convolution + BatchNormalization + LeakyReLU
# ---------------------------------------------------
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


# --------------------------------------------------------------------------
# feature map ---> outputs [batch, 13, 13, num_anchors*(num_classes+5)]
# This function is used in yoloV3
# --------------------------------------------------------------------------
def make_last_layers(x, num_filters, out_filters):

    # process 5 convolution operation
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)

    # adjust the output channels to num_anchors*(num_classes+5)
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3, 3))(x)
    y = DarknetConv2D(out_filters, (1, 1))(y)
            
    return x, y


# ---------------------------------------------------------------------------------------
# CSPDarknet + SPP + PANet(yoloV4 changes adding operation to concatenating operation)
# TODO: Path Aggregation Network for Instance Segmentation
# https://arxiv.org/abs/1803.01534
#      |
#      |
#   -------        -------        -------
#   |     | -----> |     | -----> |     | -----> [batch, 52, 52, 3 * 85]
#   -------        -------        -------
#      |              ^              |
#      |              |              V
#   -------        -------        -------
#   |     | -----> |     | -----> |     | -----> [batch, 26, 26, 3 * 85]
#   -------        -------        -------
#      |              ^              |
#      |              |              V
#   -------        -------        -------
#   |     | -----> |     | -----> |     | -----> [batch, 13, 13, 3 * 85]
#   -------        -------        -------
#
# ---------------------------------------------------------------------------------------
def yolo_body(inputs, num_anchors, num_classes):

    # get feature map from backbone
    feat1, feat2, feat3 = csp_darknet_body(inputs)
    feat3 = DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)
    feat3 = DarknetConv2D_BN_Leaky(1024, (3, 3))(feat3)
    feat3 = DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)

    # spatial pooling pyramid
    # enhance respective fields
    pool1 = MaxPooling2D(13, strides=1, padding='same')(feat3)
    pool2 = MaxPooling2D(9, strides=1, padding='same')(feat3)
    pool3 = MaxPooling2D(5, strides=1, padding='same')(feat3)
    pool_fusion = Concatenate()([pool1, pool2, pool3, feat3])

    y5 = DarknetConv2D_BN_Leaky(512, (1, 1))(pool_fusion)
    y5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(y5)
    y5 = DarknetConv2D_BN_Leaky(512, (1, 1))(y5)
    y5_upsample = DarknetConv2D_BN_Leaky(256, (1, 1))(y5)
    y5_upsample = UpSampling2D()(y5_upsample)

    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)
    y4 = Concatenate()([y4, y5_upsample])
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)
    y4 = DarknetConv2D_BN_Leaky(512, (3, 3))(y4)
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)
    y4 = DarknetConv2D_BN_Leaky(512, (3, 3))(y4)
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)

    y4_upsample = DarknetConv2D_BN_Leaky(128, (1, 1))(y4)
    y4_upsample = UpSampling2D()(y4_upsample)

    y3 = DarknetConv2D_BN_Leaky(128, (1, 1))(feat1)
    y3 = Concatenate()([y3, y4_upsample])
    y3 = DarknetConv2D_BN_Leaky(128, (1, 1))(y3)
    y3 = DarknetConv2D_BN_Leaky(256, (3, 3))(y3)
    y3 = DarknetConv2D_BN_Leaky(128, (1, 1))(y3)
    y3 = DarknetConv2D_BN_Leaky(256, (3, 3))(y3)
    y3 = DarknetConv2D_BN_Leaky(128, (1, 1))(y3)

    # the output of C3 and it reference to little anchors
    y3_output = DarknetConv2D_BN_Leaky(256, (3, 3))(y3)
    y3_output = DarknetConv2D(num_anchors * (5 + num_classes), (1, 1))(y3_output)

    # down sample and start aggregating the path from lower layers to higher layers
    # PANet
    y3 = ZeroPadding2D(padding=((1, 0), (1, 0)))(y3)
    y3_downsample = DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(y3)
    y4 = Concatenate()([y3_downsample, y4])
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)
    y4 = DarknetConv2D_BN_Leaky(512, (3, 3))(y4)
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)
    y4 = DarknetConv2D_BN_Leaky(512, (3, 3))(y4)
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)

    # the output of C4 and it reference to middle anchors
    y4_output = DarknetConv2D_BN_Leaky(512, (3, 3))(y4)
    y4_output = DarknetConv2D(num_anchors * (5 + num_classes), (1, 1))(y4_output)

    y4 = ZeroPadding2D(padding=((1, 0), (1, 0)))(y4)
    y4_downsample = DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(y4)
    y5 = Concatenate()([y4_downsample, y5])
    y5 = DarknetConv2D_BN_Leaky(512, (1, 1))(y5)
    y5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(y5)
    y5 = DarknetConv2D_BN_Leaky(512, (1, 1))(y5)
    y5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(y5)
    y5 = DarknetConv2D_BN_Leaky(512, (1, 1))(y5)

    # the output of C5 and it reference to large anchors
    y5_output = DarknetConv2D_BN_Leaky(1024, (3, 3))(y5)
    y5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(y5_output)

    return Model(inputs, [y5_output, y4_output, y3_output])


# --------------------------------------------------------------------------------------------
# decode raw prediction to confidence, class probability and bounding boxes location
# TODO: add scale exceed 1 to eliminate grid sensitivity
# --------------------------------------------------------------------------------------------
def yolo_head(i,
              feats,
              anchors,
              num_classes,
              input_shape,
              sigmoid_factor=CONFIG.TRAIN.SCALE_XY,
              calc_loss=False):

    # assert sigmoid_factor >= 1, 'sigmoid_factor must exceed 1'
    sigmoid_factor = sigmoid_factor[i]
    num_anchors = len(anchors)

    # [1, 1, 1, num_anchors, 2]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # generator grid xy
    # (13, 13, 1, 2)
    # feature:(batch, 13, 13, 3, 85)
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # (batch_size, 13 13, 3, 85)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # decode the raw prediction
    # box_xy is the center of ground truth
    # box_wh is the w and h of gt
    box_xy = (sigmoid_factor * K.sigmoid(feats[..., :2]) -
              0.5 * (sigmoid_factor - 1) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # if we train are training the model, return grid, feats, box_xy, box_wh
    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# -------------------------------------------------------------------
#   Before feeding the image data into network, we use  letter box.
#   So, we need to transform bounding boxes to original shape.
# -------------------------------------------------------------------
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


# ---------------------------------------------------#
#   get boxes location and scores
# ---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, i):

    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(i, feats, anchors, num_classes, input_shape)
    # from input_img coordinate to original_img coordinate
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)

    boxes = K.reshape(boxes, [-1, 4])
    # score = confidence * class_probability
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


# ---------------------------------------------------
# detect the image
# ---------------------------------------------------
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              diou_threshold=.43,
              nms_method='conventional'):
    """ yolo evaluate

    Args:
        yolo_outputs: [batch, 13, 13, 3*85]
        anchors: [9, 2]
        num_classes: num of your own classes
        image_shape: the shape of original image

        max_boxes: need to make it bigger when have a lot of targets in task

        score_threshold: when score > score threshold, the anchor is positive
        iou_threshold: the threshold used in non max suppression
        nms_method
        diou_threshold

    Returns:
        boxes_, scores_, classes_

    """
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]],
                                                    num_classes,
                                                    input_shape,
                                                    image_shape,
                                                    l)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):

        # get positive anchors by using box_scores >= score_threshold
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # NMS
        if nms_method == 'conventional':
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        elif nms_method == 'diou':
            nms_index = diou_nms(class_boxes, class_box_scores, max_boxes_tensor, diou_threshold)

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


# ---------------------------------------------------
# detect the image
# use weighted nms
# ---------------------------------------------------
def yolo_eval_weighted_nms(yolo_outputs,
                           anchors,
                           num_classes,
                           image_shape,
                           score_threshold=.6):
    """ yolo evaluate

    Args:
        yolo_outputs: [batch, 13, 13, 3*85]
        anchors: [9, 2]
        num_classes: num of your own classes
        image_shape: the shape of original image
        score_threshold: when score > score threshold, the anchor is positive

    Returns:
        boxes_, scores_, classes_

    """
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]],
                                                    num_classes,
                                                    input_shape,
                                                    image_shape,
                                                    l)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):

        # get positive anchors by using box_scores >= score_threshold
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    return boxes_, scores_, classes_


if __name__ == '__main__':

    yolo = yolo_body(keras.Input(shape=(416, 416, 3)), 3, 80)
    yolo.summary()

    # yolo.load_weights('../logs/yolo4_weight.h5')
    #
    for layer in yolo.layers[-3:]:
        print(layer.name)
    print(len(yolo.layers))

    # from PIL import Image
    # from utils.utils import letterbox_image
    # import numpy as np

    # street = Image.open('../img/street.jpg')
    # print(street.size)
    #
    # street = letterbox_image(street, (416, 416))
    # street = np.expand_dims(np.array(street), axis=0)
    # street = street / 255.0
    # # street = tf.convert_to_tensor(street, dtype=tf.float32)
    #
    # with K.get_session() as sess:
    #     print(sess.run(yolo.output, {yolo.input: street, K.learning_phase(): 0}))
