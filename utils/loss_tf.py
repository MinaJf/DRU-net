import time
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.image import extract_patches_2d
import cv2
# loss functions -------------------------------------------
def cross_entropy(data_dict):
    logits = data_dict['logits']
    labels = data_dict['labels']
    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(loss_map)

def dice_coefficient(data_dict):
    logits = tf.cast(data_dict['logits'], tf.float32)
    labels = tf.cast(data_dict['labels'], tf.float32)
    axis = tuple(range(1, len(labels.shape) - 1)) if len(labels.shape) > 1 else -1
    pred = tf.nn.softmax(logits)
    # pred = tf.one_hot(tf.argmax(logits, -1), labels.shape[-1])
    
    intersection = tf.reduce_sum(pred * labels, axis)
    sum_ = tf.reduce_sum(pred + labels, axis)
    dice = 1 - 2 * intersection / sum_
    return dice
