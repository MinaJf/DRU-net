import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers


class DRUNet2D(Model):
    def __init__(self, n_class, n_layer, root_filters, kernal_size=3, pool_size=2, use_bn=True, use_res=True,
                 padding='SAME', concat_or_add='concat'):
        super().__init__()
        self.dw_layers = dict()
        self.up_layers = dict()
        self.max_pools = dict()
        self.dw1_layers = dict()

        for layer in range(n_layer):
            filters = 2 ** layer * root_filters
            dict_key = str(n_layer - layer - 1)
            if layer == 0:
               Dw1 = _DownSev(filters, 7, 'dw1_%d' % layer)
               self.dw1_layers[dict_key] = Dw1
            dw = _DownSampling(filters, kernal_size, 'dw_%d' % layer, use_bn, use_res)
            self.dw_layers[dict_key] = dw
            if layer < n_layer - 1:
                pool = layers.MaxPool2D([2, 2], padding=padding)
                self.max_pools[dict_key] = pool

        for layer in range(n_layer - 2, -1, -1):
            filters = 2 ** (layer + 1) * root_filters
            dict_key = str(n_layer - layer - 1)
            up = _UpSampling(filters, kernal_size, [2, 2], concat_or_add)
            self.up_layers[dict_key] = up

        stddev = np.sqrt(2 / (kernal_size ** 2 * root_filters))
        self.conv_out = layers.Conv2D(n_class, 1, padding=padding, use_bias=False,
                                      kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                      name='conv_out')

    @tf.function
    def __call__(self, x_in, drop_rate, training):
        dw_tensors = dict()
        dw1_tensors = dict()
        x = x_in
        dict_key = str(4)
        dw1_tensors[dict_key] = self.dw1_layers[dict_key](x, drop_rate, training)
        x = dw1_tensors[dict_key]
        n_layer = len(self.dw_layers)
        for i in range(n_layer):
            dict_key = str(n_layer - i - 1)
            dw_tensors[dict_key] = self.dw_layers[dict_key](x, drop_rate, training)
            x = dw_tensors[dict_key]
            if i < len(self.max_pools):
                x = self.max_pools[dict_key](x)

        for i in range(n_layer - 2, -1, -1):
            dict_key = str(n_layer - i - 1)
            x = self.up_layers[dict_key](x, dw_tensors[dict_key], drop_rate, training)

        x = self.conv_out(x)
        x = tf.nn.relu(x)
        return x


class _DownSev(Model):
    def __init__(self, filters, kernel_size, name, use_bn=True, use_res=True, padding='SAME', use_bias=True):
        super(_DownSev, self).__init__(name=name)
        stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
        self.conv1 = layers.Conv2D(filters, 7, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv2')

    def call(self, input_tensor, keep_prob, train_phase):
        # conv1
        x = self.conv1(input_tensor)
        x = tf.nn.dropout(x, keep_prob)

        return x


class _DownSampling(layers.Layer):
    def __init__(self, filters, kernel_size, name, use_bn=True, use_res=True, padding='SAME', use_bias=True):
        super().__init__(name=name)
        self.use_bn = use_bn
        self.use_res = use_res
        stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
        self.conv1 = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv1')
        self.conv2 = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv2')

        if use_bn:
            self.bn1 = layers.BatchNormalization(momentum=0.99, name='bn1')
            self.bn2 = layers.BatchNormalization(momentum=0.99, name='bn2')
        if use_res:
            self.res = _Residual('res')

    def __call__(self, x_in, drop_rate, training=True):
        # conv1

        x = x_in
        x = self.conv1(x)
        x = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x1 = self.bn1(x, training=training)
        x = tf.nn.relu(x1)

        # conv2
        x = self.conv2(x)
        x = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x = self.bn2(x, training=training)

        x = x + x1 + x_in
        x = tf.nn.relu(x)
        x = tf.concat((x, x_in), -1)

        return x


class _UpSampling(layers.Layer):
    def __init__(self, filters, kernel_size, pool_size, name, concat_or_add='concat', use_bn=True, use_res=True,
                 padding='SAME', use_bias=True):
        super().__init__(name=name)
        self.use_bn = use_bn
        self.use_res = use_res
        self.concat_or_add = concat_or_add
        stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
        self.deconv = layers.Conv2DTranspose(filters // 2, kernel_size, strides=pool_size, padding=padding,
                                             use_bias=use_bias,
                                             kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                             name='deconv')
        self.conv1 = layers.Conv2D(filters // 2, kernel_size, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv1')
        self.conv2 = layers.Conv2D(filters // 2, kernel_size, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv2')
        self.conv3 = layers.Conv2D(filters //2, 1, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv3')

        if use_bn:
            self.bn_deconv = layers.BatchNormalization(momentum=0.99, name='bn_deconv')
            self.bn1 = layers.BatchNormalization(momentum=0.99, name='bn1')
            self.bn2 = layers.BatchNormalization(momentum=0.99, name='bn2')
            self.bn3 = layers.BatchNormalization(momentum=0.99, name='bn3')
        if use_res:
            self.res = _Residual('res')

    def __call__(self, x_in, x_dw, drop_rate, training):
        # deconv
        x = self.deconv(x_in)
        if self.use_bn:
            x = self.bn_deconv(x, training=training)
        x = tf.nn.relu(x)

        # skip connection
        if self.concat_or_add == 'concat':
            x = tf.concat((x_dw, x), -1)
        elif self.concat_or_add == 'add':
            x = x_dw + x
        else:
            raise Exception('Wrong concatenate method!')

        res_in = x
        # conv1
        x = self.conv1(x)
        x = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x1 = self.bn1(x, training=training)
        x = tf.nn.relu(x1)

        # conv2
        x = self.conv2(x)
        x = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x = self.bn2(x, training=training)

        x3 = self.conv3(res_in)
        x3 = self.bn3(x3, training=training)

        x = x + x1 + x3
        x = tf.nn.relu(x)

        return x


class _Residual(layers.Layer):
    def __init__(self, name_scope):
        super(_Residual, self).__init__(name=name_scope)

    def call(self, x1, x2):
        if x1.shape[-1] < x2.shape[-1]:
            x = tf.concat([x1, tf.zeros(list(x1.shape[:-1]) + [x2.shape[-1] - x1.shape[-1]])], axis=-1)
        else:
            x = x1[..., :x2.shape[-1]]
        x = x + x2
        return x