import tensorflow as tf
import numpy as np


class Rig2DConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, phi, un_rotate=True, padding='SAME', **kwargs):
        # Non-trainable variables initialization
        self.filters = filters
        self.un_rotate = un_rotate
        self.phi = phi
        self.pad = padding
        self.kernel_size = kernel_size
        self.ks = tf.constant(kernel_size, dtype='float32')
        self.angle = tf.constant(0, dtype='float32')
        self.angle2 = tf.constant(360, dtype='float32')
        self.ang = tf.linspace(self.angle, self.angle2, self.phi + 1)
        self.mid = tf.constant(0.5, dtype='float32')
        self.rds = tf.constant(np.pi / 180, dtype='float32')
        self.center = tf.round((self.ks * self.mid) - self.mid)
        super(Rig2DConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Trainable variables initialization
        self._l = self.add_weight(name='l', shape=(self.filters,), initializer=tf.initializers.constant(value=0.5111867),
                                  trainable=True, constraint=tf.keras.constraints.NonNeg())
        self.alpha = self.add_weight(name='alpha', shape=(self.filters,), initializer=tf.initializers.constant(value=0.5651333),
                                     trainable=True, constraint=tf.keras.constraints.NonNeg())
        self.beta = self.add_weight(name='beta', shape=(self.filters,), initializer=tf.initializers.constant(value=1.4184482),
                                    trainable=True,  constraint=tf.keras.constraints.NonNeg())
        # Input size dependent variables
        self.nf = input_shape[-1]
        self.width = input_shape[-2]
        self.height = input_shape[-3]

        super(Rig2DConv, self).build(input_shape)

    def call(self, x, **kwargs):
        # Generate a set of coordinates
        d = tf.range(self.ks)
        x_coord, y_coord = tf.meshgrid(d, d)
        x_coord = tf.cast(x_coord, dtype="float32") - self.center
        y_coord = tf.cast(y_coord, dtype="float32") - self.center

        # Horizontal basis filter
        arx = tf.math.divide((-2 * self.alpha[0] * tf.square(self._l[0]) * x_coord), np.pi) * tf.exp(
            -self._l[0] * ((self.alpha[0] * tf.square(x_coord)) + (self.beta[0] * tf.square(y_coord))))

        # Vertical basis filter
        ary = tf.math.divide((-2 * self.beta[0] * tf.square(self._l[0]) * y_coord), np.pi) * tf.exp(
            -self._l[0] * ((self.alpha[0] * tf.square(x_coord)) + (self.beta[0] * tf.square(y_coord))))

        arx = tf.expand_dims(arx, axis=-1)
        ary = tf.expand_dims(ary, axis=-1)

        # First filter generation
        ar = (tf.cos(self.ang[0] * self.rds) * arx) + (tf.sin(self.ang[0] * self.rds) * ary)
        # Convolve input with first generated filter
        par1 = tf.nn.conv2d(x, tf.reshape(ar, (self.ks, self.ks, self.nf, 1)), strides=(1, 1, 1, 1), padding=self.pad)
        # Rotation compensation to get translational feature space
        par2 = tf.contrib.image.rotate(par1, self.ang[0] * self.rds, interpolation='BILINEAR')

        # Second filter generation
        ar1 = (tf.cos(self.ang[1] * self.rds) * arx) + (tf.sin(self.ang[1] * self.rds) * ary)
        # Convolve input with second generated filter
        par3 = tf.nn.conv2d(x, tf.reshape(ar1, (self.ks, self.ks, self.nf, 1)), strides=(1, 1, 1, 1), padding=self.pad)
        # Rotation compensation to get translational feature space
        par4 = tf.contrib.image.rotate(par3, self.ang[1] * self.rds, interpolation='BILINEAR')

        if self.un_rotate:
            out = tf.concat([par2, par4], axis=3)
        else:
            out = tf.concat([par1, par3], axis=3)

        # Apply same process from filters up to Phi
        for aa in range(2, self.phi):
            ar = (tf.cos(self.ang[aa] * self.rds) * arx) + (tf.sin(self.ang[aa] * self.rds) * ary)
            partial = tf.nn.conv2d(x, tf.reshape(ar, (self.ks, self.ks, self.nf, 1)), strides=(1, 1, 1, 1),
                                   padding=self.pad)
            partial2 = tf.contrib.image.rotate(partial, self.ang[aa] * self.rds, interpolation='BILINEAR')
            if self.un_rotate:
                out = tf.concat([out, partial2], axis=3)
            else:
                out = tf.concat([out, partial], axis=3)
        out_f1 = tf.reshape(out, shape=(-1, self.height, self.width, self.phi, 1))
        out_f1 = tf.transpose(out_f1, perm=(0, 3, 1, 2, 4))

        # If only one filter ensemble is used return this
        if self.filters == 1:
            return out_f1
        # If more ensembles are required do the same
        arx = tf.math.divide((-2 * self.alpha[1] * tf.square(self._l[1]) * x_coord), np.pi) * tf.exp(
            -self._l[1] * ((self.alpha[1] * tf.square(x_coord)) + (self.beta[1] * tf.square(y_coord))))

        ary = tf.math.divide((-2 * self.beta[1] * tf.square(self._l[1]) * y_coord), np.pi) * tf.exp(
            -self._l[1] * ((self.alpha[1] * tf.square(x_coord)) + (self.beta[1] * tf.square(y_coord))))

        arx = tf.expand_dims(arx, axis=-1)
        ary = tf.expand_dims(ary, axis=-1)

        ar = (tf.cos(self.ang[0] * self.rds) * arx) + (tf.sin(self.ang[0] * self.rds) * ary)
        par1 = tf.nn.conv2d(x, tf.reshape(ar, (self.ks, self.ks, self.nf, 1)), strides=(1, 1, 1, 1), padding=self.pad)
        par2 = tf.contrib.image.rotate(par1, self.ang[0] * self.rds, interpolation='BILINEAR')

        ar1 = (tf.cos(self.ang[1] * self.rds) * arx) + (tf.sin(self.ang[1] * self.rds) * ary)
        par3 = tf.nn.conv2d(x, tf.reshape(ar1, (self.ks, self.ks, self.nf, 1)), strides=(1, 1, 1, 1), padding=self.pad)
        par4 = tf.contrib.image.rotate(par3, self.ang[1] * self.rds, interpolation='BILINEAR')

        if self.un_rotate:
            out = tf.concat([par2, par4], axis=3)
        else:
            out = tf.concat([par1, par3], axis=3)

        for aa in range(2, self.phi):
            ar = (tf.cos(self.ang[aa] * self.rds) * arx) + (tf.sin(self.ang[aa] * self.rds) * ary)
            partial = tf.nn.conv2d(x, tf.reshape(ar, (self.ks, self.ks, self.nf, 1)), strides=(1, 1, 1, 1),
                                   padding=self.pad)
            partial2 = tf.contrib.image.rotate(partial, self.ang[aa] * self.rds, interpolation='BILINEAR')
            if self.un_rotate:
                out = tf.concat([out, partial2], axis=3)
            else:
                out = tf.concat([out, partial], axis=3)
        out_f2 = tf.reshape(out, shape=(-1, self.height, self.width, self.phi, 1))
        out_f2 = tf.transpose(out_f2, perm=(0, 3, 1, 2, 4))

        out_final = tf.concat([out_f1, out_f2], axis=4)

        for bb in range(2, self.filters):

            arx = tf.math.divide((-2 * self.alpha[bb] * tf.square(self._l[bb]) * x_coord), np.pi) * tf.exp(
                -self._l[bb] * ((self.alpha[bb] * tf.square(x_coord)) + (self.beta[bb] * tf.square(y_coord))))

            ary = tf.math.divide((-2 * self.beta[bb] * tf.square(self._l[bb]) * y_coord), np.pi) * tf.exp(
                -self._l[bb] * ((self.alpha[bb] * tf.square(x_coord)) + (self.beta[bb] * tf.square(y_coord))))

            arx = tf.expand_dims(arx, axis=-1)
            ary = tf.expand_dims(ary, axis=-1)

            ar = (tf.cos(self.ang[0] * self.rds) * arx) + (tf.sin(self.ang[0] * self.rds) * ary)
            par1 = tf.nn.conv2d(x, tf.reshape(ar, (self.ks, self.ks, self.nf, 1)), strides=(1, 1, 1, 1),
                                padding=self.pad)
            par2 = tf.contrib.image.rotate(par1, self.ang[0] * self.rds, interpolation='BILINEAR')

            ar1 = (tf.cos(self.ang[1] * self.rds) * arx) + (tf.sin(self.ang[1] * self.rds) * ary)
            par3 = tf.nn.conv2d(x, tf.reshape(ar1, (self.ks, self.ks, self.nf, 1)), strides=(1, 1, 1, 1),
                                padding=self.pad)
            par4 = tf.contrib.image.rotate(par3, self.ang[1] * self.rds, interpolation='BILINEAR')

            if self.un_rotate:
                out = tf.concat([par2, par4], axis=3)
            else:
                out = tf.concat([par1, par3], axis=3)

            for aa in range(2, self.phi):
                ar = (tf.cos(self.ang[aa] * self.rds) * arx) + (tf.sin(self.ang[aa] * self.rds) * ary)
                partial = tf.nn.conv2d(x, tf.reshape(ar, (self.ks, self.ks, self.nf, 1)), strides=(1, 1, 1, 1),
                                       padding=self.pad)
                partial2 = tf.contrib.image.rotate(partial, self.ang[aa] * self.rds, interpolation='BILINEAR')
                if self.un_rotate:
                    out = tf.concat([out, partial2], axis=3)
                else:
                    out = tf.concat([out, partial], axis=3)
            out_fn = tf.reshape(out, shape=(-1, self.height, self.width, self.phi, 1))
            out_fn = tf.transpose(out_fn, perm=(0, 3, 1, 2, 4))

            out_final = tf.concat([out_final, out_fn], axis=4)

        return out_final

    def get_config(self):
        base_config = super(Rig2DConv, self).get_config()
        base_config['filters'] = self.filters
        base_config['kernel_size'] = self.kernel_size
        base_config['phi'] = self.phi
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# A cyclic convolution can be implemented with a linear convolution over a periodically padded feature space
class Periodic_Pad(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Periodic_Pad, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Periodic_Pad, self).build(input_shape)

    def call(self, inputs):
        x = tf.concat([inputs, inputs], axis=1)
        return x[:, :-1, :, :, :]

    def get_config(self):
        base_config = super(Periodic_Pad, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)