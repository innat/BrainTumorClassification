import tensorflow as tf 
from tensorflow.keras.layers import * 
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

class Conv3DBatchNorm(tf.keras.layers.Layer):
    def __init__(self, nb_filters, kernel_size, padding, strides):
        super(Conv3DBatchNorm, self).__init__()
        # parameters 
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size 
        self.padding = padding 
        self.strides = strides 

        # layers
        self.conv = tf.keras.layers.Conv3D(self.nb_filters, self.kernel_size, 
                                           self.strides, self.padding)
        self.bn   = tf.keras.layers.BatchNormalization()
        
    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)
    
    def get_config(self):
        return {
            'nb_filters': self.nb_filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'strides': self.strides
        }


class Inception3DModule(tf.keras.layers.Layer):
    def __init__(self, nb_filters, kernel_size1x1, kernel_size3x3):
        super(Inception3DModule, self).__init__()
        # params 
        self.nb_filters = nb_filters
        self.kernel_size1x1 = kernel_size1x1
        self.kernel_size3x3 = kernel_size3x3

        # layers 
        self.conv1 = Conv3DBatchNorm(self.nb_filters, kernel_size=self.kernel_size1x1,
                                     strides=1, padding='same')
        self.conv2 = Conv3DBatchNorm(self.nb_filters, kernel_size=self.kernel_size3x3, 
                                     strides=1, padding='same')
        self.cat   = tf.keras.layers.Concatenate()

    def call(self, input_tensor, training=False):
        x_1x1 = self.conv1(input_tensor)
        x_3x3 = self.conv2(input_tensor)
        x = self.cat([x_1x1, x_3x3])
        return tf.nn.relu(x) 

    def get_config(self):
        return {
            'nb_filters': self.nb_filters,
            'kernel_size1x1': self.kernel_size1x1,
            'kernel_size3x3': self.kernel_size3x3
        }


class Identity3DBlock(tf.keras.layers.Layer):
    def __init__(self, nb_filters, kernel_size, padding, strides, shortcut = False):
        super(Identity3DBlock, self).__init__()
        # params 
        self.shortcut = shortcut 
        self.nb_filters = nb_filters 
        self.kernel_size = kernel_size 
        self.padding = padding 
        self.strides = strides 
        
        # layers 
        self.conv1 = Conv3DBatchNorm(self.nb_filters, self.kernel_size, 
                                     self.padding, self.strides)
        self.conv2 = Conv3DBatchNorm(self.nb_filters, self.kernel_size, 
                                     self.padding, self.strides)
        self.conv3 = Conv3DBatchNorm(self.nb_filters, self.kernel_size,
                                     self.padding, self.strides)
        self.inception = Inception3DModule(self.nb_filters, 
                                           kernel_size1x1 = (1,1,1),
                                           kernel_size3x3 = (3,3,3))
    
    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        if self.shortcut:
            srtct = self.inception(input_tensor)
            srtct = self.conv3(srtct)
            x = Dropout(0.5)(x)
            x = Add()([x, srtct])
            return tf.nn.relu(x)
        else:
            x = Add()([x, input_tensor])
            return tf.nn.relu(x)
        
    def get_config(self):
        return {
            'shortcut': self.shortcut,
            'nb_filters': self.nb_filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'strides': self.strides,
        }


def InceptionResNet3D(width=128, height=128, depth=32, num_cls=1):
    inpt = Input((height, width, depth, 4), name='input3D')
    
    # conv3d + relu + maxplo3d 
    x = ZeroPadding3D((1, 1, 1))(inpt)
    x = Conv3DBatchNorm(nb_filters=16, kernel_size=(3, 3, 3), strides=1, padding='valid')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=1)(x)

    # inception_resnet_block_ab
    x = Identity3DBlock(nb_filters=16, kernel_size=(2, 2, 2), 
                        padding='same', strides=1, shortcut=True)(x)
    x = Identity3DBlock(nb_filters=16, kernel_size=(2, 2, 2), 
                        padding='same', strides=1)(x)
    x = AveragePooling3D(pool_size=(2, 2, 2))(x)
    x = Identity3DBlock(nb_filters=32, kernel_size=(3, 3, 3), 
                        padding='same', strides=1, shortcut=True)(x)
    x = Identity3DBlock(nb_filters=32, kernel_size=(3, 3, 3), 
                        padding='same', strides=1)(x)
    
    x = GlobalAveragePooling3D()(tf.nn.relu(x))
    x = Dense(num_cls, activation='sigmoid')(x)

    model = Model(inputs=inpt, outputs=x)
    return model