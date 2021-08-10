# Keras built-in image processing layer for image data augmentaiton 

import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental.preprocessing import (RandomFlip, 
                                                                RandomRotation, 
                                                                RandomTranslation, 
                                                                RandomContrast,
                                                                RandomCrop,
                                                                Rescaling,
                                                                RandomZoom)

class RandomInvert(tf.keras.layers.Layer):
    def __init__(self, prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
    def call(self, inputs, training=True):
        if tf.random.uniform([]) < self.prob:
            return tf.cast(255.0 - inputs, dtype=tf.float32)
        else: 
            return tf.cast(inputs, dtype=tf.float32)
    def get_config(self):
        config = {
            'prob': self.prob,
        }
        base_config = super(RandomInvert, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class RandomEqualize(tf.keras.layers.Layer):
    def __init__(self, prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
    def call(self, inputs, training=True):
        if tf.random.uniform([]) < self.prob:
            return tf.cast(tfa.image.equalize(inputs), dtype=tf.float32)
        else: 
            return tf.cast(inputs, dtype=tf.float32)
    def get_config(self):
        config = {
            'prob': self.prob,
        }
        base_config = super(RandomEqualize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class RandomCutout(tf.keras.layers.Layer):
    def __init__(self, prob=0.5, mask_size=(20, 20), replace=0, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob 
        self.replace = replace
        self.mask_size = mask_size
    def call(self, inputs, training=True):
        if tf.random.uniform([]) < self.prob:
            inputs = tfa.image.random_cutout(inputs,
                                           mask_size=self.mask_size,
                                           constant_values=self.replace)  
            return tf.cast(inputs, dtype=tf.float32)
        else: 
            return tf.cast(inputs, dtype=tf.float32)
    def get_config(self):
        config = {
            'prob': self.prob,
            'replace': self.replace,
            'mask_size': self.mask_size
        }
        base_config = super(RandomCutout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
        
#  keras augmentation
keras_image_augmentation = tf.keras.Sequential(
    [
        RandomFlip("horizontal", dtype=tf.float32),
        RandomRotation(factor=0.01, dtype=tf.float32),
        RandomTranslation(height_factor=0.0, width_factor=0.1, dtype=tf.float32),
        RandomZoom(height_factor=(-0.1, -0.2), width_factor=(-0.1, -0.2)),
        RandomInvert(prob=0.2),
        RandomCutout(prob=0.8, replace=0, 
                     mask_size=(int(input_height * 0.2),
                                int(input_width * 0.2))),
        RandomEqualize(prob=0.5)
    ],
    name='keras_augment_layers'
)

class Keras3DAugmentation(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.random_flip = RandomFlip("horizontal", dtype=tf.float32)
        self.random_rotate = RandomRotation(factor=0.01, dtype=tf.float32)
        self.random_translation = RandomTranslation(height_factor=0.0, width_factor=0.1, dtype=tf.float32)
        self.random_cutout = RandomCutout(prob=0.8, replace=0, 
                                          mask_size=(int(input_height * 0.1),
                                                     int(input_width * 0.1)))
        self.random_equalize = RandomEqualize(prob=0.5)
        self.random_invert = RandomInvert(prob=0.2)
        
    def call(self, inputs, training=True):
        splitted_modalities = tf.split(tf.cast(inputs, tf.float32), input_channel, axis=-1)
        splitted_modalities = [tf.squeeze(i, axis=-1) for i in splitted_modalities] 
        
        
        flair = []
        t1w = []
        t1wce = []
        t2w = []
        
        for j, each_modality in enumerate(splitted_modalities):
            x = self.random_flip(each_modality)
            x = self.random_rotate(x)
            x = self.random_translation(x)
            x = self.random_cutout(x)
            x = self.random_invert(x)
            
            if j == 0:
                flair.append(tf.expand_dims(x, axis=-1))
            elif j == 1:
                t1w.append(tf.expand_dims(x, axis=-1))
            elif j == 2:
                t1wce.append(tf.expand_dims(x, axis=-1))
            elif j == 3:
                t2w.append(tf.expand_dims(x, axis=-1))
                
        image = tf.stack([flair, t1w, t1wce, t2w], axis=-1)
        image = tf.reshape(image, [-1, input_height, input_width, input_depth, input_channel])
        return image