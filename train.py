import tensorflow as tf 
import tensorflow.keras import Model, Input
import tensorflow.keras import layers 
import numpy as np 
import pandas as pd 
import os, random 

import config 
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
# seed all
seed_all(config.seed)

df = pd.read_csv(config.train_csv_path)

def fold_generator(fold):
    # for way one - data generator
    train_labels = df[df.fold != fold].reset_index(drop=True)
    val_labels = df[df.fold == fold].reset_index(drop=True)
    
    return (
        BrainTumorGenerator(config.train_sample_path, train_labels),
        BrainTumorGenerator(config.train_sample_path, val_labels)
    )

# first fold 
train_gen, val_gen = fold_generator(config.fold)


if config.modeling_in == '2D':
    train_data = tf.data.Dataset.from_generator(
        lambda: map(tuple, train_gen),
        (tf.float32, tf.float32),
        (
            tf.TensorShape([config.input_height, config.input_width, config.input_depth]),
            tf.TensorShape([]),
        ),
    )

    val_data = tf.data.Dataset.from_generator(
        lambda: map(tuple, val_gen),
        (tf.float32, tf.float32),
        (
            tf.TensorShape([config.input_height, config.input_width, config.input_depth]),
            tf.TensorShape([]),
        ),
    )
else:
    train_data = tf.data.Dataset.from_generator(
        lambda: map(tuple, train_gen),
        (tf.float32, tf.float32),
        (
            tf.TensorShape([config.input_height, config.input_width, config.input_depth, config.input_channel]),
            tf.TensorShape([]),
        ),
    )

    val_data = tf.data.Dataset.from_generator(
        lambda: map(tuple, val_gen),
        (tf.float32, tf.float32),
        (
            tf.TensorShape([config.input_height, config.input_width, config.input_depth, config.input_channel]),
            tf.TensorShape([]),
        ),
    )

class TFDataGenerator:
    def __init__(self, data, modeling_in, 
                 shuffle, aug_lib,
                batch_size, rescale, rescaled_offset):
        if modeling_in not in ['2D', '3D']:
            raise ValueError('volume is not set either 2D or 3D')
        self.data = data                  # data files 
        self.modeling_in = modeling_in    # 2D or 3D 
        self.shuffle = shuffle            # true for training 
        self.aug_lib = aug_lib            # type of augmentation library 
        self.batch_size = batch_size      # batch size number 
        self.rescale = rescale                  # normalize or not 
        self.rescaled_offset = rescaled_offset  # normalize in (-1, 1) or (0, 1)
        
    # a convinient function to get 2D data set 
    def get_2D_data(self):
        self.data = self.data.shuffle(buffer_size = self.batch_size * 100) \
                                                        if self.shuffle else self.data
        
        # applied augmentations based on condition 
        if self.aug_lib == 'albumentation' and self.shuffle:
            # known issue: only support 3 channel 
            # it would be tricky to use mask to add additional 3 chnnel, total 6 
            self.data = self.data.map(partial(albu_image_process_aug), 
                                      num_parallel_calls=AUTO).batch(self.batch_size, 
                                                                     drop_remainder=self.shuffle)
            
        elif self.aug_lib == 'tf' and self.shuffle:
            # augmentaiton using tf.image.stateless_random* functions 
            # applicable for multiple channel (channel > 3 )
            # same augmentation would be applied to each modalites: flair, t1w, t1wce, t2w
            # check the tf_image_augmentation code for details 
            self.data = self.data.map(lambda x, y: (tf_image_augmentation(x), y),
                                      num_parallel_calls=AUTO).batch(self.batch_size,
                                                                     drop_remainder=self.shuffle)
        elif self.aug_lib == 'keras' and self.shuffle:
            # augmentaion using keras image preprocessing layers 
            # applicable for multiple channels (channel > 3)
            # [known issue]:  same augmentaion would be applied for all modalites 
            # check the tf_image_augmentation code for details  
            self.data = self.data.batch(self.batch_size, drop_remainder=self.shuffle) 
            self.data = self.data.map(lambda x, y: (keras_image_augmentation(x, training=True), y), 
                                      num_parallel_calls=AUTO)
        elif not self.shuffle:
            # no shuffle generally assuming no augmentaion too, so not training 
            # for inference and evaluation 
            self.data = self.data.batch(self.batch_size, drop_remainder=self.shuffle) 
            
        # TO DO
        if self.modeling_in == '2D+3D':
            # here we expand 1 channel axis for 2D modeling in case to run on 3D model 
            self.data = self.data.map(lambda x, y: (tf.expand_dims(x, axis=-1), y),
                                      num_parallel_calls=AUTO)
            
        # rescaling the data for faster convergence 
        if self.rescale:    
            if self.rescaled_offset: # within range (-1, 1)
                self.data = self.data.map(lambda x, y: (Rescaling(scale=1./127.5, offset=-1)(x), y), 
                                          num_parallel_calls=AUTO)
            else: # within range (0, 1)
                self.data = self.data.map(lambda x, y: (Rescaling(scale=1./255, offset=0.0)(x), y), 
                                          num_parallel_calls=AUTO)
            
        # prefetching the data 
        return self.data.prefetch(AUTO) 
            
    # a convinient function to get 3D data set 
    def get_3D_data(self):
        # augmentation on 3D data set
        # volumentation is based on albumentation 
        if self.aug_lib == 'volumentations' and self.shuffle:
            # if true, augmentation would be applied separately for each depth 
            self.data = self.data.map(partial(volumentations_aug), num_parallel_calls=AUTO)
            self.data = self.data.batch(batch_size, drop_remainder=self.shuffle)
        else:
            # true for evaluation and inference, no augmentation 
            self.data = self.data.batch(self.batch_size, drop_remainder=self.shuffle)
            
        # prefetching the data 
        return self.data.prefetch(AUTO) 



tf_gen = TFDataGenerator(train_data,
                         modeling_in=modeling_in, # choose modeling approach in 3D or 2D 
                         shuffle=True,     # shuffling the data for train set
                         aug_lib='tf',     # set: 'tf' and 'albumentation' and 'keras'
                         batch_size=32,    # batch size of model training
                         rescale=False,     # if true, data would be divided by 255.0
                         rescaled_offset=False)  # if 0: normalize data [0, 1], if 1: normalize data [-1, 1]

train_generator = tf_gen.get_2D_data()


tf_gen = TFDataGenerator(val_data,
                         modeling_in=modeling_in, # choose modeling approach in 3D or 2D 
                         shuffle=False,     # shuffling the data for train set
                         aug_lib=None,     # set: 'tf' and 'albumentation' and 'keras'
                         batch_size=batch_size,    # batch size of model training
                         rescale=False,     # if true, data would be divided by 255.0
                         rescaled_offset=False)  # if ture: [-1, 1], else [-1, 1]

valid_generator = tf_gen.get_2D_data()


from tensorflow.keras import Input, Model 
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import *

tf.keras.backend.clear_session()
input_dim = (input_height, input_width, input_depth)
input_tensor = Input(input_dim)
x = Conv2D(3, (3, 3), padding='same', use_bias=False)(input_tensor)
efnet = EfficientNetB2(weights='imagenet', 
                       include_top = False, 
                       input_shape=(input_height, 
                                    input_width, 3))
curr_output = efnet(x)
curr_output = GlobalAveragePooling2D()(curr_output)
output = Dense(1, activation='sigmoid') (curr_output)
model = Model(input_tensor, output)

model.compile(
    loss=tfa.losses.SigmoidFocalCrossEntropy(gamma = 2.0, alpha = 0.80),
    optimizer=opt,
    metrics=[tf.keras.metrics.AUC(), 
         tf.keras.metrics.BinaryAccuracy(name='bacc')]
)
    
def get_lr_callback(batch_size=8):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    return lr_callback


reduceLROnPlat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc',
                                                      factor=0.3, patience=2,
                                                      verbose=1, mode='max',
                                                      epsilon=0.0001, cooldown=1,
                                                      min_lr=0.00001)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath="model.{epoch:02d}-{val_auc:.4f}.h5", 
    monitor='val_auc', mode='max', 
    save_best_only=True, verbose=1
)

# Train the model, doing validation at the end of each epoch
epochs = 15
model.fit(
    train_generator, # train_generator
    epochs=epochs,
    validation_data=valid_generator, # valid_generator
    callbacks=[checkpoint_cb, get_lr_callback(batch_size)],  #   get_lr_callback(batch_size)
    workers=psutil.cpu_count()
)



