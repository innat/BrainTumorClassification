
import config
import tensorflow as tf 

# tf.random.stateless_* functions for image processing 
# using here for image data augmentation 

def tf_image_augmentation(image):  
    # splitted based on modalites. we have 4 type of modalities. Input shape (h, w, depth, channel==4)
    if config.modeling_in == '3D':
        # input shape: image.shape -> (h, w, input_depth, input_channel)
        # in such condition, we split based on the number of channel, which is 4 here. 
        # after that the variable will contains 4 splitted tensor 
        # each of which is in a shape of (h, w, input_depth, 1)
        splitted_modalities = tf.split(tf.cast(image, tf.float32), config.input_channel, axis=-1) # input_channel, input_depth
    elif config.modeling_in == '2D':
        # input shape: image.shape -> (h, w, input_depth * input_channel)
        # in such condition, we split based on the number of channel, which is 4 here.
        # after that the variable will contains 4 splitted tensor 
        # each of which is in a shape of (h, w, input_depth)
        splitted_modalities = tf.split(tf.cast(image, tf.float32), config.input_channel, axis=-1) # input_channel, input_depth
    
    
    # augmented frames for 2d modeling 
    # for 2d modeling we use 1 container to gather all augmented samples from 4 modalites
    # however, same augmentation is ensured for each modality for one study
    augment_img = []
    
    # augmented frames for 3d modeling 
    # for 3d modeling we use 4 container to gather all augmented samples from 4 modalites
    # however, same augmentation is ensured for each modality for one study
    flair_augment_img = []
    t1w_augment_img = []
    t1wce_augment_img = []
    t2w_augment_img = []
    
    
    if config.modeling_in == '3D':
        # remove the last axis.
        # input: (h, w, input_depth, 1) : output: (h, w, input_depth)
        splitted_modalities = [tf.squeeze(i, axis=-1) for i in splitted_modalities] 
    
    # iterate over each modalities, e.g: flair, t1w, t1wce, t2w
    for j, modality in enumerate(splitted_modalities):
        # now splitting each frame from one modality 
        splitted_frames = tf.split(tf.cast(modality, tf.float32), modality.shape[-1], axis=-1)
        
        # iterate over each frame to conduct same augmentation on 
        # each frame 
        for i, img in enumerate(splitted_frames):
            # Given the same seed, they return the same results independent of 
            # how many times they are called.
            # seed is a Tensor of shape (2,) whose values are any integers.
            # for some operation we don't need channel == 3, just 1 is enough 
            img = tf.image.stateless_random_flip_left_right(img, seed = (j, 2))
            img = tf.image.stateless_random_flip_up_down(img, seed = (j, 2))
            img = tf.image.stateless_random_contrast(img, 0.2, 0.8, seed = (j, 2))
            img = tf.image.stateless_random_brightness(img, 0.2, seed = (j, 2))
            
            # some operation require channel == 3 
            img = tf.image.stateless_random_saturation(tf.image.grayscale_to_rgb(img), 0.9, 1.6, seed = (j, 2))
            img = tf.image.stateless_random_hue(img, 0.3, seed = (j, 2))

            # for some operation we don't need channel == 3, just 1 is enough 
            img = tf.image.rgb_to_grayscale(img)
            img = tf.cast(
                tf.image.stateless_random_jpeg_quality(
                    tf.cast(img, tf.uint8), 
                    min_jpeg_quality=20, max_jpeg_quality=40, seed = (j, 2)
                ), tf.float32)

            # ensuring same augmentation for each modalities 
            tf.random.set_seed(j)
            if tf.random.uniform((), seed=j) > 0.7:
                np.random.seed(j)
                kimg = np.random.choice([1,2,3,4])
                kgma = np.random.choice([0.7, 0.9, 1.2])
                img = tf.image.rot90(img, k=kimg)
                img = tf.image.adjust_gamma(img, gamma=kgma)
            

            #mask_size should be divisible by 2. 
            #src: https://www.tensorflow.org/addons/api_docs/python/tfa/image/random_cutout#args
#             tf.random.set_seed(j)
#             if tf.random.uniform((), seed=j) > 0.6:
#                 img = tfa.image.random_cutout(tf.expand_dims(img, 0),
#                                               mask_size=(int(input_height * 0.2),
#                                                          int(input_width * 0.2)), 
#                                               constant_values=0) 
#                 img = tf.squeeze(img, axis=0)
            
            # gathering all frames 
            if config.modeling_in == '3D':
                if j == 0: # 1st modality 
                    flair_augment_img.append(img)
                elif j == 1: # 2nd modality 
                    t1w_augment_img.append(img)
                elif j == 2: # 3rd modality 
                    t1wce_augment_img.append(img)
                elif j == 3:  # 4th modality 
                    t2w_augment_img.append(img)
            elif config.modeling_in == '2D':
                augment_img.append(img)
      
    
    if config.modeling_in == '3D':
        image = tf.transpose([flair_augment_img, t1w_augment_img, t1wce_augment_img, t2w_augment_img])
        image = tf.reshape(image, [config.input_height, config.input_width, config.input_depth, config.input_channel])
    elif config.modeling_in == '2D':
        image = tf.concat(augment_img, axis=-1)
        image = tf.reshape(image, [config.input_height, config.input_width, config.input_depth*config.input_channel])
    return image