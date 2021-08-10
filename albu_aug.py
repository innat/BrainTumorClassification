import matplotlib.pyplot as plt 
import random 
from functools import partial
import albumentations as A 
 
transforms =  A.Compose([
    A.OneOf([
        A.CLAHE(clip_limit=4.0, p=0.7),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, 
                   hue=0.2, p=0.7)
    ], p=1.0),

    A.CoarseDropout(max_holes=3, 
                    max_height=int(input_height * 0.1),
                    max_width=int(input_width * 0.1), p=0.5),
    
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.5),
    A.RandomGamma(gamma_limit=(60, 120), p=0.6),
    A.PiecewiseAffine(p=0.4),
], p=1.)


def albu_image_process_aug(image, label):
    # for CLAHE. Do we have to do this? 
    image = tf.cast(image, tf.uint8) 
    
    def aug_fn(image):
        aug_data = transforms(**{"image":image})
        aug_img = aug_data["image"]
        return tf.cast(aug_img, tf.float32)
    
    # Wraps a python function and uses it as a TensorFlow op.
    aug_img = tf.numpy_function(func=aug_fn, 
                                inp=[image], 
                                Tout=tf.float32)
    aug_img.set_shape((input_height, input_width, input_depth))
    return aug_img, label
