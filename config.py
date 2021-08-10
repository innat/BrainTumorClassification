
seed = 2021
fold = 0
batch_size = 6
modeling_in = '3D' # either '2D' or '3D' modeling,  
aug_lib='tf' # set: 2D: 'albumentation', `tf`, `keras`... 3D: 'volumentations'

input_height = 120
input_width = 120

input_modality = ["FLAIR"] # "FLAIR","T1w", "T1wCE", "T2w"

input_channel =  len(input_modality)  # total number of channel 
input_depth = 30 # how many sample would be piced from each modality 

train_sample_path = '../input/rsna-miccai-brain-tumor-radiogenomic-classification/train'
train_csv_path = '../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv'

