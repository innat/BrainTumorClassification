import tensorflow as tf 
import cv2, glob, random 
AUTO = tf.data.AUTOTUNE

class BrainTumorGenerator(tf.keras.utils.Sequence):
    def __init__(self, dicom_path, data, is_train=True):
        self.is_train = is_train # to control training/validation/inference part 
        self.data = data
        self.dicom_path = dicom_path
        self.label = self.data['MGMT_value']
  
    def __len__(self):
        return len(self.data['BraTS21ID'])
    
    def __getitem__(self, index):
        patient_ids = f"{self.dicom_path}/{str(self.data['BraTS21ID'][index]).zfill(5)}/"
   
        # for 2D modeling 
        channel = []
    
        # for 3D modeling 
        flair = []
        t1w = []
        t1wce = []
        t2w = [] 
        
        for m, t in enumerate(input_modality):  # "FLAIR", "T1w", "T1wCE", "T2w"
            t_paths = sorted(
                glob.glob(os.path.join(patient_ids, t, "*")), 
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            
            # pick input_depth times slices 
            # from middle range possible 
            strt_idx = (len(t_paths) // 2) - (input_depth // 2)
            end_idx = (len(t_paths) // 2) + (input_depth // 2)
            # slicing extracting elements with 2 intervals 
            picked_slices = t_paths[strt_idx:end_idx:1]
            
            # removing black borders 
            # and add multi-modal features maps / channel depth
            for i in picked_slices:
                image = self.read_dicom_xray(i)
                j = 0
                while True:
                    # if it's a black image, try to pick any random slice of non-black  
                    # otherwise move on with black image. 
                    if image.mean() == 0:
                        # do something 
                        image = self.read_dicom_xray(random.choice(t_paths)) 
                        j += 1
                        if j == 100:
                            break
                    else:
                        break
                        
                # remove black borders from ROI 
                rows = np.where(np.max(image, 0) > 0)[0]
                cols = np.where(np.max(image, 1) > 0)[0]
                if rows.size:
                    image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
                else:
                    image = image[:1, :1]
           
                # for modeling in 3D 
                if modeling_in == '3D':
                    if m == 0:
                        flair.append(cv2.resize(image, (input_height, input_width)))
                    elif m == 1:
                        t1w.append(cv2.resize(image, (input_height, input_width)))
                    elif m == 2:
                        t1wce.append(cv2.resize(image, (input_height, input_width)))
                    elif m == 3:
                        t2w.append(cv2.resize(image, (input_height, input_width)))
                elif modeling_in == '2D':
                    channel.append(cv2.resize(image, (input_height, input_width)))
                    

        if modeling_in == '3D':
            # (None, h, w, depth, channel)
            # it's possible that with current set up, all modalities don't have same number of slice
            # in that case, append the existing slice with small variation. 
            # for flair 
            while True:
                if len(flair) < input_depth and flair:
                    flair.append(cv2.convertScaleAbs(random.choice(flair), alpha=1.1, beta=0))
                else:
                    break
            
            # for t1w
            while True:
                if len(t1w) < input_depth and t1w:
                    t1w.append(cv2.convertScaleAbs(random.choice(t1w), alpha=1.1, beta=0))
                else:
                    break

            # for t1wce
            while True:
                if len(t1wce) < input_depth and t1wce:
                    t1wce.append(cv2.convertScaleAbs(random.choice(t1wce), alpha=1.1, beta=0))
                else:
                    break

            # for t2w
            while True:
                if len(t2w) < input_depth and t2w:
                    t2w.append(cv2.convertScaleAbs(random.choice(t2w), alpha=1.1, beta=0))
                else:
                    break
            
    
            if 'FLAIR' in input_modality:
                return np.expand_dims(np.array(flair, dtype="object").T, axis=3), self.label.iloc[index,]
            
            elif 'T1w' in input_modality:
                return np.expand_dims(np.array(t1w, dtype="object").T, axis=3), self.label.iloc[index,]
    
            elif 'T1wCE' in input_modality:
                return np.expand_dims(np.array(t1wce, dtype="object").T, axis=3), self.label.iloc[index,]
            
            elif 'T2w' in input_modality:
                return np.expand_dims(np.array(t2w, dtype="object").T, axis=3), self.label.iloc[index,]
            else:
                return np.array((flair, t1w, t1wce, t2w), dtype="object").T, self.label.iloc[index,]
            
   
    
        elif modeling_in == '2D':
            # (None, h, w, channel == depth)
            return np.array(channel).T, self.label.iloc[index,]
        
    # function to read dicom file 
    def read_dicom_xray(self, path):
        data = pydicom.read_file(path).pixel_array
        if data.mean() == 0:
            return data 
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        return data