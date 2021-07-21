import random
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

class MultiDigitDataLoader(tf.keras.utils.Sequence):
    def __init__(self, 
                 df_base_path, 
                 image_base_path, 
                 batch_size,
                 bucket_size, 
                 shuffle=True):
        '''
        df_path: path for dataframe which has file_name and labels
        image_base_path: folder where images are
        batch_size: batch_size while training
        img_height: height of image
        img_width: width of image
        num_time_steps = number of input time steps for lstm layer
        '''
        self.batch_size = int(batch_size)
        self.image_base_path = image_base_path
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        
        self.df_1 = self.get_df(df_base_path, 'labels_1.csv')
        self.df_2 = self.get_df(df_base_path, 'labels_2.csv')
        self.df_3 = self.get_df(df_base_path, 'labels_3.csv')
        self.df_4 = self.get_df(df_base_path, 'labels_4.csv')
        self.df_5 = self.get_df(df_base_path, 'labels_5.csv')
        self.df_6 = self.get_df(df_base_path, 'labels_6.csv')
        self.df_7 = self.get_df(df_base_path, 'labels_7.csv')
        self.df_8 = self.get_df(df_base_path, 'labels_8.csv')

        self.df_list = [self.df_1, self.df_2, self.df_3, self.df_4, self.df_5, self.df_6, self.df_7, self.df_8]
        self.bucket_weights = {}
        self.total_samples = len(self.df_1) + len(self.df_2) + len(self.df_3) + len(self.df_4) + \
                             len(self.df_5) + len(self.df_6) + len(self.df_7) + len(self.df_8)
        for i in range(0, self.bucket_size):
            self.bucket_weights[i] = len(self.df_list[i]) / self.total_samples
        
        
        ## shuffle the df
        # if self.shuffle:
        #     self.shuffle_df()
    
    def create_y_true(self, label):
        num_list = [int(char) for char in label]
        return num_list
    
    def get_df(self, base_path, file_name):
        df = pd.read_csv(base_path + file_name, header = None, dtype={0: str, 1: str})
        df.columns = ["file_name", "labels"]
        df['y_true'] = df['labels'].apply(self.create_y_true)
        return df
    
    def shuffle_df(self):
        self.df_1 = self.df_1.sample(frac=1).reset_index(drop=True)
        self.df_2 = self.df_2.sample(frac=1).reset_index(drop=True)
        self.df_3 = self.df_3.sample(frac=1).reset_index(drop=True)
        self.df_4 = self.df_4.sample(frac=1).reset_index(drop=True)
        self.df_5 = self.df_5.sample(frac=1).reset_index(drop=True)
        self.df_6 = self.df_6.sample(frac=1).reset_index(drop=True)
        self.df_7 = self.df_7.sample(frac=1).reset_index(drop=True)
        self.df_8 = self.df_8.sample(frac=1).reset_index(drop=True)

        self.df_list = [self.df_1, self.df_2, self.df_3, self.df_4, self.df_5, self.df_6, self.df_7, self.df_8]
    
    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_df()
            
    
    def __len__(self):
        return int(np.ceil(self.total_samples / float(self.batch_size)))
    
    def get_image(self, filename):
        img = cv2.imread(filename, 0)
        img = img.reshape((img.shape[0], img.shape[1], 1))
        return img
    
    def __getitem__(self, idx):
        random_bucket_idx = np.random.choice(self.bucket_size, 1, p=[value for value in self.bucket_weights.values()])
        random_bucket_idx = int(random_bucket_idx[0])

        df = self.df_list[random_bucket_idx]
        image_batch = []
        label_batch = []

        for i in range(self.batch_size):
            sample_idx = random.randint(0, len(df) - 1)
            image_batch.append(self.get_image(self.image_base_path + df.iloc[sample_idx]['file_name']))
            label_batch.append(df.iloc[sample_idx]['y_true'])
        
        image_batch = np.array(image_batch).astype('float32')
        label_batch = np.array(label_batch).astype(np.int32)

        image_batch = (image_batch - 127.5) / 127.5
        
        return image_batch, label_batch



class MultiDigitDataLoaderActualShape(tf.keras.utils.Sequence):
    def __init__(self, 
                 df_path, 
                 image_base_path, 
                 batch_size, 
                 num_time_steps,
                 transform,
                 max_digit_length=3, 
                 shuffle=True):
        '''
        df_path: path for dataframe which has file_name and labels
        image_base_path: folder where images are
        batch_size: batch_size while training
        img_height: height of image
        img_width: width of image
        num_time_steps = number of input time steps for lstm layer
        '''
        self.batch_size = int(batch_size)
        self.image_base_path = image_base_path
        self.shuffle = shuffle
        self.max_digit_length = max_digit_length
        self.num_time_steps = num_time_steps
        self.transform = transform
        
        self.df = pd.read_csv(df_path, header = None, dtype={0: str, 1: str})
        self.df.columns = ["file_name", "labels"]
        self.df['y_true'] = self.df['labels'].apply(self.create_y_true)
        self.df['label_length'] = self.df['labels'].apply(self.get_label_length)
        
        ## shuffle the df
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def create_y_true(self, label):
        y_true = np.ones([1,self.max_digit_length]) * (-1)
        num_list = [int(char) for char in label]
        y_true[0, 0:len(label)] = num_list
        return y_true
    
    def get_label_length(self, label):
        return len(label)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))
    
    def get_image(self, filename):
        img = cv2.imread(filename, 0)
        img = self.transform(image=img)["image"]
        img_height = img.shape[0]
        img_width = img.shape[1]
        img = img.reshape((img_height, img_width, 1))
        return img
    
    def __getitem__(self, idx):
        idx_start = idx * self.batch_size
        idx_end = min((idx + 1) * self.batch_size, len(self.df))
        
        y_train = np.concatenate(self.df['y_true'][idx_start:idx_end].values,axis=0)
        
        x_train = [self.get_image(self.image_base_path + self.df['file_name'][i]) for i in range(idx_start,idx_end)]
        x_train = np.array(x_train)
        x_train = np.transpose(x_train, axes=[0,2,1,3])
        
        input_length_arr = self.num_time_steps * np.ones(shape=(idx_end - idx_start, 1), dtype="int64")
        
        label_length_arr = self.df['label_length'][idx_start:idx_end].values.reshape(idx_end - idx_start,1)
        
        inputs = {'image': x_train,
                  'label': y_train,
                  'input_length': input_length_arr,
                  'label_length': label_length_arr,
                  }
        
        return inputs