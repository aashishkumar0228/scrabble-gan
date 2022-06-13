import random
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

class IAMDbWordsDataLoader(tf.keras.utils.Sequence):
    def __init__(self, 
                 df_base_path, 
                 image_base_path, 
                 batch_size,
                 char_vector,
                 image_height=32,
                 bucket_size=10, 
                 shuffle=True):
        '''
        df_base_path: folder where the CSVs are present
        image_base_path: folder where images are
        batch_size: batch_size while training
        '''

        self.image_base_path = image_base_path
        self.batch_size = int(batch_size)
        self.char_vector = char_vector
        self.image_height = image_height
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
        self.df_9 = self.get_df(df_base_path, 'labels_9.csv')
        self.df_10 = self.get_df(df_base_path, 'labels_10.csv')

        self.df_list = [self.df_1, self.df_2, self.df_3, self.df_4, self.df_5, self.df_6, self.df_7, self.df_8, self.df_9, self.df_10]
        self.bucket_weights = {}
        self.total_samples = len(self.df_1) + len(self.df_2) + len(self.df_3) + len(self.df_4) + \
                             len(self.df_5) + len(self.df_6) + len(self.df_7) + len(self.df_8) + \
                             len(self.df_9) + len(self.df_10)
        for i in range(0, self.bucket_size):
            self.bucket_weights[i] = len(self.df_list[i]) / self.total_samples
        
        
        ## shuffle the df
        # if self.shuffle:
        #     self.shuffle_df()
    
    def create_y_true(self, label):
        char_list = [self.char_vector.index(char) for char in label]
        return char_list
    
    def get_df(self, base_path, file_name):
        df = pd.read_csv(base_path + file_name)
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
        self.df_9 = self.df_9.sample(frac=1).reset_index(drop=True)
        self.df_10 = self.df_10.sample(frac=1).reset_index(drop=True)

        self.df_list = [self.df_1, self.df_2, self.df_3, self.df_4, self.df_5, self.df_6, self.df_7, self.df_8, self.df_9, self.df_10]
    
    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_df()
            
    
    def __len__(self):
        return int(np.ceil(self.total_samples / float(self.batch_size)))
    
    def get_image(self, filename, len_label):
        img = cv2.imread(filename, 0)
        img = cv2.resize(img, (int(self.image_height / 2) * len_label, self.image_height))
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
            image_batch.append(self.get_image(self.image_base_path + df.iloc[sample_idx]['file_name'], len(df.iloc[sample_idx]['labels'])))
            label_batch.append(df.iloc[sample_idx]['y_true'])
        
        image_batch = np.array(image_batch).astype('float32')
        label_batch = np.array(label_batch).astype(np.int32)

        image_batch = (image_batch - 127.5) / 127.5
        
        return image_batch, label_batch

