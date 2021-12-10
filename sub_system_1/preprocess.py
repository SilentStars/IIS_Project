import pandas as pd
import os
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

def preprocess(letters : list, hand_landmarks : list, outliers : list, width : int, height : int, shape_factor : int) -> tuple([list,list]):
  X = [] #List of frames by order of videos and frames
  Y = [] #List of coordinates by order of videos and frames
  for letter in letters: #We browse each letter folder

    #Read dataset and withdraw useless rows 'hand_position'
    dataset = pd.read_csv(f'../course_dataset/ASL_letter_{letter}/annotations.csv')
    dataset = dataset.drop(dataset[dataset['joint'] == 'hand_position'].index,axis=0)

    nb_videos = len(os.listdir(f'../course_dataset/ASL_letter_{letter}/videos'))
    for video_idx in range(nb_videos): #We browse each letter video
        print(f"Letter {letter} : video_n°{video_idx}")

        if video_idx in outliers: #Videos of  different shape
          pass
          
        else:
          import copy
          #New list for each video to add to X/Y
          x_video = []
          y_video = []
        
          cap = cv2.VideoCapture(f'../course_dataset/ASL_letter_{letter}/videos/video_{video_idx}.mp4')
          video = dataset[dataset['video_idx']==video_idx]

          #Apply the reduction to coordinates
          video.loc[:,('x')] = video.loc[:,('x')].div(shape_factor)       
          video.loc[:,('y')] = video.loc[:,('y')].div(shape_factor)

          nb_frame = video['frame'].max()
          
          for frame in range(nb_frame+1):
              landmarks = []
              for joint in hand_landmarks:
                  
                  x = float(video['x'][(video['joint']==joint)&(video['frame']==frame)])
                  y = float(video['y'][(video['joint']==joint)&(video['frame']==frame)])

                  landmarks.append(y)#OpenCV flip coordinates
                  landmarks.append(x)
              y_video.append(np.array(landmarks).reshape(2*len(hand_landmarks),1))
          
          #Filling X list
          while cap.isOpened():
                  
                  ret,frame = cap.read()
                  
                  if type(frame) is type(None):
                      break

                  #Image configuration
                  image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)             
                  
                  #Resizing
                  dim = (int(image.shape[1]/shape_factor),int(image.shape[0]/shape_factor))
                  image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

                  x_video.append(image)


          #There is difference between the number of frame read by CV2 and the number in CSV_annotations.
          #We take the lowest one to avoid issues.
          MIN = min(len(x_video),len(y_video))
          
          #Avoid the case when X/Y = []
          if video_idx == 0 and letters.index(letter)== 0:
              X = np.array(x_video[:MIN])
              
              Y = np.array(y_video[:MIN])
          else:
            X = np.concatenate((np.array(X), np.array(x_video[:MIN])),axis=0)

            Y = np.concatenate((np.array(Y), np.array(y_video[:MIN])),axis=0)
  return X, Y  

def build_model(input_shape : tuple([int,int,int]),output_shape : int) -> keras.models : 
  #First Block
  inputs = tf.keras.Input(shape=(input_shape), name="Input_Image") 
  x = tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',name='conv_32_1')(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf. keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='valid')(x)

  #Second Block
  x = tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',name='conv_32_2')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf. keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='valid')(x)

  #Third block
  x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',name='conv_64_1')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf. keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='valid')(x)

  #Fourth block
  x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',name='conv_64_2')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x =tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',name='conv_64_3')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf. keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='valid')(x)

  #Fifth Block
  x = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',name='conv_128_1',kernel_regularizer =tf.keras.regularizers.l2(l=0.01))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf. keras.layers.MaxPool2D(pool_size=(2, 2),strides=(1, 1),padding='valid')(x)

  #Output Block
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(units=1024
                            ,activation='relu',use_bias=True,kernel_regularizer =tf.keras.regularizers.l2(l=0.01))(x)
  x = tf.keras.layers.BatchNormalization()(x)

  outputs = tf.keras.layers.Dense(units=output_shape,activation='relu',use_bias=True)(x)
  model = tf.keras.Model(inputs=inputs,outputs=outputs)

  return model