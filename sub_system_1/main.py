import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from preprocess import preprocess, build_model

#Variables
letters=['A','B','C','L','R','U']
hand_landmarks = ['root','thumb_1','thumb_2','thumb_3',
                  'index_1','index_2','index_3','index_4',
                 'middle_1','middle_2','middle_3','middle_4',
                 'ring_1','ring_2','ring_3','ring_4',
                 'pinky_1','pinky_2','pinky_3','pinky_4']

outliers = [9,40] #Videos 9 and 40 are not 480x640
width, height, shape_factor = 480, 640, 4

input_shape = (int(width/shape_factor),int(height/shape_factor),3)
output_shape = len(hand_landmarks)*2 #2 coordinates per landmarks

if __name__ == '__main__':
  #If you don't have data preprocess
  if not os.path.isfile('DATA_XY/X.npy'):

    X, Y = preprocess(letters=letters, 
                      hand_landmarks=hand_landmarks, 
                      outliers=outliers, 
                      width=width, height=height, shape_factor=shape_factor)

    os.makedirs('DATA_XY',exist_ok=True)
    np.save('DATA_XY/X.npy',X)
    np.save('DATA_XY/Y.npy',Y)
    
  #If you already have Data preprocess
  else:
    X = np.load('DATA_XY/X.npy')
    Y = np.load('DATA_XY/Y.npy')

  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
  X_train,X_val,y_train,y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

  BATCH_SIZE = 32
  EPOCHS = 500
  PATIENCE = 100
  train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
  test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

  train_dataset = train_dataset.batch(BATCH_SIZE)
  val_dataset = val_dataset.batch(BATCH_SIZE)
  test_dataset = test_dataset.batch(BATCH_SIZE)


  model = build_model(input_shape,output_shape)

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss=keras.losses.mean_squared_error)#keras.losses.huber)

  early_stop = tf.keras.callbacks.EarlyStopping(
      monitor="val_loss",
      min_delta=1e-2,
      patience=PATIENCE,
      verbose=1,
      mode="auto",
  )
  model.fit(train_dataset, validation_data=val_dataset,verbose=1,epochs=EPOCHS,callbacks=[early_stop])


  model.save("models/Trained_model")