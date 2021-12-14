import pickle
from typing import Tuple
import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn as sk
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier

def _concat_data(data_folder="../course_dataset")->pd.DataFrame:
    """Concat the 6 datasets in one dataframe
    Some preprocessing tasks are also done

    Args:
        data_folder (str, optional): folder where we can find each gesture folder. Defaults to "./course_dataset".

    Returns:
        pd.DataFrame: whole dataset
    """
    data_folders = glob(os.path.join(data_folder,'ASL_*'))
    all = pd.concat([pd.read_csv(os.path.join(folder,'annotations.csv'),index_col='ID') for folder in data_folders],ignore_index=True)
    all = all.drop(all[all.joint == 'hand_position'].index)
    return all


def _get_Xs_ys(ds:pd.DataFrame)->Tuple:
    """Split the dataset in training, validation and test
    X are list of numpy array of shape 20*2
    y are list of labels A,B,C,R,L,U

    Args:
        ds (pd.DataFrame): the whole dataset

    Returns:
        Tuple: tuple of 6 lists : X_train,X_val,X_test,y_train,y_val,y_test
    """
    X_train,X_val,X_test = [],[],[]
    y_train,y_val,y_test = [],[],[]

    for label in ds.gesture.unique():
        df = ds.loc[ds.gesture == label]
        video_idx = df.video_idx.unique()
        train,test = train_test_split(video_idx,test_size=.2)
        train,val = train_test_split(train,test_size=.2)
        for idx in video_idx:
            frames = df.loc[df.video_idx == idx]
            coordinates = frames[["x","y"]].to_numpy().reshape(len(frames.frame.unique()),40).tolist()
            if idx in train:
                X_train+= coordinates
                y_train+= [label]*len(coordinates)
            elif idx in val:
                X_val+= coordinates
                y_val+= [label]*len(coordinates)

            else:
                assert idx in test
                X_test+= coordinates
                y_test+= [label]*len(coordinates)

    X_train,X_val,X_test = np.array(X_train),np.array(X_val),np.array(X_test)
    y_train,y_val,y_test = np.array(y_train),np.array(y_val),np.array(y_test)

    return X_train,X_val,X_test,y_train,y_val,y_test


def _get_label_encoder(y:np.array)->sk.preprocessing.OneHotEncoder:
    """Returns a scikit-learn encoder object.
    This object is used to one hot encode our labels

    Args:
        y (np.array): list of labels

    Returns:
        sk.preprocessing.OneHotEncoder: encoder
    """
    encoder = sk.preprocessing.OneHotEncoder(dtype=np.float32)
    encoder.fit(y.reshape((-1, 1)))
    return encoder

def _encode_labels(y:np.array,encoder:sk.preprocessing.OneHotEncoder)->np.array:
    """Encodes a list of labels

    Args:
        y (np.array): list of categorical lables
        encoder (sk.preprocessing.OneHotEncoder): encoder object

    Returns:
        np.array: list of one hot encoded vectors
    """
    encoded = encoder.transform(y.reshape((-1, 1))).toarray()
    return encoded


def _build_model()->tf.keras.models.Sequential:
    """Build our hand gesture classification model

    Returns:
        tf.keras.models.Sequential: keras model
    """
    # build the model
    model = tf.keras.models.Sequential(
        layers = [
            tf.keras.layers.Input(shape=(40,)),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50,activation='relu'),
            tf.keras.layers.Dense(50,activation='relu'),
            tf.keras.layers.Dense(50,activation='relu'),
            tf.keras.layers.Dense(50,activation='relu'),
            tf.keras.layers.Dense(6,activation='softmax'),
        ]
    )

    # Set up gradient descent / create training pipeline.
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=1e-4),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()]
                )

    return model


def _train_model(model:tf.keras.models.Sequential,X_train:np.array,y_train:np.array,X_val:np.array,y_val:np.array):
    """Train our model
    Returns history of training

    Args:
        model (tf.keras.models.Sequential): compiled model
        X_train (np.array):
        y_train (np.array):
        X_val (np.array):
        y_val (np.array):
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=5,
        verbose=1,
        mode="auto",
    )
    # Perform gradient descent / train.
    history = model.fit(x=X_train,
            y=y_train,
            validation_data=(X_val,y_val),
            batch_size=50,
            epochs=250,
            verbose=0,
            callbacks=[early_stop]
            )
    return history

def _get_KNN_clf(X_train,y_train,**kwargs):
    clf = KNeighborsClassifier(**kwargs)
    clf.fit(X_train,y_train)
    return clf


class GestureClassifier:
    """"
    """
    
    def __init__(self,model_path:str,encoder_path:str) -> None:
        
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise(e)
        with open(encoder_path,'rb') as f:
            self.encoder = pickle.load(f)
    
    def predict(self,coordinates:np.array)->str:
        oneHot = self.model.predict(coordinates)
        pred = self.encoder.inverse_transform(oneHot)[0,0]
        return pred

    
