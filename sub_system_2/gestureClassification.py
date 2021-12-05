
from typing import Tuple
import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn as sk
import tensorflow as tf
import plotly.express as px
#'width': 640, 'height': 480

def _concat_data(data_folder="course_dataset")->pd.DataFrame:
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
    X = []
    y = []
    frames = ds.groupby(by=['gesture','video_idx','frame']).groups
    for (gesture,_,_),idx in frames.items():
        y.append(gesture.split('_')[-1])
        X.append(ds.loc[idx][["x","y"]].to_numpy().tolist())

    X,y = np.array(X), np.array(y)

    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=.2)
    X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=.2)

    X_train,y_train = shuffle(X_train,y_train)

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
            tf.keras.layers.Input(shape=(20,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(20,activation='relu'),
            tf.keras.layers.Dense(20,activation='relu'),
            tf.keras.layers.Dense(20,activation='relu'),
            tf.keras.layers.Dense(6,activation='softmax'),
        ]
    )

    # Set up gradient descent / create training pipeline.
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()]
                )

    return model


def _train_model(model:tf.keras.models.Sequential,X_train:np.array,y_train:np.array,X_val:np.array,y_val:np.array,x_test:np.array,y_test:np.array):
    """Train our model
    Returns history of training

    Args:
        model (tf.keras.models.Sequential): compiled model
        X_train (np.array):
        y_train (np.array):
        X_val (np.array):
        y_val (np.array):
        x_test (np.array):
        y_test (np.array):
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
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

