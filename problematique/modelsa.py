import time
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import keras as K
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as ttsplit

import helpers.analysis as an
import helpers.classifiers as classifiers
from sklearn.preprocessing import OneHotEncoder
import glob
import os
from skimage import io as skiio
from functions import *
import tensorflow as tf
from keras.optimizers import Adam


class NN:
    def __init__(self):

        start=time.time()
        print('Starting NN')

        image_folder = r"." + os.sep + "baseDeDonneesImages"
        _path = glob.glob(image_folder + os.sep + r"*.jpg")
        labellist = ['coast', 'coast_sun', 'forest', 'forest_for', 'forest_nat', 'street', 'street_urb', 'street_gre']
        labellist = ['coast', 'forest', 'street']

        pathlist = []
        for label in labellist:
            pathlist = pathlist + glob.glob(image_folder + os.sep + label + r"*.jpg")

        target=np.zeros(len(pathlist))
        for i,path in enumerate(pathlist):
            for j,label in enumerate(labellist):
                if label in path:
                    target[i]=j
        #Remove to process all data
        target=target
        pathlist = pathlist

        funclist=[avgBlue,avgRed,corner,nbedges,entropy]
        # funclist=[avgBlue,avgRed]
        data=np.zeros((len(target),len(funclist)))

        for id,path in enumerate(pathlist):
            img = skiio.imread(path)
            for funcID,func in enumerate(funclist):
                data[id][funcID]=func(img)
        print(f'Fetched components : {time.time() - start} seconds')

        encoder = OneHotEncoder(sparse=False)
        target = encoder.fit_transform(target.reshape(-1, 1))
        target_decode = np.argmax(target, axis=-1)
        target_decode=target_decode.T
        training_data, validation_data, training_target, validation_target = ttsplit(data, target, test_size=0.10)

        # # Create neural network
        model = Sequential()
        model.add(Dense(units=5, activation='tanh',
                        input_shape=(data.shape[-1],)))
        model.add(Dense(units=4, activation='tanh'))
        model.add(Dense(units=3, activation='tanh'))
        model.add(Dense(units=target.shape[-1], activation='sigmoid'))
        print(model.summary())
        #
        # # Define training parameters
        # model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy',metrics=['accuracy'])
        #
        # # Perform training
        callback_list = [tf.keras.callbacks.ModelCheckpoint(
            filepath='iris.h5',
            verbose=1,
            save_best_only=True),

            tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=50,
            verbose=1,
            mode='auto',
            baseline=None,
            restore_best_weights=False)
            ]
        model.fit(training_data, training_target, batch_size=20, verbose=1,
                  epochs=500, shuffle=True, callbacks=callback_list[0],
                  validation_data=(validation_data, validation_target))
        print(f'Done training : {time.time() - start} seconds')
        #

        an.plot_metrics(model)
        history = model.history.history
        print('Model Best accuracy: ' + str(max(history['accuracy'])*100) + '%')
        # # Test model (loading from disk)
        model = load_model('iris.h5')
        targetPred = model.predict(data)


        # # Print the number of classification errors from the training data
        # error_indexes = classifiers.calc_erreur_classification(targetPred, target_decode)
        targetPred=np.argmax(targetPred, axis=-1)
        error_indexes = classifiers.calc_erreur_classification(targetPred, target_decode)
        #print(error_indexes)
        #
        plt.show()


class print_every_N_epochs(K.callbacks.Callback):
    """
    Helper callback pour remplacer l'affichage lors de l'entraînement
    """
    def __init__(self, N_epochs):
        self.epochs = N_epochs

    def on_epoch_end(self, epoch, logs=None):
        if True:
            print("Epoch: {:>3} | Loss: ".format(epoch) +
                  f"{logs['loss']:.4e}" + " | Valid loss: " + f"{logs['val_loss']:.4e}" +
                  (f" | Accuracy: {logs['accuracy']:.4e}" + " | Valid accuracy " + f"{logs['val_accuracy']:.4e}"
                   if 'accuracy' in logs else "") )