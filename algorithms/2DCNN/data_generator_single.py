################################################################################
#
#   Author: Marissa DelRocini
#
#   Porpoise: Dynamically concatinates and loads images into tensorflow for training
#
################################################################################
import tensorflow as tf
import cv2
import numpy as np
import keras
from PIL import Image
import matplotlib
import scipy.misc

# Sequencer based on the tensorflow sequencer for abnormal dataset folders
class dataGenerator(keras.utils.Sequence):
    # Initialization function
    def __init__(self, X_data, y_data, batch_size, time_steps, input_shape, num_classes, shuffle=True):
        self.X_data = X_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.shuffle = shuffle
        # Array of indexes with shuffle
        if self.shuffle:
            self.indexes = np.arange(len(self.X_data))
            np.random.shuffle(self.indexes)
        else:
            self.indexes = np.arange(len(self.X_data))
        self.on_epoch_end()

    # Get length of batch training
    def __len__(self):
        return int(np.floor(len(self.X_data) / self.batch_size))

    # Returning a batch item
    def __getitem__(self, index):
        # Get range of indexes based on the training index and the batch size
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Create empty arrays used a ndarray for some reason with the shape
        #X = np.ndarray((self.batch_size, self.input_shape[0], self.input_shape[1]), self.time_steps, dtype=float)
        X = np.ndarray((self.batch_size, self.input_shape[0], self.input_shape[1], self.time_steps), dtype=np.uint8)
        y = np.empty((self.batch_size, self.num_classes), dtype=float)

        for i, index in enumerate(indexes):
            # For time frames
            for time_idx in range(self.time_steps):
                # Load image
                scan = cv2.imread(self.X_data[index][time_idx], 1)
                # Resize image
                res_img = np.asarray(cv2.resize(scan, (self.input_shape[0], self.input_shape[1])))

                # Add image to the appropriate time step
                X[i, :, :, time_idx] = res_img

                #im = Image.fromarray(X[i, :, :, time_idx])
                #print(im.size)
                #im.save("test_image3.png")

            y[i] = self.y_data[index]

        #print(X.shape)
        #print(y.shape)
        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)