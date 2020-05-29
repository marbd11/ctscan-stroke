################################################################################
#
#   Author: Christopher Angelini
#
#   Porpoise: Dynamically concatinates and loads images into tensorflow for training
#
################################################################################
import tensorflow as tf
import cv2
import numpy as np

# Sequencer based on the tensorflow sequencer for abnormal dataset folders
class CTSequencer(tf.keras.utils.Sequence):
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

        # For one time step
        if self.time_steps == 1:
            # Create empty arrays used a ndarray for some reason with the shape
            X = np.ndarray((self.batch_size, *self.input_shape), dtype=float)
            y = np.empty((self.batch_size, 1), dtype=float)

            # For the batch indexes that were found previously
            for i, idx in enumerate(indexes):
                video = 0
                # Loop for video and change the background of the image at each new video
                while index > self.video_start[video]:
                    video += 1
                bgimg = self.background[video]

                # Load image
                frame = cv2.imread(self.X_data[idx][0], 1)
                # Add background into the image
                comp_image = compose_image(frame, bgimg)
                # Resize the image
                res_img = np.asarray(cv2.resize(comp_image, (self.input_shape[0], self.input_shape[1])))
                # Normalize the image
                normed_img = (res_img - res_img.mean()) / res_img.std()
                # Enter the data into X
                X[i, :] = normed_img
                y[i,:] = self.y_data[idx]

            return X, tf.keras.utils.to_categorical(y, self.num_classes)
        else:
            # For all timesteps greater than 1
            # Create empty arrays used a ndarray for some reason with the shape
            X = np.ndarray((self.batch_size, self.time_steps, *self.input_shape), dtype=float)
            y = np.empty((self.batch_size, self.num_classes), dtype=float)

            # Loop for finding where the video frames switch
            for i, index in enumerate(indexes):
                # For time frames
                for time_idx in range(self.time_steps):
                    # Load image
                    frame = cv2.imread(self.X_data[index][time_idx], 1)
                    # Resize image
                    res_img = np.asarray(cv2.resize(frame, (self.input_shape[0], self.input_shape[1])))
                    # Normalize image
                    #normed_img = (res_img - res_img.mean()) / res_img.std()
                    # Add image to the appropriate time step
                    X[i, time_idx, :] = res_img
                y[i] = self.y_data[index]
            #return X, tf.keras.utils.to_categorical(y, self.num_classes)
            return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
