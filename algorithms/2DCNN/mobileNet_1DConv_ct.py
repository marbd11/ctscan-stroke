################################################################################
#
#   Author: Marissa DelRocini
#
#   Purpose: Main file for the QureAI dataset VGG16 architecture
#
################################################################################
import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import os
import tensorflow.keras.layers as layers
from tensorflow import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import DepthwiseConv2D
from keras.layers import SeparableConv2D
from keras.layers import GlobalAveragePooling2D
import cv2
from keras.layers import Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TimeDistributed as td
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.core.protobuf import rewriter_config_pb2
from keras.optimizers import adam
from data_generator import dataGenerator
from dataset import oneChannelData
from keras.utils import multi_gpu_model
from sklearn.model_selection import KFold

# import the necessary packages
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from tensorflow.keras import backend as K

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] =  "0,1,2"
num_gpus=3

seed = 42
np.random.seed(42)
tf.set_random_seed(3901)

# Inputs class
class Args:
    def __init__(self):
        self.data_dir  = '/data/delroc72/QureAI_jpegs'
        self.label_dir = '/data/delroc72/reads_groundTruths.csv'

        now = strftime("%m%d%y_%H%M%S", gmtime())
        self.outputPath = './mobileNet_outputs/' + now + '/'

        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)

        self.epochs = 100
        self.batch_size = 32
        self.num_slices = 38
        self.num_classes = 14
        self.channels = 1
        self.width = 224
        self.height = 224

if __name__=='__main__':
    args = Args()

    #keras.backend.set_image_data_format('channels_first')

    ct_scan_gen = oneChannelData(args.num_slices)

    #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    kfold = KFold(10, True, seed)

    it = 0
    model_images, model_labels = ct_scan_gen.create_data_kfold(args.data_dir, args.label_dir)

    for train, test in kfold.split(model_images):

        print("TRAIN:", train, "TEST:", test)

        it = it+1

        model = Sequential()
        # Conv Block
        model.add(Conv2D(32, (3, 3), padding="same",input_shape=( args.width, args.height, args.num_slices)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=2))

        # Depthwise Conv Block 1
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 2
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(2,2)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 3
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 4
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(2,2)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 5
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 6
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(2,2)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 7
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 8
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 9
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 10
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 11
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 12
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(2,2)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(1024, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        # Depthwise Conv Block 13
        model.add(DepthwiseConv2D((3,3), padding="same", depth_multiplier=1, strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.add(Conv2D(1024, (3, 3), padding="same", strides=(1,1)))
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))

        model.add(GlobalAveragePooling2D())
        model.add(Reshape((128, -1)))
        model.add(Conv1D(64, 3, padding='same', activation='relu', strides=1))
        model.add(Flatten())
        model.add(Reshape((1,1,1024)))
        model.add(Dropout(0.5))
        model.add(Conv2D(args.num_classes, (1,1), padding="same"))
        model.add(Reshape((args.num_classes,)))
        model.add(Activation("sigmoid"))

        model.summary()

        #model = load_model('./resnetModels/042619_055142/wholeModel.h5')
        params = {'batch_size': args.batch_size,
                  'time_steps': args.num_slices,
                  'input_shape': (args.width,args.height,args.channels),
                  'num_classes': args.num_classes,
                  'shuffle': True}

        training_generator = dataGenerator(model_images[train], model_labels[train], **params)
        testing_generator = dataGenerator(model_images[test], model_labels[test], **params)

        opt = adam(lr=0.0001)

        model = multi_gpu_model(model, gpus=num_gpus)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        csv_logger = CSVLogger(args.outputPath+'train'+str(it)+'.log')

        model.fit_generator(generator=training_generator,
                            steps_per_epoch=len(training_generator),
                            epochs=args.epochs, shuffle=True, workers=2,  use_multiprocessing=False,
                            validation_data=testing_generator, validation_steps=len(testing_generator),
                            callbacks=[csv_logger])

        model.save(args.outputPath + 'wholeModel_'+str(it)+'.h5')
        model.save_weights(args.outputPath + 'weightsOnlyModel'+str(it)+'.h5')
