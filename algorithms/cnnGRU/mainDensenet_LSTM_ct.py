################################################################################
#
#   Author: Christopher Angelini
#
#   Porpoise: Main file for the VGG16_LSTM architecture
#
################################################################################
import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import os
import tensorflow.keras.layers as layers
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TimeDistributed as td
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session
from tensorflow.keras.utils import multi_gpu_model
from sequencer import CTSequencer
from dataset import oneChannelData

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] =  "0,1,2,3,4"

np.random.seed(42)
tf.set_random_seed(3901)

# Inputs class
class Args:
    def __init__(self):
        self.data_dir  = '/data/delroc72/QureAI_jpegs'
        self.label_dir = '/data/delroc72/reads_groundTruths.csv'

        now = strftime("%m%d%y_%H%M%S", gmtime())
        self.outputPath = './DenseNetLSTMOutputs/' + now + '/'

        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)

        self.epochs = 100
        self.batch_size = 16
        self.max_angle = 90
        self.num_slices = 38
        self.num_classes = 14
        self.channels = 3
        self.width = 224
        self.height = 224

if __name__=='__main__':
    args = Args()

    ct_scan_gen = oneChannelData(args.num_slices)

    train_images, train_labels, test_images, test_labels = ct_scan_gen.create_data(args.data_dir, args.label_dir)

    base_model = tf.keras.applications.DenseNet121(include_top = False, weights = "imagenet", input_shape = (args.width, args.height, args.channels))

    cnnModel = tf.keras.Model(base_model.input, base_model.output)

    for layer in cnnModel.layers:
        layer.trainable = False

    input = layers.Input(batch_shape=(args.batch_size, args.num_slices, args.width,args.height,args.channels))
    tdOut = td(cnnModel)(input)
    flOut = td(layers.Flatten())(tdOut)
    lstmOut = layers.LSTM(50, activation='tanh')(flOut)
    preds = layers.Dense(args.num_classes, activation='sigmoid')(lstmOut)

    model = tf.keras.models.Model(inputs=input, outputs=preds)

    #model = load_model('./resnetModels/042619_055142/wholeModel.h5')
    params = {'batch_size': args.batch_size,
              'time_steps': args.num_slices,
              'input_shape': (args.width,args.height,args.channels),
              'num_classes': args.num_classes,
              'shuffle': True}

    training_generator = CTSequencer(train_images, train_labels, **params)

    testing_generator = CTSequencer(test_images, test_labels, **params)

    opt = tf.keras.optimizers.Adam(lr=0.0001)

    model = multi_gpu_model(model, gpus=5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    csv_logger = CSVLogger(args.outputPath+'train.log')

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=len(training_generator),
                        epochs=args.epochs, shuffle=True, workers=2,  use_multiprocessing=False,
                        validation_data=testing_generator, validation_steps=len(testing_generator),
                        callbacks=[csv_logger])

    model.save(args.outputPath + 'wholeModel.h5')
    model.save_weights(args.outputPath + 'weightsOnlyModel.h5')
    
