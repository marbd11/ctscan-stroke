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
from sklearn.model_selection import KFold

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
num_gpus = 8

seed = 42
np.random.seed(42)
tf.set_random_seed(3901)


# Inputs class
class Args:
    def __init__(self):
        self.data_dir = '/data/delroc72/QureAI_jpegs'
        self.label_dir = '/data/delroc72/reads_groundTruths.csv'

        now = strftime("%m%d%y_%H%M%S", gmtime())
        self.outputPath = './VGG16LstmModels/' + now + '/'

        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)

        self.epochs = 50
        self.batch_size = 8
        self.num_slices = 35
        self.num_classes = 14
        self.channels = 3
        self.width = 224
        self.height = 224


if __name__ == '__main__':
    args = Args()

    ct_scan_gen = oneChannelData(args.num_slices)

    #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    kfold = KFold(10, True, seed)

    it = 0
    model_images, model_labels = ct_scan_gen.create_data_kfold(args.data_dir, args.label_dir)

    for train, test in kfold.split(model_images):

        it = it +1

        base_model = tf.keras.applications.VGG16(weights='imagenet',
                                                 input_shape=(args.width, args.height, args.channels),
                                                 include_top=False)

        cnnOut = base_model.layers[18].output

        cnnModel = tf.keras.Model(base_model.input, cnnOut)

        for layer in cnnModel.layers:
            layer.trainable = True

        input = layers.Input(batch_shape=(args.batch_size, args.num_slices, args.width, args.height, args.channels))
        tdOut = td(cnnModel)(input)
        flOut = td(layers.Flatten())(tdOut)
        lstmOut = layers.LSTM(50, activation='tanh')(flOut)
        preds = layers.Dense(args.num_classes, activation='sigmoid')(lstmOut)

        model = tf.keras.models.Model(inputs=input, outputs=preds)

        # model = load_model('./resnetModels/042619_055142/wholeModel.h5')
        params = {'batch_ersize': args.batch_size,
                  'time_steps': args.num_slices,
                  'input_shape': (args.width, args.height, args.channels),
                  'num_classes': args.num_classes,
                  'shuffle': True}

        training_generator = CTSequencer(model_images[train], model_labels[train], **params)
        testing_generator = CTSequencer(model_images[train], model_labels[train], **params)

        opt = tf.keras.optimizers.Adam(lr=0.0001)

        model = multi_gpu_model(model, gpus=num_gpus)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        csv_logger = CSVLogger(args.outputPath + 'train_'+str(it)+'.log')

        model.fit_generator(generator=training_generator,
                            steps_per_epoch=len(training_generator),
                            epochs=args.epochs, shuffle=True, workers=8, use_multiprocessing=True,
                            validation_data=testing_generator, validation_steps=len(testing_generator),
                            callbacks=[csv_logger])

        model.save(args.outputPath + 'wholeModel_'+str(it)+'.h5')
        model.save_weights(args.outputPath + 'weightsOnlyModel_'+str(it)+'.h5')