################################################################################
#
#   Repurposed from code by Christopher Angelini
#
#   Porpoise: Traverses data library and loads image path information into a vector
#
################################################################################
import numpy as np
import glob
import os
import cv2
from PIL import Image
import csv
import sys
import random
import math
import pickle
# Dataset information

# Numpy random seed for constraining random variables
np.random.seed(42)
random.seed(42)

# count number of folders in path
def get_patient_dirs(path):
    patient_dirs = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path, dI))]
    return patient_dirs

def generate_slice_list(path, patient, num_slices):
    not_enough_slice_flag = 0
    #patient_dir = path + '/' + patient + '/approx_five_mm'
    
    patient_dir = path + '/' + patient + '/rgb_channels'
    slice_dirs = [f for f in os.listdir(patient_dir) if os.path.isfile(os.path.join(patient_dir, f))]

    # perform zero padding if there are between 30 and 38 images in a scan
    if len(slice_dirs) >= 19 and len(slice_dirs) < 38:
        im = cv2.imread(patient_dir + '/' + slice_dirs[len(slice_dirs) - 1])
        h, w, c = im.shape

        last_slice = slice_dirs[len(slice_dirs) - 1]
        zero_pads = num_slices - len(slice_dirs)
        for i in range(len(slice_dirs), num_slices):
            zero_image = np.zeros((h, w))
            zero_image = zero_image.astype(np.uint8)
            im = Image.fromarray(zero_image)
            save_zero_dir = patient_dir + '/' + 'CT00000' + str(i) + '.jpg'
            im.save(save_zero_dir)
        print('Done Saving for Patient ' + patient)
        slice_dirs = [f for f in os.listdir(patient_dir) if os.path.isfile(os.path.join(patient_dir, f))]

    final_slice_dirs = []
    if len(slice_dirs) < num_slices:
        # print('Not enough slices')
        not_enough_slice_flag = 1

    elif len(slice_dirs) >= num_slices:
        not_enough_slice_flag = 0

        for i in range(0, num_slices):
            final_slice_dirs.append(patient_dir + '/' + slice_dirs[i])

    final_slice_dirs = np.array(final_slice_dirs)
    return final_slice_dirs, not_enough_slice_flag


def generate_labels(csv_dir, patient):
    # input number you want to search
    patient_string = patient

    # read csv, and split on "," the line
    csv_file = csv.reader(open(csv_dir, "rt"), delimiter=",")

    # loop through csv list
    row_idx = 0;
    for row in csv_file:
        row_idx + 1
        # if current rows 2nd value is equal to input, print that row
        if patient_string == row[0]:
            labels = row
            labels.pop(0)
            labels.pop(0)
            labels.pop(0)
            for i in range(0, len(labels)):
                labels[i] = int(labels[i])
    return np.array(labels)


# Class to be instantiated in the main functions
class oneChannelData():
    # Initialization function, setup the fields
    def __init__(self, slices):
        self.slices = slices

    # Creating the data given a data_dir, model_list, and movement_list
    def create_data(self, data_dir, label_dir):
        # Create an empty matrix of 1xtimestep for the full list of images
        full_image_train_list = np.empty((0, self.slices))
        full_image_test_list = np.empty((0, self.slices))
        # Create an empty matrix of 1x3 for the labels
        full_label_train_list = np.empty((0, 14))
        full_label_test_list = np.empty((0, 14))
        # Get lsit of patient directories
        patient_dirs = get_patient_dirs(data_dir)

        num_not_train = math.floor(len(patient_dirs) * 0.1)
        num_test = math.floor(num_not_train / 2)

        valid_patients = random.sample(patient_dirs, num_not_train)
        test_patients = random.sample(patient_dirs, num_test)

        with open('test_patients', 'wb') as fp:
            pickle.dump(test_patients, fp)

        # For the subject and a model list generate a matrix of images and a matrix of labels
        for i in range(0, len(patient_dirs)):
            if patient_dirs[i] not in test_patients and patient_dirs[i] in valid_patients:
                test_flag = 1
            else:
                test_flag = 0

            image_files, trash_flag = generate_slice_list(data_dir, patient_dirs[i], self.slices)
            if trash_flag != 1:
                image_labels = generate_labels(label_dir, patient_dirs[i])

                # Verticlely? Verticly? stack the the image files on the full image file list
                if test_flag == 1:
                    full_image_test_list = np.vstack((full_image_test_list, image_files))
                    full_label_test_list = np.vstack((full_label_test_list, image_labels))
                else:
                    full_image_train_list = np.vstack((full_image_train_list, image_files))
                    full_label_train_list = np.vstack((full_label_train_list, image_labels))
        print(full_label_test_list.shape)
        print(full_label_train_list.shape)

        return full_image_train_list, full_label_train_list, full_image_test_list, full_label_test_list
