# Process DICOMs from Qure.AI to see if they are useful
# As a result of this code, we will have a list of DICOM files that once loaded will have pixel_array (DICOM data) available
# For some files I cannot get pixel_array, these subjects will be excluded
# Input - Path to DICOM folder that has all QUORE.AI data
# Output - a list of paths to usable patients' DICOMS

import numpy as np
import pydicom
import os
from glob import glob
import scipy.ndimage
import scipy.misc
import imageio


# Function that will load DICOM files
# Input - path to the folder that has all DICOM files
# output - dicom DS
def load_scan(path):
    # print(path)
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    # print(len(slices))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


# A function to build paths for all patients.
# Each patient has DICOM files in folders with different names.
# I have tried to build special cases
# Input - root path for QURE.AI folder
# Output - a list of individual paths for each patient

def build_path(dicom_path):
    # Get names of folders at this path

    folder_names = [name for name in os.listdir(dicom_path) if os.path.isdir(os.path.join(dicom_path, name))]
    thestudy = r'''Unknown Study'''
    case1 = r'''CT PLAIN THIN'''
    case2 = r'''CT Thin Plain'''
    case3 = r'''CT 5mm'''
    case4 = r'''CT PRE CONTRAST 5MM STD'''
    case5 = r'''CT Plain 3mm'''
    case6 = r'''CT 2.55mm'''
    case7 = r'''CT Plain'''
    case8 = r'''CT PRE CONTRAST THIN'''
    case9 = r'''CT 5mm Plain'''
    case10 = r'''CT 55mm Plain'''
    case11 = r'''CT 0.625mm'''
    nfolders = len(folder_names)
    path_to_dicom = []

    for x in range(nfolders):
        # print(x)
        # print(folder_names[x])
        new_path1 = os.path.join(dicom_path, folder_names[x], thestudy, case1)
        new_path2 = os.path.join(dicom_path, folder_names[x], thestudy, case2)
        new_path3 = os.path.join(dicom_path, folder_names[x], thestudy, case3)
        new_path4 = os.path.join(dicom_path, folder_names[x], thestudy, case4)
        new_path5 = os.path.join(dicom_path, folder_names[x], thestudy, case5)
        new_path6 = os.path.join(dicom_path, folder_names[x], thestudy, case6)
        new_path7 = os.path.join(dicom_path, folder_names[x], thestudy, case7)
        new_path8 = os.path.join(dicom_path, folder_names[x], thestudy, case8)
        new_path9 = os.path.join(dicom_path, folder_names[x], thestudy, case9)
        new_path10 = os.path.join(dicom_path, folder_names[x], thestudy, case10)
        new_path11 = os.path.join(dicom_path, folder_names[x], thestudy, case11)

        # try one by one and save to the list
        if os.path.exists(new_path1):
            path_to_dicom.append(new_path1)
        elif os.path.exists(new_path2):
            path_to_dicom.append(new_path2)
        elif os.path.exists(new_path3):
            path_to_dicom.append(new_path3)
        elif os.path.exists(new_path4):
            path_to_dicom.append(new_path4)
        elif os.path.exists(new_path5):
            path_to_dicom.append(new_path5)
        elif os.path.exists(new_path6):
            path_to_dicom.append(new_path6)
        elif os.path.exists(new_path7):
            path_to_dicom.append(new_path7)
        elif os.path.exists(new_path8):
            path_to_dicom.append(new_path8)
        elif os.path.exists(new_path9):
            path_to_dicom.append(new_path9)
        elif os.path.exists(new_path10):
            path_to_dicom.append(new_path10)
        elif os.path.exists(new_path11):
            path_to_dicom.append(new_path11)
        else:
            print('Patient Not Used')
            print(os.path.join(dicom_path, folder_names[x], thestudy))
            print(os.listdir(os.path.join(dicom_path, folder_names[x], thestudy)))

    return path_to_dicom


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

if __name__=='__main__':

    dicom_path = "F:/Research2/raw_data/QureAI_data"

    paths = build_path(dicom_path)
    n_patients = len(paths)
    print(n_patients)

    # Check patients for which we were unable to load pixel_data
    err_ct = 0
    path_sel_patinets = []
    for x in range(n_patients):
        print('*********************************')
        print(x)
        print(paths[x])
        patient = load_scan(paths[x])
        try:
            imgs = get_pixels_hu(patient)
            path_sel_patinets.append(paths[x])
            del imgs
        except:
            print('***********ERROR**********************')
            print(x)
            print(paths[x])

            err_ct = err_ct + 1
        del patient[:]

    print(err_ct)

    print(len(path_sel_patinets))

    # Save list
    import pickle

    with open('outfile', 'wb') as fp:
        pickle.dump(path_sel_patinets, fp)

    # Load list
    #with open('outfile', 'rb') as fp:
    #    itemlist = pickle.load(fp)