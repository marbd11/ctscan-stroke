import pydicom
import numpy as np
import os
from PIL import Image
import imageio
from glob import glob
import pickle
import cv2

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_image_window(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    wl = slices[0].WindowCenter
    ww = slices[0].WindowWidth

    lower_bound = wl - (ww/2)
    upper_bound = wl + (ww/2)

    return [lower_bound, upper_bound]

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

# Resampling algorithm
def resample_5mm(imges, width):
    if width <= 5:
        jump = np.around(5/width)
        jump = int(jump)
        newimges = imges[0::jump,:,:]
    else:
        newimges = imges
    return newimges

# Get Patient ID to save data from the Path
def get_patient_id(path):
    b = path.split('\\')
    e = b[1].split(' ')
    print(e)
    return e[0]

# Save JPGs
def save_images(image_window, output_path, pat_id, folder_name):
    # How many images to save
    n_images = image_window.shape[0]

    # Go over each window
    for x in range(n_images):
        # Specify path
        out_path = os.path.join(output_path, pat_id, folder_name)
        # Patient Folder
        pat_path = os.path.join(output_path, pat_id)

        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        file_name = out_path + 'CT00000{}.jpg'.format(x)
        img_data = image_window[x, :, :]

        img_data = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        imageio.imwrite(file_name, img_data)

if __name__=='__main__':

    # Constants
    output_path = "F:/Research2/processed_images_test"

    with open('outfile', 'rb') as fp:
        patient_list = pickle.load(fp)

    n_patients = len(patient_list)

    for x in range(n_patients):
        print('*********************************')
        print(patient_list[x])

        patient_id = get_patient_id(patient_list[x])

        # Load patinet
        patient = load_scan(patient_list[x])
        [image_lower, image_upper] = get_image_window(patient_list[x])

        width = patient[0].SliceThickness
        # Get pixel data
        imgs = get_pixels_hu(patient)

        print("Width: " + str(width))
        print("Before Resampling: " + str(imgs.shape))

        # Resample
        new_imges = resample_5mm(imgs, width)

        # Save Resampled
        print("After Resampling: " + str(new_imges.shape))

        # Get three windows and save JPGs (resampled)
        # Brain window
        brain_window = np.array(new_imges)
        brain_window[brain_window < 0] = 0
        brain_window[brain_window > 80] = 80

        # Bone Window
        bone_window = np.array(new_imges)
        bone_window[bone_window < -1000] = -1000
        bone_window[bone_window > 2000] = 2000

        # Subdural
        subdural_window = np.array(new_imges)
        subdural_window[subdural_window < 150] = 150
        subdural_window[subdural_window > 200] = 200

        # Radiologist Window
        radiologist_window = np.array(new_imges)
        radiologist_window[radiologist_window < image_lower] = image_lower
        radiologist_window[radiologist_window > image_upper] = image_upper

        # RGB Image
        rgb_window = np.zeros((new_imges.shape[0],512, 512, 3), 'uint8')
        rgb_window[..., 0] = brain_window
        rgb_window[..., 1] = bone_window
        rgb_window[..., 2] = subdural_window

        # Now save these all to a new folder
        save_images(subdural_window, output_path, patient_id, 'subdural/')
        save_images(bone_window, output_path, patient_id, 'bone/')
        save_images(brain_window, output_path, patient_id, 'brain/')
        save_images(imgs, output_path, patient_id, 'all/')
        save_images(radiologist_window, output_path, patient_id, 'approx_five_mm/')
        save_images(rgb_window, output_path, patient_id, 'rgb_channels/')

        del patient[:]
        del imgs
        del new_imges
        del subdural_window
        del brain_window
        del bone_window
        del radiologist_window
        del rgb_window
