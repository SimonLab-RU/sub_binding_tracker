from math import sqrt
from skimage.feature import blob_log
from skimage import io
import numpy as np
from scipy.signal import medfilt2d

import matplotlib.pyplot as plt


def blob_detector_for_image(file_name, radius, min_radius, blob_threshold, plot=False):
    # blob_detector from files; mainly for testing purpose
    image = io.imread(file_name)
    filtered_image = medfilt2d(image.astype(dtype='float64'))

    # convert image to np array
    filtered_image = np.asarray(filtered_image)
    blobs = blob_log(filtered_image,
                     max_sigma=1.5*radius/sqrt(2),
                     min_sigma=min_radius/sqrt(2),
                     num_sigma=radius/sqrt(2),
                     threshold=blob_threshold)

    blobs[:, 2] = blobs[:, 2]*sqrt(2) # calculate the resulting radius

    if plot:
        fig, ax = plt.subplots()
        plt.imshow(image, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='r', linewidth=1, fill=False)
            ax.add_patch(c)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    return blobs.astype(np.int64)


def detect_blobs(image_slice_array, radius, min_radius, blob_threshold):
    # data array must be 2D numpy array representing the image slice, with its datatype being float32 or float64

    # filter the data using a 3x3 median filter to remove dead pixels and other potential artifacts
    filtered_data = medfilt2d(image_slice_array)

    # blobs is a 2D array of three columns; the columns are y, x, r
    blobs = blob_log(filtered_data,
                     max_sigma=1.5 * radius / sqrt(2),
                     min_sigma=min_radius / sqrt(2),
                     num_sigma=radius / sqrt(2),
                     threshold=blob_threshold)

    blobs[:, 2] = blobs[:, 2]*sqrt(2)  # calculate the resulting radius

    return blobs.astype(np.int64)

