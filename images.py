from skimage import util
import numpy as np
from scipy import optimize
import math


def image_crop(x, y, box_size, source_image_array):
    # The function crops the source image (2d array) down to the specified box,
    # which is centered at the specified (x,y) coordinate

    l = int(box_size/2)
    x1 = int(x - l)
    x2 = int(x + l + 1)
    y1 = int(y - l)
    y2 = int(y + l + 1)

    return source_image_array[y1:y2, x1:x2]


def image_crop_with_padding(x, y, box_size, source_image_array, pad_width):
    """The function first pads around the source image with the specified pad_width, then
    crops the source image (2d array) down to the specified box,
    which is centered at the specified (x,y) coordinate"""

    # calculate the new coordinate of the ROI after padding
    y = int(y + pad_width)
    x = int(x + pad_width)

    # Use skimage.util to pad around the image
    padded_image_array = util.numpy_pad(source_image_array, int(pad_width), mode="constant")

    # Pass the padded image to the
    cropped_image = image_crop(x,y, box_size, padded_image_array)

    return cropped_image


def distance_to_closest_border_table(matrix_shape):
    # this function generates a 2D numpy array according to the shape specified;
    # the value of each item is the distance to the closest border of the image. The pixel on the border of the image
    # has a value of zero on this table
    # the argument "matrix_shape" is a 2-tuple in the form of (y,x)
    distance_table = np.zeros(matrix_shape)
    matrix_height, matrix_width = matrix_shape

    for y in range(matrix_height):
        for x in range(matrix_width):

            # calculate distance to each border
            distance_to_left = x
            distance_to_right = matrix_width - distance_to_left - 1
            distance_to_top = y
            distance_to_bottom = matrix_height - distance_to_top - 1

            # define the matrix using the minimal value to the border
            distance_table[y][x] = min(distance_to_left,
                                       distance_to_right,
                                       distance_to_top,
                                       distance_to_bottom)

    return distance_table


def create_circular_mask(source_h, source_w, mask_diameter):
    """The function creates a circular ROI_mask of specified diameter that fits an source_image"""

    # center is expressed as [y, x]
    center = [int(source_w / 2), int(source_h / 2)]
    radius = mask_diameter/2

    # the function below creates two vectors, the row vector x and the column vector y
    # x y can be viewed as index arrays for (x,y) coordinates
    y, x = np.ogrid[:source_h, :source_w]

    # the matrix operation below follows the broadcasting guideline of numpy. The resulting array is the distance array
    # from the center
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius

    return mask


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()

    # the elements of the tuples are: height, x, y, width_x, width_y
    return height, x, y, width_x, width_y


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    # I got this function directly from a scipy cookbook, and this one seems to have stylistic problems with PEP 8
    # this function seems to be able to fit over anything.
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)

    return p


def dot_gaussian_quality_filter(xcoor: int, ycoor: int, image_slice: np.ndarray, diameter_of_fit,
                                max_centroid_deviation, min_height_threshold, max_elliptic_aspect_ratio):
    # this function is also depreciated

    """The function takes the x and y coordinates of a dot, and the image_slice that the dot appears in. The function
    returns 1, if the dot meets the criteria. Otherwise, the function will return 0; in the future, I'm planning to add
    a sliding scale between 0 and 1 to reflect the quality of the dots"""

    # At the moment I don't have a good criteria. I probably don't have to mess around centroid_deviation, but I guess
    # the height_threshold must be changed accordingly between experiments
    # by default, diameter of fit should be around 11 for images from 100x objective and 6.5x6.5 micron pixels

    # first, crop out the region for fitting. Note that if the punctum is too close to the edge the gaussian probably
    # won't work properly.

    padding_required = check_padding_requirement(xcoor, ycoor, image_slice, diameter_of_fit)

    if padding_required:
        distance_to_edge = distance_to_closest_border_table(image_slice.shape)
        padding_width = int(diameter_of_fit / 2) - distance_to_edge[ycoor][xcoor]
        data = image_crop_with_padding(xcoor, ycoor, diameter_of_fit, image_slice, padding_width)

    else:
        data = image_crop(xcoor, ycoor, diameter_of_fit, image_slice)

    # just a reminder: fitgaussian returns (height, x, y, width_x, width_y)
    height, x, y, width_x, width_y = fitgaussian(data)

    centroid_x = centroid_y = diameter_of_fit / 2
    squared_deviation = (x - centroid_x) ** 2 + (y - centroid_y) ** 2
    elliptic_aspect_ratio = max(width_x, width_y) / min(width_x, width_y)

    if (squared_deviation < max_centroid_deviation**2) \
            and (elliptic_aspect_ratio < max_elliptic_aspect_ratio) \
            and (height > min_height_threshold):
        return 1

    else:
        return 0


def dot_gaussian_fit(xcoor: int, ycoor: int, image_slice: np.ndarray, diameter_of_fit):

    padding_required = check_padding_requirement(xcoor, ycoor, image_slice, diameter_of_fit)

    if padding_required:
        distance_to_edge = distance_to_closest_border_table(image_slice.shape)
        padding_width = int(diameter_of_fit / 2) - distance_to_edge[ycoor][xcoor]
        data = image_crop_with_padding(xcoor, ycoor, diameter_of_fit, image_slice, padding_width)
    else:
        data = image_crop(xcoor, ycoor, diameter_of_fit, image_slice)

    return fitgaussian(data)


def measure_stack_profile(x, y, source_stack, measuring_disc_diameter):
    # use an roi_mask, which is an array of boolean values to filter the source image
    # first, crop down to a square according to x & y, and the measuring_disc_diameter

    cropped_stack = stack_crop(x, y, source_stack, measuring_disc_diameter)

    result = []
    roi_mask = create_circular_mask(*[measuring_disc_diameter]*3)

    # first, iterate through the stack by slices
    for image_slice in cropped_stack:

        # ~ is "not" in numpy; make the area of no interest 0
        masked_image = image_slice
        masked_image[~roi_mask] = 0

        # calculate means by flatten the masked_image and remove zeros, then calculate mean
        masked_image = masked_image.flatten()
        result.append(np.mean(masked_image[masked_image != 0]))

    return np.array(result)


def stack_crop(x, y, source_image_stack_array, box_size):
    # please use an odd number as box_size
    # (for now; in the future I will automatically correct this if there's an even number)
    # source image stack is a ndarray, with the coordinate arranged as (z, y, x)
    # ROI yx is a list

    # unused statement: stack_depth = source_image_stack_array.shape[0]

    cropped_stack = []
    # make a distance lookup table by constructing a 2d matrix where the value is the distance to the closest edge
    # this is used to check if padding is needed
    distance_lookup_table = distance_to_closest_border_table(source_image_stack_array.shape[1:])
    padding_width = int(box_size/2) - distance_lookup_table[y][x]

    if padding_width > 0:

        # padding is required, use a special padding crop function defined elsewhere
        for source_stack_slice in source_image_stack_array:
            cropped_stack.append(image_crop_with_padding(x, y, box_size, source_stack_slice, padding_width))

    else:
        for source_stack_slice in source_image_stack_array:
            # The following command non-explicitly convert the ndarray to list
            cropped_stack.append(image_crop(x, y, box_size, source_stack_slice))

    return np.array(cropped_stack)


def measure_mean_at_xy(x, y, source_slice, measuring_disc_diameter):
    """The function measures a target area specified by the indicated (x,y) coordinate at the source_slice, using
    a measuring disc with the indicated diameter"""

    # This function is extremely confusing, due to the indexing convention of numpy arrays. I write this function this
    # way to replace the predecessor due to its performance issues. The current one, although confusing, is faster.

    h, w = source_slice.shape

    measuring_disc_radius = math.floor(measuring_disc_diameter/2)
    padding_required = False

    padding_width = [0, 0, 0, 0]  # for elements represent padding width added to the left, right, top, and bottom edges

    # check x edges
    if x - measuring_disc_radius < 0:
        x1 = 0
        x2 = x + measuring_disc_radius + 1  # the +1 simply reflects the indexing convention of numpy
        padding_required = True
        padding_width[0] = measuring_disc_radius - x

    elif (x + measuring_disc_radius) >= w:
        x1 = x - measuring_disc_radius
        x2 = w
        padding_required = True
        padding_width[1] = x + measuring_disc_radius - w + 1
    else:
        x1 = x - measuring_disc_radius
        x2 = x + measuring_disc_radius + 1

    # check y edges
    if y - measuring_disc_radius < 0:
        y1 = 0
        y2 = y + measuring_disc_radius + 1
        padding_required = True
        padding_width[2] = measuring_disc_radius - y
    elif (y + measuring_disc_radius) >= h:
        y1 = y - measuring_disc_radius
        y2 = h
        padding_required = True
        padding_width[3] = y + measuring_disc_radius - h + 1
    else:
        y1 = y - measuring_disc_radius
        y2 = y + measuring_disc_radius + 1

    # crop the source slice according to the edges
    cropped_target = source_slice[y1:y2, x1:x2]

    # prepare the masking disc to measure the mean
    roi_mask = create_circular_mask(*[measuring_disc_diameter] * 3)
    if padding_required:
        roi_mask_h, roi_mask_w = roi_mask.shape  # I write this way so that I can change the shape of the mask later
        roi_x1 = padding_width[0]
        roi_x2 = roi_mask_w - padding_width[1]
        roi_y1 = padding_width[2]
        roi_y2 = roi_mask_h - padding_width[3]
        roi_mask = roi_mask[roi_y1:roi_y2, roi_x1:roi_x2]

    cropped_target[~roi_mask] = 0
    cropped_target = cropped_target.flatten()
    result = np.mean(cropped_target[cropped_target != 0])

    return result


def check_padding_requirement(x, y, source_slice, measuring_disc_diameter):
    # this function is simplified from the measure_mean_at_xy function. I realize that I use this function quite a lot
    # so might just write this up
    h, w = source_slice.shape
    measuring_disc_radius = math.floor(measuring_disc_diameter/2)
    padding_required = False

    # check y edges
    if (x - measuring_disc_radius < 0) or \
            (x + measuring_disc_radius) >= w or \
            (y - measuring_disc_radius < 0) or \
            (y + measuring_disc_radius) >= h:

        padding_required = True

    return padding_required


def mean_square_displacement(x_data, y_data):
    r = np.sqrt(x_data**2 + y_data**2)
    diff = np.diff(r)  #this calculates r(t + dt) - r(t)
    diff_sq = diff**2
    MSD = np.mean(diff_sq)

    return MSD
