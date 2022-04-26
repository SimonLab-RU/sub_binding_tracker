import puncta_tracker as tracker
import numpy as np
from skimage import io
import pandas as pd
import data
import glob
import os


def batch_gather_dots_from_directory(input_directory: str, output_directory: str, blob_threshold: int):
    file_names = os.listdir(input_directory)
    file_names = [file for file in file_names if '.tif' in file]

    for file in file_names:
        dots = gather_dots(input_directory+file, blob_threshold)
        dots.to_csv('{}{}_threshold_{}_dot_database.csv'.format(output_directory, file[:-4], str(blob_threshold)))


def gather_dots(data_image_file_path: str, blob_threshold):
    print('currently finding dots in sample...' + data_image_file_path)
    data_image = io.imread(data_image_file_path)
    data_image = np.asarray(data_image, dtype=np.float64)
    dots = tracker.find_puncta(data_image,
                               target_radius=5,
                               blob_min_radius=3,
                               blob_threshold=blob_threshold)
    return dots


def filter_dots_by_intensity_and_background(data_image_file_path: str, associated_dot_database: pd.DataFrame):
    data_image = io.imread(data_image_file_path)
    background_mean = data_image.mean()
    background_std = data_image.std()
    dot_mean = associated_dot_database['mean_intensity'].mean()
    dot_std = associated_dot_database['mean_intensity'].std()

    return data.filter_dots(dot_dataframe=associated_dot_database,
                            max_blob_r=5,
                            mean_threshold=background_mean + 2 * background_std,
                            max_intensity=dot_mean + 3 * dot_std)


def batch_filter_gaussian_fitted_dots(input_directory: str,
                                      output_directory: str,
                                      database_file_unique_keyword=None,
                                      max_gaussian_deviation=1.5,
                                      max_gaussian_height=1000,
                                      max_elliptic_ratio=1.2,
                                      max_mean_height_percentage_difference=0.2):

    image_files = glob.glob(input_directory+'*.tif')
    dot_database_files = glob.glob(input_directory+'*.csv')

    if database_file_unique_keyword is not None:
        dot_database_files = [file for file in dot_database_files if database_file_unique_keyword in file]

    image_to_dot_pairing = {}

    for image_file in image_files:
        associated_dot_database = [file for file in dot_database_files if image_file[:-4] in file]
        image_to_dot_pairing[image_file] = associated_dot_database[0]

    for image_file, dot_database_file in image_to_dot_pairing.items():
        associated_dot_database = pd.read_csv(dot_database_file, index_col=0)
        print('Currently filtering dot database for {} '.format(image_file.split(os.sep)[-1]))
        filtered_database = filter_dots_by_intensity_and_background(data_image_file_path=image_file,
                                                                    associated_dot_database=associated_dot_database)

        # Note that mean_height_percentage_difference is a shortcut for assessing whether the Gaussian peak is at the
        # center of the dot, but the way  the mean is calculated is not accessible from outside the data.filter_dots()
        # function. This might be changed in the future, but for now, this works only when the mean is calculated using
        # a 5x5 window and Gaussian fitted with 11x11 dot. The simple logic behind this filter is that the brightness of
        # the geometric center of the dot should not deviate much from the height of the 2D Gaussian if the peak of the
        # 2D Gaussian is at the geometric center.
        filtered_database = data.filter_dots(
            filtered_database,
            max_gaussian_deviation=max_gaussian_deviation,
            max_gaussian_height=max_gaussian_height,
            max_elliptic_ratio=max_elliptic_ratio,
            max_mean_height_percentage_difference=max_mean_height_percentage_difference
        )

        filtered_database.reset_index()
        output_file_name = image_file.split(os.sep)[-1][:-4]
        filtered_database.to_csv('{}{}_filtered_gaussian_fitted_dots.csv'.format(output_directory, output_file_name))


def gaussian_fit_dots(data_image_file_path: str, associated_dot_database: pd.DataFrame, gaussian_fit_diameter: int):
    data_image = io.imread(data_image_file_path)
    data_image = np.asarray(data_image, dtype=np.float64)

    return data.add_gaussian_fit_params_to_dot_database(associated_dot_database, data_image, gaussian_fit_diameter)


def batch_gaussian_fit_dots(input_directory: str, output_directory: str, gaussian_fit_diameter=11):
    # The function takes all the dot-dababase files and the associated tif files and do gaussian fitting on the dots
    image_files = glob.glob(input_directory+'*.tif')
    dot_database_files = glob.glob(input_directory+'*.csv')
    image_to_dot_pairing_map = {}

    # map dot database to images
    for image_file in image_files:
        associated_dot_database = [file for file in dot_database_files if image_file[:-4] in file]
        image_to_dot_pairing_map[image_file] = associated_dot_database[0]

    for image_file, dot_database_file in image_to_dot_pairing_map.items():
        associated_dot_database = pd.read_csv(dot_database_file, index_col=0)
        print('Currently processing sample {}'.format(image_file.split(os.sep)[-1]))
        fitted_gaussian_database = gaussian_fit_dots(image_file, associated_dot_database, gaussian_fit_diameter)

        output_file_name = image_file.split(os.sep)[-1][:-4]
        fitted_gaussian_database.to_csv('{}{}_gaussian_fitted_dots.csv'.format(output_directory, output_file_name))


def batch_tracing(input_directory: str,
                  output_directory: str,
                  database_file_unique_keyword=None,
                  max_frame_gap=0,
                  max_spatial_jump=0,
                  frame_time_interval=1):

    dot_database_files = glob.glob(input_directory + '*.csv')

    if database_file_unique_keyword is not None:
        dot_database_files = [file for file in dot_database_files if database_file_unique_keyword in file]

    for dot_database_file in dot_database_files:
        dots = pd.read_csv(dot_database_file, index_col=0)
        print('Tracing {}'.format(dot_database_file.split(os.sep)[-1]))
        dot_trace_assignment = tracker.simple_tracker(dots, max_frame_gap, max_spatial_jump)
        traces = data.compile_trace_data(dots_database=dots,
                                         dot_trace_mapping=dot_trace_assignment,
                                         frame_time_interval=frame_time_interval)
        output_file_name = dot_database_file.split(os.sep)[-1][:-4]
        dot_trace_map = pd.DataFrame(list(dot_trace_assignment.items()), columns=['dot_ID', 'trace_ID'])
        traces.to_csv('{}{}_traces.csv'.format(output_directory, output_file_name))
        dot_trace_map.to_csv('{}{}_mapping.csv'.format(output_directory, output_file_name))


def batch_filter_traces(input_directory: str,
                        output_directory: str,
                        database_file_unique_keyword=None,
                        min_spatial_difference=0,
                        gap_threshold_for_eliminate_co_localized_traces=0,
                        min_x_coor=10,
                        max_x_coor=790,
                        min_y_coor=10,
                        max_y_coor=790):

    trace_database_file_names = glob.glob(input_directory+'*.csv')

    if database_file_unique_keyword is not None:
        trace_database_file_names = [file for file in trace_database_file_names if database_file_unique_keyword in file]

    for trace_database_file_name in trace_database_file_names:
        trace_data = pd.read_csv(trace_database_file_name, index_col=0)
        print('Filtering trace data {}'.format(trace_database_file_name.split(os.sep)[-1]))

        if min_spatial_difference > 0:
            trace_data = data.remove_concurrent_and_overlapping_traces(trace_data, min_spatial_difference)

        if gap_threshold_for_eliminate_co_localized_traces > 0:
            trace_data = data.remove_traces_with_long_gaps(trace_data, max_spatial_difference=2,
                                                           gap_threshold=gap_threshold_for_eliminate_co_localized_traces)

        trace_data = data.filter_trace_by_start_end_and_xy(trace_data, min_x_coor, max_x_coor, min_y_coor, max_y_coor)
        output_file_name = trace_database_file_name.split(os.sep)[-1][:-4]
        trace_data.to_csv('{}{}_cropped.csv'.format(output_directory, output_file_name))


def batch_tally_trace_dwell_times(input_directory: str,
                                  output_directory: str,
                                  input_file_unique_keyword=None,
                                  output_file_keyword='tally'):

    trace_database_file_names = glob.glob(input_directory+'*.csv')

    if input_file_unique_keyword is not None:
        trace_database_file_names = [file for file in trace_database_file_names if input_file_unique_keyword in file]

    for trace_database_file_name in trace_database_file_names:
        trace_data = pd.read_csv(trace_database_file_name, index_col=0)
        dwell_time_tally = data.count_instances(trace_data, 'dwell_time')
        output_file_name = trace_database_file_name.split(os.sep)[-1][:-4]
        dwell_time_tally.to_csv('{}{}_{}.csv'.format(output_directory, output_file_name, output_file_keyword))


def adjust_dwell_time_by_frame_intervals(input_directory: str,
                                         output_directory: str,
                                         frame_interval,
                                         input_file_unique_keyword=None):

    trace_database_file_names = glob.glob(input_directory + '*.csv')

    if input_file_unique_keyword is not None:
        trace_database_file_names = [file for file in trace_database_file_names if input_file_unique_keyword in file]

    output_file_unique_keywords = ''
    if input_directory == output_directory:
        output_file_unique_keywords = '_processed'

    for trace_database_file_name in trace_database_file_names:
        trace_data = pd.read_csv(trace_database_file_name, index_col=0)
        trace_data['dwell_time'] = trace_data['dwell_by_frame'] * frame_interval
        output_file_name = trace_database_file_name.split(os.sep)[-1][:-4]
        trace_data.to_csv(('{}{}{}.csv'.format(output_directory, output_file_name, output_file_unique_keywords)))


def batch_filter_trace_sampling_space(input_directory: str,
                                      output_directory: str,
                                      dwell_time_limit=None,
                                      input_file_unique_keyword=None,
                                      output_file_unique_keyword=None):

    trace_database_file_names = glob.glob(input_directory + '*.csv')

    if input_file_unique_keyword is not None:
        trace_database_file_names = [file for file in trace_database_file_names if input_file_unique_keyword in file]

    if output_file_unique_keyword is None:
        if input_directory == output_directory:
            output_file_unique_keyword = '_processed'

    for trace_database_file_name in trace_database_file_names:
        trace_data = pd.read_csv(trace_database_file_name, index_col=0)
        filtered_data = data.filter_traces_sample_space(trace_data, dwell_time_limit)
        output_file_core_name = trace_database_file_name.split(os.sep)[-1][:-4]
        filtered_data.to_csv(('{}{}{}.csv'.format(output_directory, output_file_core_name, output_file_unique_keyword)))
