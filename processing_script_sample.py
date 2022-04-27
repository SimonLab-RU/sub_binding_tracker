import processing

# find dots from time lapse movie *.tiff files in a directory using LoG method
# the threshold, set between 6-7, is used for this publication, although optimization might be needed
# to examine the effect of LoG threshold, check the functions in blob_detector.py module
# the gathered dots are exported to the designated output folder as csv files,
# with the added keyword _dot_database in the file name
processing.batch_gather_dots_from_directory('Input/', 'Input/', 7)

# fit the gaussian function on the dots. In order to do this, in the designated input folder the dot database must come
# with the corresponding image file. The image files and the dot databases are matched based on the file name.
# Specifically, the dot database file name must contain the full name of the corresponding image file (except for the
# .tif extension) in order to be matched
# the numbered parameter indicates the desired diameter of the fitted Gaussian function.
# the fitted Gaussian parameters for the dots are exported into an designated folder, containing the keyword "gaussian"
# in the file name.
# This function is the most time consuming step in the entire processing script
processing.batch_gaussian_fit_dots('Input/', 'Input/', 11)

# filter the Gaussian fitted dots by shape, brightness, etc. Similar to processing.batch_gaussian_fit_dots(), this
# function requires the gaussian_fitted dot databases to be name matched with the imaging file. However, I added a
# keyword method distinguish gaussian_fitted dots from other csv files in the input directory. This might be added to
# processing.batch_gaussian_fit_dots() in the future. The filtered dots are exported in the designated directory, with
# the keyword "filtered_gaussian" added to the file name
processing.batch_filter_gaussian_fitted_dots(input_directory='Input/',
                                             output_directory='Input/',
                                             database_file_unique_keyword='gaussian',
                                             max_gaussian_deviation=2,
                                             max_gaussian_height=1000,
                                             max_elliptic_ratio=2,
                                             max_mean_height_percentage_difference=0.2)

# construct traces from filtered Gaussian-fitted dot databases. This function does not require the imaging file.
# the method outputs a trace database as well as a dot-trace map, where the dot IDs are mapped with trace-IDs. This map
# file can be used for future utilities such as for building FRET analysis tools.
processing.batch_tracing(input_directory='Input/',
                         output_directory='Input/',
                         database_file_unique_keyword='filtered_gaussian',
                         max_frame_gap=5,
                         max_spatial_jump=2,
                         frame_time_interval=2)

# traces are filtered based on proximity to each other, proximity to the edge of the image, whether have defined start
# or end, and whether two traces reside the same x-y coordinate but at different time points of the movie.
# note that the filter does not eliminate traces that last only one frame, but these traces should be eventually excluded
# when constructing the histogram frmo the traces.
# the output is a database of traces, which can be tallied or manipulated easily in any programs that read csv files. 
processing.batch_filter_traces('Input/',
                               'Output/',
                               database_file_unique_keyword='traces',
                               gap_threshold_for_eliminate_co_localized_traces=100,
                               min_spatial_difference=15)
