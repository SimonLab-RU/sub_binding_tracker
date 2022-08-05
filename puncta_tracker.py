import blob_detector
import numpy as np
import pandas as pd
from images import measure_mean_at_xy


def find_puncta(image_data_array, target_radius, blob_min_radius, blob_threshold):
    # The function takes an 3D image_array and parameters for defining the blob finder and returns a
    # dataFrame object; each column corresponds to a factor of the dot; see specific factors below.
    # First, initialize the dots database (which for now, is a dictionary) using individual factors
    # as the keys
    dots = {'frame': [],
            'dot_ID': [],
            'xcoor': [],
            'ycoor': [],
            'blob_r': [],
            'mean_intensity': []}

    current_id = int(1)  # initialize the dot ID counter
    current_frame = int(1)  # initialize the frame counter

    for image_slice in image_data_array:
        # report progress:
        print('Finding dots on frame {}'.format(current_frame))
        # Use blob_detector function to find blobs in the image_slice. The blob_detector applies a
        # median filter with 3x3 kernel before using the LoG blob detector.
        # The detect_blobs method then returns a nx3 numpy int64 array.
        # The reason to use int64 is that I don't do sub-pixel localization.

        dots_in_current_frame = blob_detector.detect_blobs(image_slice,
                                                           radius=target_radius,
                                                           min_radius=blob_min_radius,
                                                           blob_threshold=blob_threshold)

        # I want to add a viewer to tweak the parameters later; for now I'll check parameters using a test script

        number_of_dots = len(dots_in_current_frame)
        if number_of_dots == 0:
            # skip packing things if there's no dot found
            current_frame += 1
            continue

        # Since each element in the blob list is a 3-element list representing y, x, and r,
        # split dot list into three column vectors. Note that the indexing is different from using a list, namely that
        # I always need to specify the column as [0]
        ycoor, xcoor, r = np.hsplit(dots_in_current_frame, 3)
        # create frame number & dot IDs for the dot database
        # from here, I find it easier to use python list to append data to the database than using ndarray
        dot_ids = list(range(current_id, current_id + number_of_dots))
        dot_frames = [current_frame] * number_of_dots

        # now, calculate the mean intensity; the measuring_disc_diameter is set at 9 across the board
        # remember to change mean measuring disc diameter if this is modified
        means = []
        for i in range(number_of_dots):
            current_mean = measure_mean_at_xy(xcoor[i][0], ycoor[i][0], image_slice, measuring_disc_diameter=5)
            means.append(current_mean)

        # pack the factors into a dictionary. Each dot's parameters have the same index under each key
        # of the database. In the end, this database will be converted to a Pandas dataFrame, where the
        # parameters of the same dot will appear in the same row

        dots['frame'] = dots['frame'] + dot_frames
        dots['dot_ID'] = dots['dot_ID'] + dot_ids
        dots['xcoor'] = dots['xcoor'] + xcoor.flatten().tolist()
        dots['ycoor'] = dots['ycoor'] + ycoor.flatten().tolist()
        dots['blob_r'] = dots['blob_r'] + r.flatten().tolist()
        dots['mean_intensity'] = dots['mean_intensity'] + means

        # after adding the data to the dots database, update the dot ID and frame counters
        current_id = dot_ids[-1] + 1
        current_frame += 1

    return pd.DataFrame.from_dict(dots)


def find_puncta_in_single_frame_image(image_data_slice, target_radius, blob_min_radius, blob_threshold):

    dots = {'frame': [],
            'dot_ID': [],
            'xcoor': [],
            'ycoor': [],
            'blob_r': [],
            'mean_intensity': []}

    current_id = int(1)  # initialize the dot ID counter
    current_frame = int(1)
    means = []

    dots_in_current_frame = blob_detector.detect_blobs(image_data_slice,
                                                       radius=target_radius,
                                                       min_radius=blob_min_radius,
                                                       blob_threshold=blob_threshold)
    number_of_dots = len(dots_in_current_frame)
    if number_of_dots == 0:
        # skip packing things if there's no dot found
        print('No Dots Found')

    ycoor, xcoor, r = np.hsplit(dots_in_current_frame, 3)
    dot_ids = list(range(current_id, current_id + number_of_dots))
    dot_frames = [current_frame] * number_of_dots

    for i in range(number_of_dots):
        current_mean = measure_mean_at_xy(xcoor[i][0], ycoor[i][0], image_data_slice, measuring_disc_diameter=5)
        means.append(current_mean)

    dots['frame'] = dots['frame'] + dot_frames
    dots['dot_ID'] = dots['dot_ID'] + dot_ids
    dots['xcoor'] = dots['xcoor'] + xcoor.flatten().tolist()
    dots['ycoor'] = dots['ycoor'] + ycoor.flatten().tolist()
    dots['blob_r'] = dots['blob_r'] + r.flatten().tolist()
    dots['mean_intensity'] = dots['mean_intensity'] + means

    return pd.DataFrame.from_dict(dots)


def simple_tracker(dot_database: pd.DataFrame, max_frame_gaps, max_spatial_jumps):
    # The "simple_tracker" means that there's no consideration for branching events.
    # the "dot_database" needs to be a Pandas DataFrame object, with the following columns:
    # 'frame', 'dot_ID', 'xcoor', 'ycoor', 'blob_r', as generated by the "find_puncta" function.
    # The idea for the simple_tracker is that I will construct a dictionary that maps a dot ID to a trace ID;
    # the dot IDs are keys in the dictionary. On the other hand, once the dictionary contains a specific dot ID,
    # it means that the dot has been processed and has been assigned a trace ID.
    # max_special_jumps is the maximum distance between frames that are allowed; this is measured in pixels

    last_frame = max(dot_database['frame'])  # find out the last frame that has dots on it
    trace_assignment = {}  # initialize a trace-assignment database, where dot ID is mapped to a trace ID
    current_trace_id = 1  # initialize a trace ID

    # use for loop to iterate through all the frames, while keeping the frame number:
    for current_frame in range(1, last_frame+1):

        # slice the dot_database to only include dots found on the current frame
        dots_within_current_frame: pd.DataFrame = dot_database[dot_database['frame'] == current_frame]

        # Iterate through all dots in the frame, and find out if each dot has been assigned a trace ID. If it doesn't
        # have a trace ID, it means that the dot should be the start of a new trace.
        for index, row in dots_within_current_frame.iterrows():
            if row['dot_ID'] not in trace_assignment:

                if current_trace_id % 100 == 0:
                    print('Tracking trace #'+str(current_trace_id))

                # when the first dot of a new trace is found, assign the dot with a trace ID
                trace_assignment[row['dot_ID']] = current_trace_id
                current_xcoor = row['xcoor']  # initialize the xcoor for distance comparison
                current_ycoor = row['ycoor']  # initialize the ycoor for distance comparison
                next_frame = current_frame + 1  # initialize the frame counter for tracking
                remaining_gaps_allowed = max_frame_gaps  # initialize remaining gaps before tracking

                # I use a while loop to assign trace ID to subsequent dots
                trace_ended = False  # I know this is not the most elegant way to setup a while loop
                while not trace_ended:
                    # The while loop has two "break" statement: one is right below, where the loop reaches the end of
                    # the movie; the second one is when no more dots could be found.

                    # Terminate the while loop if already reaching the end of the movie
                    if next_frame > last_frame:
                        current_trace_id += 1
                        break

                    # Otherwise, find all the dots within the next frame
                    dots_within_next_frame: pd.DataFrame = dot_database[dot_database['frame'] == next_frame]

                    # narrow down the list of dots within the next frame of which the distance to the current dot
                    # is within the threshold
                    candidate = dots_within_next_frame[
                        (dots_within_next_frame['xcoor'] - current_xcoor)**2 +
                        (dots_within_next_frame['ycoor'] - current_ycoor)**2 <=
                        max_spatial_jumps**2]

                    # here are the possible cases for the candidate shortlist (dataFrame):
                    # 1) there's no candidate
                    # 2) there's a single candidate. This is the easiest possibility. Both other cases require
                    # subsequent discussion of sub-cases
                    # 3) there's more than one candidate
                    # so I will begin to work on the easiest case, and get back to other cases later

                    if len(candidate) == 1:
                        # Assign the current trace ID candidate dot ID. Note that [0] index is required because
                        # otherwise candidate.iloc['dot_ID'] would only return a Series object instead of the value
                        # contained within the object
                        trace_assignment[candidate.iloc[0]['dot_ID']] = current_trace_id

                        # Update parameters for searching the next frame. Use the coordinates of the dot just found for
                        # the searching the next frame. Refresh the gap counter
                        next_frame += 1
                        remaining_gaps_allowed = max_frame_gaps
                        current_xcoor = candidate.iloc[0]['xcoor']
                        current_ycoor = candidate.iloc[0]['ycoor']

                    elif len(candidate) > 1:

                        # Need to narrow down the candidate to a single one; right now, I'll only pick the closest dot
                        # to the current_xcoor and current_ycoor

                        squared_distances = (candidate['xcoor'] - current_xcoor)**2 + \
                                            (candidate['ycoor'] - current_ycoor)**2

                        # use the first occurrence of the minimum value of the squared distance as the index to pick the
                        # a candidate from the list of candidates

                        candidate = candidate.loc[squared_distances.idxmin()]

                        # Assign the current trace ID candidate dot ID and update parameters for the next search
                        trace_assignment[candidate['dot_ID']] = current_trace_id
                        next_frame += 1
                        remaining_gaps_allowed = max_frame_gaps
                        current_xcoor = candidate['xcoor']
                        current_ycoor = candidate['ycoor']

                    elif len(candidate) < 1:
                        # When no candidate found and it has not yet reached the end of the movie,
                        # there are two possibilities:
                        # 1) When no more gap is allowed, this is the end of the trace
                        # 2) When there's more gap allowed, I should search for the next frame
                            if remaining_gaps_allowed > 0:
                                # In this case, subtract one from the remaining gaps:
                                remaining_gaps_allowed -= 1
                                # then move to the next frame without updating current coordinates
                                # or making trace assignments
                                next_frame += 1
                            else:
                                # When all the allowed gaps are exhausted, update the trace_id for the series of dots,
                                # and terminate the while loop
                                current_trace_id += 1
                                break

            else:
                # If the dot already been assigned a trace ID, do nothing.
                continue

    return trace_assignment
