# TIRF_binding_tracker

This processing script is used to derive puncta dwell time from TIRF images. 
The script is organized into three layers. 

The frist layer deals with the most basic methods of data handling. The layer contains four modules:
- blob_detectror, for fiding blobs in an image using LoG method
- data, for importing and exporting csv files, applying data filters to the csv files, etc. 
- images, for functions such as measuring means, fitting Gaussians, cropping images, etc.
- puncta_tracker, for constructing traces from individual dots on frames of the image stacks. 

The second layer contains a single module, processing, which integrates the first layer functions by scripting the routine tasks.

The third layer is the user layer, where inputs and parameters are defined. By utilizing functions within the processing.py module, the user can perform specific tasks. An example is provided in the "processing_script_sample.py". 

