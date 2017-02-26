####################################################
## AUTHOR: James Beasley                          ##
## DATE: February 18, 2017                        ##
## UDACITY SDC: Project 4 (Advanced Lane Finding) ##
####################################################

#############
## IMPORTS ##
#############
from camera_calibration import generate_calibration_components, perform_undistort
from test_pipeline import test_execute_pipeline

################################
## PERFORM CAMERA CALIBRATION ##
################################

#inside corner count of chessboard calibration images
num_column_points = 9  #total inside corner points across the x-axis
num_row_points = 6     #total inside corner points across the y-axis
#path to calibration images
path_to_calibration_images = "camera_cal/*.jpg"

#generate calibration componenets used to perform undistort
calibration_object_points, calibration_image_points = generate_calibration_components(num_column_points, num_row_points, path_to_calibration_images)

######################
## TEST PIPELINE #####
######################

#test the execution of the pipeline stages (stages of the pipeline are  
test_execute_pipeline(calibration_object_points, calibration_image_points)

######################
## RUN PIPELINE ######
######################

#execute the pipeline - producing a video  
#execute_pipeline(calibration_object_points, calibration_image_points)

