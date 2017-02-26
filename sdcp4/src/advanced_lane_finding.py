####################################################
## AUTHOR: James Beasley                          ##
## DATE: February 18, 2017                        ##
## UDACITY SDC: Project 4 (Advanced Lane Finding) ##
####################################################

#############
## IMPORTS ##
#############
import numpy as np
import matplotlib.image as mpimg
import cv2
from camera_calibration import generate_calibration_components, perform_undistort

########################
## CAMERA CALIBRATION ##
########################

#inside corner count of chessboard calibration images
num_column_points = 9  #total inside corner points across the x-axis
num_row_points = 6     #total inside corner points across the y-axis
#path to calibration images
path_to_calibration_images = "camera_cal/*.jpg"

#generate calibration componenets used to perform undistort
calibration_object_points, calibration_image_points = generate_calibration_components(num_column_points, num_row_points, path_to_calibration_images)

#test camera calibration by undistorting a test chessboard image
#load image
test_chessboard_image = mpimg.imread("camera_cal/calibration1.jpg")
#undistort image
undistorted_test_chessboard_image = perform_undistort(test_chessboard_image, calibration_object_points, calibration_image_points)
#save image
mpimg.imsave("output_images/undistorted_calibration1.jpg", undistorted_test_chessboard_image)

#test camera calibration by undistorting a test road image
#load image
test_road_image = mpimg.imread("test_images/test4.jpg")
#undistort image
undistorted_test_road_image = perform_undistort(test_road_image, calibration_object_points, calibration_image_points)
#save image
mpimg.imsave("output_images/undistorted_test4.jpg", undistorted_test_road_image)

###############################
## COLOR/GRADIENT THRESHOLD  ##
###############################



###########################
## PERSPECTIVE TRANSFORM ##
###########################

