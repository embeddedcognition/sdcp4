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
import glob
import cv2

########################
## CAMERA CALIBRATION ##
########################

#inside corner count of chessboard calibration images
num_inside_corners_x = 9
num_inside_corners_y = 6

#load calibration image file path list
calibration_image_file_path_list = glob.glob("camera_cal/*.jpg")
#enumerate calibration image file path list, loading each image, converting to grayscale, then retrieving inside corner points
for cur_calibration_image_file_path in calibration_image_file_path_list:
    #load image located at cur_calibration_image_file_path
    cur_calibration_image = mpimg.imread(cur_calibration_image_file_path)
    #convert image to grayscale
    cur_calibration_image_grayscale = cv2.cvtColor(cur_calibration_image, cv2.COLOR_RGB2GRAY)
    #find inside corners of chessboard
    ret, corners = cv2.findChessboardCorners(cur_calibration_image_grayscale, (num_inside_corners_x, num_inside_corners_y), None)
    

# prepare object points
#nx = 8#TODO: enter the number of inside corners in x
#ny = 6#TODO: enter the number of inside corners in y

# Make a list of calibration images
#fname = 'calibration_test.png'
#img = cv2.imread(fname)

# Convert to grayscale
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
#ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

