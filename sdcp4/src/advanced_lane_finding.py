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

#a matrix of 3d coordinate values (each row holds an (x, y, z) point with each column being x, y, z)
#z will stay 0 since the chessboard is a plane, but we'll generate the x and y coordinates automatically
calibration_object_points_template = np.mgrid[0:num_inside_corners_x, 0:num_inside_corners_y, 0:1].T.reshape(-1, 3)

#lists to store object points and image points for all calibration images
calibration_object_points = [] #3d points in real world space
calibration_image_points = []  #2d points in image plane

#load calibration image file path list (file paths to images of chessboards to calibrate from) 
calibration_image_file_path_list = glob.glob("camera_cal/*.jpg")
#enumerate calibration image file path list, loading each image, converting to grayscale, then retrieving inside corner points
for cur_calibration_image_file_path in calibration_image_file_path_list:
    #load image located at cur_calibration_image_file_path
    cur_calibration_image = mpimg.imread(cur_calibration_image_file_path)
    #convert image to grayscale
    cur_calibration_image_grayscale = cv2.cvtColor(cur_calibration_image, cv2.COLOR_RGB2GRAY)
    #find image points (inside corners of chessboard) for cur_calibration_image_grayscale
    _, corners = cv2.findChessboardCorners(cur_calibration_image_grayscale, (num_inside_corners_x, num_inside_corners_y), None)
    #add image points found for cur_calibration_image_grayscale to image_points list
    calibration_image_points.append(corners)
    #add associated objects points for this calibrations image (same for all calibration images)
    calibration_object_points.append(calibration_object_points_template)
    
#convert to arrays
calibration_object_points = np.array(calibration_object_points)
calibration_image_points = np.array(calibration_image_points)

#transform an image to compensate for radial and tangential lens distortion
#undistort an image based on points taken from previous calibration images taken on that same camera 
def undistort_image(image, calibration_object_points, calibration_image_points):
    #get image size in (x, y)
    image_size = (image.shape[1], image.shape[0])
    #derive camera matrix (needed to transform 3d object points to 2d image points) and distortion coefficients
    #based on image and object points derived from calibration images taken on that same camera
    _, camera_matrix, dist_coeff, _, _ = cv2.calibrateCamera(calibration_object_points, calibration_image_points, image_size)
    #return the undistorted image
    return cv2.undistort(image, camera_matrix, dist_coeff)