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
from perspective_transform import perform_perspective_transform
from color_gradient_thresholding import perform_thresholding

#test the pipeline components and produce outputs in the 'output_images' folder
def test_execute_pipeline(calibration_object_points, calibration_image_points):
    
    ################################
    ## TEST CAMERA CALIBRATION #####
    ################################
    
    #test camera calibration by undistorting a test chessboard image
    #load image
    test_chessboard_image = mpimg.imread("camera_cal/calibration1.jpg")
    #undistort image
    undistorted_test_chessboard_image = perform_undistort(test_chessboard_image, calibration_object_points, calibration_image_points)
    #save image
    mpimg.imsave("output_images/undistorted_calibration1.jpg", undistorted_test_chessboard_image)

    #test camera calibration by undistorting a test road image
    #load image
    test_road_image = mpimg.imread("test_images/straight_lines1.jpg")
    #undistort image - this undistorted image will be used to demonstrate the pipeline along the way (all outputs will be placed in 'output_images' folder)
    undistorted_test_road_image = perform_undistort(test_road_image, calibration_object_points, calibration_image_points)
    #save image
    mpimg.imsave("output_images/undistorted_straight_lines1.jpg", undistorted_test_road_image)

    ################################
    ## TEST PERSPECTIVE TRANSFORM ##
    ################################

    #drawing parameters
    line_color = [255, 0, 0] #red
    line_thickness = 3

    #set source vertices for region mask
    src_upper_left =  (472, 500)
    src_upper_right = (806, 500)
    src_lower_left = (0, 720)
    src_lower_right = (1280, 720)

    #package source vertices (points)
    src_vertices = np.float32(
        [src_upper_left,
         src_lower_left,
         src_lower_right,
         src_upper_right])

    #draw lines on test road image to display source vertices 
    src_vertices_image = undistorted_test_road_image.copy() #copy as not to affect original image
    cv2.line(src_vertices_image, src_upper_left, src_upper_right, line_color, line_thickness)
    cv2.line(src_vertices_image, src_upper_left, src_lower_left, line_color, line_thickness)
    cv2.line(src_vertices_image, src_upper_right, src_lower_right, line_color, line_thickness)
    cv2.line(src_vertices_image, src_lower_left, src_lower_right, line_color, line_thickness)
    #save image
    mpimg.imsave("output_images/src_vertices_straight_lines1.jpg", src_vertices_image)

    #set destination vertices (for perspective transform)
    dest_upper_left = (0, 0)
    dest_upper_right = (1280, 0)
    dest_lower_left = (0, 720)
    dest_lower_right = (1280, 720)

    #package destination vertices (points)
    dest_vertices = np.float32(
        [dest_upper_left,
         dest_lower_left,
         dest_lower_right,
         dest_upper_right])

    #transform perspective (warp)
    warped_undistorted_test_road_image = perform_perspective_transform(undistorted_test_road_image, src_vertices, dest_vertices)
    #save image
    mpimg.imsave("output_images/warped_straight_lines1.jpg", warped_undistorted_test_road_image)

    #draw lines on warped test road image to check alignment of lanes
    lane_alignment_warped_undistorted_test_road_image = warped_undistorted_test_road_image.copy() #copy as not to affect original image
    #set lane alignment verticies to check correctness of perspective transform
    lane_alignment_upper_left = (205, 0)
    lane_alignment_upper_right = (1105, 0)
    lane_alignment_lower_left = (205, 720)
    lane_alignment_lower_right = (1105, 720)
    #draw lane alignment lines on warped image
    cv2.line(lane_alignment_warped_undistorted_test_road_image, lane_alignment_upper_left, lane_alignment_upper_right, line_color, line_thickness)
    cv2.line(lane_alignment_warped_undistorted_test_road_image, lane_alignment_upper_left, lane_alignment_lower_left, line_color, line_thickness)
    cv2.line(lane_alignment_warped_undistorted_test_road_image, lane_alignment_upper_right, lane_alignment_lower_right, line_color, line_thickness)
    cv2.line(lane_alignment_warped_undistorted_test_road_image, lane_alignment_lower_left, lane_alignment_lower_right, line_color, line_thickness)
    #save image
    mpimg.imsave("output_images/lane_alignment_warped_straight_lines1.jpg", lane_alignment_warped_undistorted_test_road_image)

    ####################################
    ## TEST COLOR/GRADIENT THRESHOLD  ##
    ####################################

    #apply thresholding to warped image and produce a binary result
    thresholded_warped_undistorted_test_road_image = perform_thresholding(warped_undistorted_test_road_image)
    #save image
    mpimg.imsave("output_images/thresholded_warped_straight_lines1.jpg", thresholded_warped_undistorted_test_road_image)