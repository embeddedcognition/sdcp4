####################################################
## AUTHOR: James Beasley                          ##
## DATE: February 18, 2017                        ##
## UDACITY SDC: Project 4 (Advanced Lane Finding) ##
####################################################

#############
## IMPORTS ##
#############
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from camera_calibration import perform_undistort
from perspective_transform import perform_perspective_transform
from color_gradient_thresholding import perform_thresholding
from lane_finder import compute_white_pixel_density_across_x_axis

#test the pipeline components and produce outputs in the 'output_images' folder
def test_execute_pipeline(calibration_object_points, calibration_image_points):
    
    #############################
    ## TEST CAMERA CALIBRATION ##
    #############################
    
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
    #scale to 8-bit (0 - 255) then convert to type = np.uint8
    thresholded_warped_undistorted_test_road_image_scaled = np.uint8((255 * thresholded_warped_undistorted_test_road_image) / np.max(thresholded_warped_undistorted_test_road_image))
    #stack to create final black and white image
    thresholded_warped_undistorted_test_road_image_bw = np.dstack((thresholded_warped_undistorted_test_road_image_scaled, thresholded_warped_undistorted_test_road_image_scaled, thresholded_warped_undistorted_test_road_image_scaled))
    #save image
    mpimg.imsave("output_images/thresholded_warped_straight_lines1.jpg", thresholded_warped_undistorted_test_road_image_bw)
    
    ########################
    ## TEST LANE FINDING  ##
    ########################
    
    #set start position (y position...i.e., starting row number)
    offset = np.int(thresholded_warped_undistorted_test_road_image.shape[0] / 2)
    #set window size (height...i.e., number of rows) that should be summed per x-axis column
    #this would normally be a fixed 'chunk', but to start, we're looking at the lower half of the image
    window_size = thresholded_warped_undistorted_test_road_image.shape[0] - offset 
    #compute pixel peaks across the x-axis of the image
    white_pixel_density_histogram = compute_white_pixel_density_across_x_axis(thresholded_warped_undistorted_test_road_image, offset, window_size)
    #plot result
    plt.plot(white_pixel_density_histogram, color='b', linewidth=1)
    plt.xlabel('Pixel position', fontsize=14)
    plt.ylabel('White pixel density', fontsize=14)
    plt.savefig("output_images/white_pixel_density_histogram_straight_lines1.jpg")
    
    #locate the peak of the left and right halves of the histogram
    #these will be the starting point for the left and right lane lines
    #divide the vector in half (get midpoint)
    midpoint = np.int(white_pixel_density_histogram.shape[0] / 2)
    #return the indices of the largest values in the vector from 0 to (midpoint - 1)
    leftx_base = np.argmax(white_pixel_density_histogram[:midpoint])
    #return the indices of the largest values in the vector from (midpoint + 1) to (white_pixel_density_histogram.shape[0] - 1)
    #also add in the index of the midpoint, since it was not counted in the left hand side 
    rightx_base = np.argmax(white_pixel_density_histogram[midpoint:]) + midpoint

    # Create an output image to draw on and visualize the result
    #out_img = np.dstack((final_warped_binary_image, final_warped_binary_image, final_warped_binary_image))*255