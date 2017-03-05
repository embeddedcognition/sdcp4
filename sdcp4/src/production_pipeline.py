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
from moviepy.editor import VideoFileClip
from calibration_processor import perform_undistort
from perspective_processor import perform_perspective_transform
from threshold_processor import perform_thresholding
from lane_processor import perform_educated_lane_line_pixel_search, perform_blind_lane_line_pixel_search, compute_lane_line_coefficients

#globals
calibration_object_points = None
calibration_image_points = None
previous_left_lane_line_coeff = None
previous_right_lane_line_coeff = None
previous_left_lane_fitted_poly = None
previous_right_lane_fitted_poly = None
#set source vertices for region mask
src_upper_left =  (517, 478)
src_upper_right = (762, 478)
src_lower_left = (0, 720)
src_lower_right = (1280, 720)
#package source vertices (points)
src_vertices = np.float32(
    [src_upper_left,
     src_lower_left,
     src_lower_right,
     src_upper_right])
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

#run the pipeline on the provided video
def execute_production_pipeline(my_calibration_object_points, my_calibration_image_points):
    #leverage globals
    global calibration_object_points
    global calibration_image_points
    #set globals
    calibration_object_points = my_calibration_object_points
    calibration_image_points= my_calibration_image_points
    #generate video
    clip_handle = VideoFileClip("test_video/project_video.mp4")
    image_handle = clip_handle.fl_image(process_frame)
    image_handle.write_videofile("output_video/processed_project_video.mp4", audio=False)

#process a frame of video
def process_frame(image):
    
    ###################################
    ## PERFORM DISTORTION CORRECTION ##
    ###################################
    
    #undistort image
    undistorted_image = perform_undistort(image, calibration_object_points, calibration_image_points)
    
    ###################################
    ## PERFORM PERSPECTIVE TRANSFORM ##
    ###################################

    #transform perspective (warp) - this will squish the depth of field in the source mapping into the height of the image, 
    #which will make the upper 3/4ths blurry, need to adjust dest_upper* y-values to negative to stretch it out and clear the transformed image up
    #we won't do that as we'll lose right dashes in the 720 pix height of the image frame 
    warped_undistorted_image = perform_perspective_transform(undistorted_image, src_vertices, dest_vertices)
    
    #######################################
    ## PERFORM COLOR/GRADIENT THRESHOLD  ##
    #######################################

    #apply thresholding to warped image and produce a binary result
    thresholded_warped_undistorted_image = perform_thresholding(warped_undistorted_image)
    
    #############################
    ## PERFORM LANE DETECTION  ##
    #############################
    
    #if we have previous polynomials, use them as a starting place for educated search
    
    #map out the left and right lane line pixel locations via windowed search
    left_lane_pixel_coordinates, right_lane_pixel_coordinates, _ = perform_blind_lane_line_pixel_search(thresholded_warped_undistorted_image, return_debug_image=False)
    
    #compute the polynomial coefficients for each lane line using the x and y pixel locations from the mapping function
    #we're fitting (computing coefficients of) a second order polynomial: f(y) = A(y^2) + By + C
    #we're fitting for f(y) rather than f(x), as the lane lines in the warped image are near vertical and may have the same x value for more than one y value 
    left_lane_line_coeff, right_lane_line_coeff = compute_lane_line_coefficients(left_lane_pixel_coordinates, right_lane_pixel_coordinates)
    
    #if the deviation between the current coefficients and the previous ones is > 5%, use previous
    
    #generate range of evenly spaced numbers over y interval (0 - 719) matching image height
    y_linespace = np.linspace(0, (thresholded_warped_undistorted_image.shape[0] - 1), thresholded_warped_undistorted_image.shape[0])
    
    #left lane fitted polynomial (f(y) = A(y^2) + By + C)
    left_lane_line_fitted_poly = (left_lane_line_coeff[0] * (y_linespace ** 2)) + (left_lane_line_coeff[1] * y_linespace) + left_lane_line_coeff[2]
    #right lane fitted polynomial (f(y) = A(y^2) + By + C)
    right_lane_line_fitted_poly = (right_lane_line_coeff[0] * (y_linespace ** 2)) + (right_lane_line_coeff[1] * y_linespace) + right_lane_line_coeff[2]
    
    #########################################
    ## PERFORM PROJECTION BACK ON TO ROAD  ##
    #########################################
    
    #create an image to draw the lines on
    warped_lane = np.zeros_like(warped_undistorted_image).astype(np.uint8)

    #recast the x and y points into usable format for fillPoly and polylines
    pts_left = np.array([np.transpose(np.vstack([left_lane_line_fitted_poly, y_linespace]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane_line_fitted_poly, y_linespace])))])
    pts = np.hstack((pts_left, pts_right))

    #draw the lane onto the warped blank image
    cv2.fillPoly(warped_lane, np.int_([pts]), (152, 251, 152))
    
    #draw fitted lines on image
    cv2.polylines(warped_lane, np.int_([pts_left]), False, color=(189,183,107), thickness=20, lineType=cv2.LINE_AA)
    cv2.polylines(warped_lane, np.int_([pts_right]), False, color=(189,183,107), thickness=20, lineType=cv2.LINE_AA)

    #transform perspective back to original
    warped_to_original_perspective = perform_perspective_transform(warped_lane, dest_vertices, src_vertices)

    #combine (weight) result with the original image
    return cv2.addWeighted(undistorted_image, 1, warped_to_original_perspective, 0.3, 0)