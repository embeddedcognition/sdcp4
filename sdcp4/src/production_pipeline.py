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
from lane_processor import map_lane_line_pixel_locations, compute_lane_line_coefficients

#globals
calibration_object_points = None
calibration_image_points = None
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
def execute_pipeline(my_calibration_object_points, my_calibration_image_points):
    #leverage globals
    #global calibration_object_points
    #global calibration_image_points
    #set globals
    calibration_object_points = my_calibration_object_points
    calibration_image_points= my_calibration_image_points
    #generate video
    clip_handle = VideoFileClip("test_video/project_video.mp4")
    image_handle = clip_handle.fl_image(process_frame)
    image_handle.write_videofile("output_video/processed_project_video.mp4", audio=False)

#run the pipeline on the provided video
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
    
    #map out the left and right lane line pixel locations 
    leftx, lefty, rightx, righty, _ = map_lane_line_pixel_locations(thresholded_warped_undistorted_image, return_debug_image=False)
    
    #compute the polynomial coefficients for each lane line using the x and y pixel locations from the mapping function
    #we're fitting (computing coefficients of) a second order polynomial: f(y) = A(y^2) + By + C
    #we're fitting for f(y) rather than f(x), as the lane lines in the warped image are near vertical and may have the same x value for more than one y value 
    left_fit, right_fit = compute_lane_line_coefficients(leftx, lefty, rightx, righty)
    
    #plot the left and right fitted polynomials on the debug image
    #generate x and y values for plotting
    ploty = np.linspace(0, thresholded_warped_undistorted_image.shape[0]-1, thresholded_warped_undistorted_image.shape[0])
    
    #left lane fitted polynomial (f(y) = A(y^2) + By + C)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right lane fitted polynomial (f(y) = A(y^2) + By + C)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #########################################
    ## PERFORM PROJECTION BACK ON TO ROAD  ##
    #########################################
    
    #create an image to draw the lines on
    color_warp = np.zeros_like(warped_undistorted_image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    #draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    #transform perspective back to original
    warped_to_original = perform_perspective_transform(color_warp, dest_vertices, src_vertices)

    #combine the result with the original image
    return cv2.addWeighted(undistorted_image, 1, warped_to_original, 0.3, 0)