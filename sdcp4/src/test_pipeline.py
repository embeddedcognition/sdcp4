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
from calibration_processor import perform_undistort
from perspective_processor import perform_perspective_transform
from threshold_processor import perform_thresholding
from lane_processor import map_lane_line_pixel_locations, compute_lane_line_coefficients

#test the production_pipeline components and produce outputs in the 'output_images' folder
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
    #undistort image - this undistorted image will be used to demonstrate the production_pipeline along the way (all outputs will be placed in 'output_images' folder)
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
    
    #export as black and white image (instead of current single channel binary image which would be visualized as blue (representing zeros) and red (representing positive values 1 or 255) 
    #scale to 8-bit (0 - 255) then convert to type = np.uint8
    thresholded_warped_undistorted_test_road_image_scaled = np.uint8((255 * thresholded_warped_undistorted_test_road_image) / np.max(thresholded_warped_undistorted_test_road_image))
    #stack to create final black and white image
    thresholded_warped_undistorted_test_road_image_bw = np.dstack((thresholded_warped_undistorted_test_road_image_scaled, thresholded_warped_undistorted_test_road_image_scaled, thresholded_warped_undistorted_test_road_image_scaled))
    #save image
    mpimg.imsave("output_images/thresholded_warped_straight_lines1.jpg", thresholded_warped_undistorted_test_road_image_bw)
    
    ########################
    ## TEST LANE FINDING  ##
    ########################
    
    #map out the left and right lane line pixel locations 
    leftx, lefty, rightx, righty, debug_image = map_lane_line_pixel_locations(thresholded_warped_undistorted_test_road_image, return_debug_image=True)
    
    #compute the polynomial coefficients for each lane line using the x and y pixel locations from the mapping function
    #we're fitting (computing coefficients of) a second order polynomial: f(y) = A(y^2) + By + C
    #we're fitting for f(y) rather than f(x), as the lane lines in the warped image are near vertical and may have the same x value for more than one y value 
    left_fit, right_fit = compute_lane_line_coefficients(leftx, lefty, rightx, righty)
    
    #plot the left and right fitted polynomials on the debug image
    #generate x and y values for plotting
    ploty = np.linspace(0, thresholded_warped_undistorted_test_road_image.shape[0]-1, thresholded_warped_undistorted_test_road_image.shape[0])
    
    #left lane fitted polynomial (f(y) = A(y^2) + By + C)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right lane fitted polynomial (f(y) = A(y^2) + By + C)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #draw the fitted polynomials on the debug_image and export
    #recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    #pts = np.hstack((pts_left, pts_right))
    #draw lines
    cv2.polylines(debug_image, np.int_([pts_left]), False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(debug_image, np.int_([pts_right]), False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    #save image
    mpimg.imsave("output_images/fitted_polynomials_straight_lines1.jpg", debug_image)
    
    ######################################
    ## TEST PROJECTION BACK ON TO ROAD  ##
    ######################################
    
    #create an image to draw the lines on
    color_warp = np.zeros_like(warped_undistorted_test_road_image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    #draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    #transform perspective back to original
    warped_to_original = perform_perspective_transform(color_warp, dest_vertices, src_vertices)

    #combine the result with the original image
    projected_lane = cv2.addWeighted(undistorted_test_road_image, 1, warped_to_original, 0.3, 0)
    
    #save image
    mpimg.imsave("output_images/projected_lane_straight_lines1.jpg", projected_lane)