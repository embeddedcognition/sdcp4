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
import matplotlib.pyplot as plt
from threshold_processor import compute_hot_pixel_density_across_x_axis

#estimate the base location of the lane lines using the hot (value of 1) pixel density across the bottom half of the image
def estimate_base_location_of_lane_lines(image, export_debug_image=False):
    #set start position (y position...i.e., starting row number)
    offset = np.int(image.shape[0] / 2)
    #set window size (height...i.e., number of rows) that should be summed per x-axis column
    #this would normally be a fixed 'chunk', but to start, we're looking at the lower half of the image
    window_size = image.shape[0] - offset 
    
    #compute pixel peaks across the x-axis of the image
    hot_pixel_density_histogram = compute_hot_pixel_density_across_x_axis(image, offset, window_size)
    
    #if true export a debug image
    if (export_debug_image):
        #plot result
        plt.plot(hot_pixel_density_histogram, color='b', linewidth=1)
        plt.xlabel('Pixel position', fontsize=14)
        plt.ylabel('Hot pixel density', fontsize=14)
        plt.savefig("output_images/hot_pixel_density_histogram_straight_lines1.jpg")
        
    #return vector
    return hot_pixel_density_histogram

#compute the coefficients for the line equations by fitting a polynomial to the left and right lane pixel locations
def compute_lane_line_coefficients(leftx, lefty, rightx, righty):
    #fit a second order polynomial to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    #return left and right polynomial coefficients
    return (left_fit, right_fit)

#map out the lane line pixel locations in the supplied image 
def map_lane_line_pixel_locations(image, return_debug_image=False):
    #choose the number of sliding windows
    num_windows = 9
    #set height of windows
    window_height = np.int(image.shape[0] / num_windows)
    #set the width of the windows +/- margin
    margin = 100
    #set minimum number of pixels found to recenter window
    minpix = 50
    #create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    #if debug is set, the windows are visualized on a returned debug image
    debug_image = None
    
    #estimate base location of lane lines using the hot (value of 1) pixel density counts in the lower half of the image 
    hot_pixel_density_histogram = estimate_base_location_of_lane_lines(image, export_debug_image=return_debug_image)
    
    #locate the peak of the left and right halves of the histogram
    #these will be the starting point for the left and right lane lines
    #divide the vector in half (get midpoint)
    midpoint = np.int(hot_pixel_density_histogram.shape[0] / 2)
    #return the indices of the largest values in the vector from 0 to (midpoint - 1)
    leftx_base = np.argmax(hot_pixel_density_histogram[:midpoint])
    #return the indices of the largest values in the vector from (midpoint + 1) to (hot_pixel_density_histogram.shape[0] - 1)
    #also add in the index of the midpoint, since it was not counted in the left hand side 
    rightx_base = np.argmax(hot_pixel_density_histogram[midpoint:]) + midpoint
    
    #current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    #identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #if true return a debug image
    if (return_debug_image):
        #create an output image to draw on and visualize the result
        debug_image = np.dstack((image, image, image))*255

    #step through the windows one by one
    for window in range(0, num_windows):
        #identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window+1)*window_height
        win_y_high = image.shape[0] - window*window_height
    
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
    
        #if true return a debug image
        if (return_debug_image):
            #draw the windows on the visualization image
            cv2.rectangle(debug_image,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(debug_image,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    
        #identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    
        #append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
    
        #if we found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    #concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    #extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    #if true return a debug image
    if (return_debug_image):
        #color all left lane pixels red
        debug_image[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #color all right lane pixels blue
        debug_image[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        #fyi: since we color the line pixels last, the window rectangles will look like they're in back of the identified lines (in a layer below)

    #return the left and right lane line pixel locations and debug image
    return (leftx, lefty, rightx, righty, debug_image)
    