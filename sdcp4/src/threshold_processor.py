####################################################
## AUTHOR: James Beasley                          ##
## DATE: February 18, 2017                        ##
## UDACITY SDC: Project 4 (Advanced Lane Finding) ##
####################################################

#############
## IMPORTS ##
#############
import numpy as np
import cv2

#compute the density white pixels across the x-axis within a specified y-axis chunk/window 
#remember, (0, 0) of an image is the top left corner of that image
def compute_white_pixel_density_across_x_axis(image, offset, window_size):
    #sum each column across the x-axis for the particular window in question (offset = y (row) start position, window_size = # of rows in each column to sum)
    #this will sum from offset to ((offset + window_size) - 1)
    return np.sum(image[offset:offset+window_size, :], axis=0)

#apply region mask to an image
def region_of_interest(image, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(image)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#apply gaussian blur to an image
def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

#apply finite difference filter (Sobel) to an image
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    #take the derivative in x or y given orient = 'x' or 'y'
    if (orient == 'x'):
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    #take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    #scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8((255 * abs_sobel) / np.max(abs_sobel))
    #create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    #return result
    return sbinary

#applies a color/gradient threshold process to an undistorted and warped (perpective transformed - bird's eye view) rgb image
def perform_thresholding(warped_undistorted_image):
    #convert to hsv color-space to identify yellow and white lines
    hsv = cv2.cvtColor(warped_undistorted_image, cv2.COLOR_RGB2HLS)
    #extract all channels
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    #identify lines using custom filtering
    #hue
    h_threshold = (0, 50)
    #treat the original as immutable
    filtered_h = h.copy()
    #values outside of the threshold range are set to zero
    filtered_h[(h < h_threshold[0]) | (h > h_threshold[1])] = 0
    #saturation
    s_threshold = (140, 255)
    #treat the original as immutable
    filtered_s = s.copy()
    #values outside of the threshold range are set to zero
    filtered_s[(s < s_threshold[0]) | (s > s_threshold[1])] = 0
    #value (brightness)
    v_threshold = (140, 255)
    #treat the original as immutable
    filtered_v = v.copy()
    #values outside of the threshold range are set to zero
    filtered_v[(v < v_threshold[0]) | (v > v_threshold[1])] = 0
    #recombine filtered hsv channels
    filtered_hsv = np.dstack((filtered_h, filtered_s, filtered_v))
    #convert back to rgb color-space for display
    filtered_rgb = cv2.cvtColor(filtered_hsv, cv2.COLOR_HLS2RGB)     
    #convert to binary
    filtered_binary = cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2GRAY)
    filtered_binary[filtered_binary < 128] = 0    #black
    filtered_binary[filtered_binary >= 128] = 1   #white

    #convert to hls color space to add robustness to lane identification
    hls = cv2.cvtColor(warped_undistorted_image, cv2.COLOR_RGB2HLS)
    #extract l & s channels
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    #smooth channel (blurring first allows us to set a higher 'low' threshold - e.g., less noise)
    l_blurred = gaussian_blur(l, 45)
    #apply finite difference filter (Sobel - across x-axis)
    l_sobel_x = abs_sobel_thresh(l_blurred, orient='x', thresh=(45, 255))
    #apply finite difference filter (Sobel - across x-axis)
    s_sobel_x = abs_sobel_thresh(s, orient='x', thresh=(15, 255))
    ##apply basic value thresholding to raw channel as well
    s_filter = np.zeros_like(s)
    s_filter[(s >= 150) & (s <= 255)] = 1
    #'or' the two
    s_final = cv2.bitwise_or(s_sobel_x, s_filter)
    
    #compute the white pixel density score for the s-channel (build resiliency against degraded s-channel)
    #get the count of non-zero pixels in the image (i.e., how may 1's are there) - just counting the number of y-coordinates returned 
    #(could have also counted just the x-coordinates)
    white_pixel_count = len((s_final.nonzero())[0])
    #density score is the number of positive (white) pixels in the image divided by the total number of pixels in the image
    density_score = white_pixel_count / (s_final.shape[0] * s_final.shape[1])
    
    #combine the l and binary hsv channel as normal
    final_binary_image = cv2.bitwise_or(l_sobel_x, filtered_binary)
    
    #if the s-channel has a sufficiently low white pixel density
    #a good density score (binary s-channel with sold left and dashed right identified) will be ~0.03
    #a bad density score (binary s-channel with a lot of white clouding/blotching) will be ~0.40
    #0.15 is an arbitrary threshold that gives a lot of headroom for variation and noise
    if (density_score < 0.15):
        #combine the l and binary hsv with the s-channel (if not, leave it out)
        final_binary_image = cv2.bitwise_and(s_final, final_binary_image)
    
    #combine the three and return
    return final_binary_image