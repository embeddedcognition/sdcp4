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

#compute the density white pixels across the x-axis within a specified y-axis chunk/window 
#remember, (0, 0) of an image is the top left corner of that image
def compute_white_pixel_density_across_x_axis(image, offset, window_size):
    #sum each column across the x-axis for the particular window in question (offset = y (row) start position, window_size = # of rows in column to sum)
    return np.sum(image[offset:offset+window_size, :], axis=0)