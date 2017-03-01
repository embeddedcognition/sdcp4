####################################################
## AUTHOR: James Beasley                          ##
## DATE: February 18, 2017                        ##
## UDACITY SDC: Project 4 (Advanced Lane Finding) ##
####################################################

#############
## IMPORTS ##
#############
import cv2

#transform the perspective of the supplied undistorted image
def perform_perspective_transform(undistorted_image, src_points, dest_points):
    #get perspective matrix based on source and destination point mapping
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
    #warp image to destination perspective
    return cv2.warpPerspective(undistorted_image, perspective_matrix, (undistorted_image.shape[1], undistorted_image.shape[0]), flags=cv2.INTER_LINEAR)