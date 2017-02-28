####################################################
## AUTHOR: James Beasley                          ##
## DATE: February 18, 2017                        ##
## UDACITY SDC: Project 4 (Advanced Lane Finding) ##
####################################################

#############
## IMPORTS ##
#############
from moviepy.editor import VideoFileClip

#run the pipeline on the provided video
def execute_pipeline(calibration_object_points, calibration_image_points):
    process_frame_function_handle = process_frame(calibration_object_points, calibration_image_points)
    video_handle = VideoFileClip("test_videos/project_video.mp4")
    image_handle = video_handle.fl_image(process_frame_function_handle)
    image_handle.write_videofile("output_videos/processed_project_video.mp4", audio=False)

#run the pipeline on the provided video
def process_frame(calibration_object_points, calibration_image_points):
    return None