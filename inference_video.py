import cv2
import logging
import os
from gaze_tracking import GazeTracking

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logging.debug('Starting the GazeTracking demo')

gaze = GazeTracking()

# Load the video file
video_path = 'naz_test.mp4'
logging.debug(f'Loading video from {video_path}')

# Check if the video file exists
if not os.path.exists(video_path):
    logging.error(f'Video file does not exist: {video_path}')
    exit()

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    logging.error(f'Error opening video file: {video_path}')
    exit()

# Get the video frame width, height, and frames per second (fps)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
logging.debug(f'Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}')

# Read the first frame to determine the image format
ret, first_frame = video.read()
if not ret:
    logging.error('Error reading the first frame of the video.')
    exit()

# Determine the format of the first frame
if first_frame.shape[2] == 3:
    image_format = 'RGB'
    logging.debug('First frame is in RGB format.')
else:
    logging.debug('First frame is not in RGB format, converting it.')
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    image_format = 'Converted to RGB'

# Reinitialize the video capture to start from the beginning
video.release()
video = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
output_path = 'naz_test_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    logging.error(f'Error opening video writer: {output_path}')
    exit()

frame_count = 0
while True:
    # We get a new frame from the video
    ret, frame = video.read()

    # If the frame was not read correctly, break the loop
    if not ret:
        logging.debug('No more frames to read or error reading frame.')
        break

    frame_count += 1
    logging.debug(f'Processing frame {frame_count}')

    # Convert the frame to RGB if it is not already
    if image_format == 'Converted to RGB':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Convert the frame back to BGR for saving if needed
    if image_format == 'Converted to RGB':
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Write the frame into the output video
    out.write(frame)

    logging.debug(f'Frame {frame_count} written to output video.')

# Release everything if the job is finished
video.release()
out.release()
logging.debug(f'Video processing completed. Output saved to {output_path}')