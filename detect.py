import cv2
from gaze_tracking import GazeTracking

# Initialize GazeTracking
gaze = GazeTracking()

# Load the video file
video_path = 'P204_PostMS_Trimmed.mp4' 
video = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not video.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

# Get video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
output_path = 'P204_PostMS_Trimmed_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print(f"Error opening video writer: {output_path}")
    exit()

while True:
    # Read a frame from the video
    ret, frame = video.read()

    # If the frame was not read correctly, break the loop
    if not ret:
        break

    # Analyze the frame with GazeTracking
    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    # Add gaze information to the frame (optional)
    if gaze.is_blinking():
        cv2.putText(frame, "Blinking", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, f"Left pupil:  {left_pupil}", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, f"Right pupil: {right_pupil}", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Write the annotated frame to the output video
    out.write(frame)

# Release resources
video.release()
out.release()
print(f"Video processing completed. Output saved to {output_path}")