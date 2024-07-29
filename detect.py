import cv2
import os 
import json
import logging
from gaze_tracking import GazeTracking

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_video(input_path, output_path, json_output_path):
    # Initialize GazeTracking
    gaze = GazeTracking()

    # Load the video file
    video = cv2.VideoCapture(input_path)

    # Check if the video file opened successfully
    if not video.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return

    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video properties: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        logger.error(f"Error opening video writer: {output_path}")
        return

    frame_count = 0
    saccade_count = 0
    fixation_count = 0
    gaze_events = []
    last_event = None

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # If the frame was not read correctly, break the loop
        if not ret:
            break

        # Analyze the frame with GazeTracking
        face_detected = gaze.refresh(frame)
        frame = gaze.annotated_frame()

        # Only process gaze events if a face was detected
        if face_detected:
            # Detect saccade and fixation
            is_saccade = gaze.detect_saccade()
            is_fixation = gaze.detect_fixation()

            # Record events
            current_time = frame_count / fps
            if is_saccade and last_event != 'saccade':
                gaze_events.append(('saccade', current_time))
                saccade_count += 1
                last_event = 'saccade'
            elif is_fixation and last_event != 'fixation':
                gaze_events.append(('fixation', current_time))
                fixation_count += 1
                last_event = 'fixation'

            # Add gaze information to the frame
            if gaze.is_blinking():
                cv2.putText(frame, "Blinking", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            cv2.putText(frame, f"Left pupil:  {left_pupil}", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, f"Right pupil: {right_pupil}", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        # Write the annotated frame to the output video
        out.write(frame)

        # Log progress
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        logger.info(f"Processed {frame_count}/{total_frames} frames ({progress:.2f}%)")

    # Release resources
    video.release()
    out.release()

    # Write gaze events to JSON file
    with open(json_output_path, 'w') as f:
        json.dump(gaze_events, f)

    logger.info(f"Video processing completed. Output saved to {output_path}")
    logger.info(f"Gaze events saved to {json_output_path}")
    logger.info(f"Total saccades: {saccade_count}")
    logger.info(f"Total fixations: {fixation_count}")

def create_file_if_not_exist(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
     
def main():
    input_dir = 'data'
    output_dir = 'results/videos'
    json_output_dir = 'results/gaze_events'

    create_file_if_not_exist(input_dir)
    create_file_if_not_exist(output_dir)
    create_file_if_not_exist(json_output_dir)

    for file_name in os.listdir(input_dir):
        vid_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        json_output_path = os.path.join(json_output_dir, file_name)
        logger.info(f"Starting video processing for {file_name}")
        process_video(vid_path, output_path, json_output_path)
        logger.info(f"Video processing for {file_name} finished")

if __name__ == "__main__":
    main()