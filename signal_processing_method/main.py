import cv2
import mediapipe as mp
import numpy as np
import csv
import argparse
from saccade_detection import SaccadeDetector, load_config
from visualization import generate_visualizations

def convert_frame_to_time(frame, fps):
    return frame / fps

def calculate_relative_position(x, y, x_ref, y_ref):
    return np.sqrt((x - x_ref)**2 + (y - y_ref)**2)

def process_video(video_path, output_csv, output_video, config):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frames = []
    pupil_coords = []
    
    frame_count = 0
    x_ref, y_ref = None, None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_pupil = landmarks[468]
            right_pupil = landmarks[473]
            
            x = (left_pupil.x * width + right_pupil.x * width) / 2
            y = (left_pupil.y * height + right_pupil.y * height) / 2
            
            pupil_coords.append((x, y))
            
            if x_ref is None:
                x_ref, y_ref = x, y
        else:
            pupil_coords.append((np.nan, np.nan))
        
        frames.append(frame_count)
        frame_count += 1
    
    cap.release()
    frames = np.array(frames)
    pupil_coords = np.array(pupil_coords)
    
    valid_frames = ~np.isnan(pupil_coords).any(axis=1)
    t = convert_frame_to_time(frames[valid_frames], fps)
    x = pupil_coords[valid_frames, 0]
    y = pupil_coords[valid_frames, 1]
    
    if len(x) == 0:
        print("No valid frames with eye detection found.")
        return
    
    x_ref, y_ref = x[0], y[0]
    position = calculate_relative_position(x, y, x_ref, y_ref)
    
    saccade_detector = SaccadeDetector(config)
    labeled_saccades, denoised_position = saccade_detector.detect_saccades(position, fps)
    
    generate_visualizations(t, position, denoised_position, labeled_saccades, 
                            config['saccade_onset_velocity'], config)
    
    output_data = []
    for frame in range(len(frames)):
        if frame in frames[valid_frames]:
            idx = np.where(frames[valid_frames] == frame)[0][0]
            if labeled_saccades[idx] > 0:
                event = 'saccade'
            else:
                event = 'fixation'
        else:
            event = 'no eyes detected'
        output_data.append({'frame': frame, 'gaze_event': event})
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'gaze_event']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_data:
            writer.writerow(row)
    
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index in frames[valid_frames]:
            idx = np.where(frames[valid_frames] == frame_index)[0][0]
            x, y = int(pupil_coords[frame_index][0]), int(pupil_coords[frame_index][1])
            
            if labeled_saccades[idx] > 0:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red for saccade
                cv2.putText(frame, 'Saccade', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green for fixation
                cv2.putText(frame, 'Fixation', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No eyes detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)
        frame_index += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Saccade detection from video")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--input_video", required=True, help="Path to input video file")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV file")
    parser.add_argument("--output_video", required=True, help="Path to output annotated video file")
    args = parser.parse_args()

    config = load_config(args.config)
    process_video(args.input_video, args.output_csv, args.output_video, config)

if __name__ == "__main__":
    main()