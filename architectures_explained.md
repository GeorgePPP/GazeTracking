# Eye Tracking System Documentation

## Overview
This repository contains a comprehensive eye tracking system implemented in Python. The system uses computer vision techniques to detect and track a user's gaze in real-time, capable of identifying key eye movements such as saccades and fixations.

## Architecture

The system is composed of five main components:

1. `GazeTracking`: The main class that orchestrates the entire eye tracking process
2. `Eye`: Handles the isolation and analysis of individual eyes
3. `Pupil`: Detects and processes iris/pupil information
4. `Calibration`: Manages the calibration process for optimal pupil detection
5. Video Processing Script: Processes video files and generates analysis outputs

Let's dive into each component:

### 1. GazeTracking Class

The `GazeTracking` class is the primary interface for the eye tracking system. It utilizes MediaPipe's Face Mesh for facial landmark detection and coordinates the analysis of both eyes.

Key features:
- Face detection and landmark extraction
- Coordination of eye analysis
- Gaze direction calculation
- Saccade and fixation detection

```python
class GazeTracking(object):
    def __init__(self):
        self.frame = None
        self.eye_left = None    
        self.eye_right = None
        self.calibration = Calibration()

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8)

        # Gaze tracking parameters
        self.gaze_points = deque(maxlen=30)
        self.fixation_threshold = 0.05
        self.saccade_threshold = 0.05
        self.min_fixation_duration = 10
```

The class provides methods for:
- Analyzing frames: `_analyze()`
- Calculating gaze ratios: `horizontal_ratio()`, `vertical_ratio()`
- Detecting eye movements: `detect_saccade()`, `detect_fixation()`
- Generating annotated frames: `annotated_frame()`

### 2. Eye Class

The `Eye` class handles the isolation and analysis of individual eyes using facial landmarks.

Key features:
- Eye region isolation
- Blinking detection
- Pupil detection coordination

```python
class Eye(object):
    LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]

    def __init__(self, frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self._analyze(frame, landmarks, side, calibration)
```

The class uses predefined landmark indices to isolate the eye region:
```python
def _isolate(self, frame, landmarks, points):
    region = np.array([(int(landmarks[point].x * frame.shape[1]), 
                        int(landmarks[point].y * frame.shape[0])) 
                       for point in points])
    # ... isolation process
```

### 3. Pupil Class

The `Pupil` class handles the detection and processing of the iris/pupil within an isolated eye frame.

Key features:
- Image processing for iris isolation
- Pupil position estimation

```python
class Pupil(object):
    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None
        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]
        return new_frame
```

### 4. Calibration Class

The `Calibration` class manages the calibration process to find optimal binarization thresholds for pupil detection.

```python
class Calibration(object):
    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    @staticmethod
    def find_best_threshold(eye_frame):
        average_iris_size = 0.48
        trials = {}
        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)
        best_threshold, _ = min(trials.items(), 
                               key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold
```

### 5. Video Processing Script

The script (`detect.py`) processes video files and generates various outputs:
- Annotated video with gaze visualization
- JSON file containing gaze events
- CSV file with saccade frame information

Key features:
- Batch processing of multiple videos
- Progress logging
- Multiple output formats

```python
def process_video(input_path, output_path, json_output_path, csv_output_path):
    gaze = GazeTracking()
    video = cv2.VideoCapture(input_path)
    
    # ... video processing loop
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        face_detected = gaze.refresh(frame)
        annotated_frame = gaze.annotated_frame()
        
        if face_detected:
            is_saccade = gaze.detect_saccade()
            is_fixation = gaze.detect_fixation()
            # ... event recording
```

## Usage

To process a video or directory of videos:

```bash
python detect.py <data_path> <output_vid_dir> <output_ann_dir> <output_csv_dir>
```

Where:
- `data_path`: Path to video file or directory containing videos
- `output_vid_dir`: Directory for processed videos
- `output_ann_dir`: Directory for JSON annotation files
- `output_csv_dir`: Directory for CSV files

## Dependencies

- OpenCV (cv2)
- MediaPipe
- NumPy

## Limitations and Considerations

1. Lighting conditions can affect pupil detection accuracy
2. Face must be clearly visible and within frame
3. Processing speed depends on hardware capabilities
4. Calibration quality affects overall accuracy

## Future Improvements

1. Implement real-time processing capabilities
2. Add support for multiple face tracking
3. Improve robustness to varying lighting conditions
4. Add more sophisticated eye movement analysis