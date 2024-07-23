from __future__ import division
import os
import cv2
import dlib
import numpy as np
from collections import deque
from .eye import Eye
from .calibration import Calibration

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

        self.gaze_points = deque(maxlen=30)  # Store last 30 gaze points
        self.fixation_threshold = 0.05  # Threshold for fixation detection
        self.saccade_threshold = 0.1  # Threshold for saccade detection
        self.min_fixation_duration = 5  # Minimum number of frames for a fixation

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

            if self.pupils_located:
                gaze_point = self.horizontal_ratio(), self.vertical_ratio()
                self.gaze_points.append(gaze_point)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def detect_saccade(self):
        """Detects if a saccade occurred in the last frame"""
        if len(self.gaze_points) < 2:
            return False
        
        current_point = self.gaze_points[-1]
        previous_point = self.gaze_points[-2]
        
        distance = np.sqrt((current_point[0] - previous_point[0])**2 + 
                           (current_point[1] - previous_point[1])**2)
        
        return distance > self.saccade_threshold

    def detect_fixation(self):
        """Detects if the gaze is currently in a fixation"""
        if len(self.gaze_points) < self.min_fixation_duration:
            return False
        
        recent_points = list(self.gaze_points)[-self.min_fixation_duration:]
        center = np.mean(recent_points, axis=0)
        
        for point in recent_points:
            distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
            if distance > self.fixation_threshold:
                return False
        
        return True

    def get_fixation_center(self):
        """Returns the center of the current fixation"""
        if self.detect_fixation():
            recent_points = list(self.gaze_points)[-self.min_fixation_duration:]
            return np.mean(recent_points, axis=0)
        return None

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

            h, w = frame.shape[:2]
            gaze_ratio = self.horizontal_ratio(), self.vertical_ratio()
            x = int(gaze_ratio[0] * w)
            y = int(gaze_ratio[1] * h)
            
            cv2.circle(frame, (x, y), 10, (0, 255, 255), 2)
            
            if self.detect_fixation():
                cv2.putText(frame, "Fixation", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if self.detect_saccade():
                cv2.putText(frame, "Saccade", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame