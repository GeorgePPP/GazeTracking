import math
import numpy as np
import cv2
from .pupil import Pupil

class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]  # MediaPipe landmark indices for left eye
    RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]  # MediaPipe landmark indices for right eye

    def __init__(self, frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self.blinking = None

        self._analyze(frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (tuple): First point (x, y)
            p2 (tuple): Second point (x, y)
        """
        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (list): Facial landmarks from MediaPipe
            points (list): Points of an eye (from the MediaPipe landmarks)
        """
        region = np.array([(int(landmarks[point].x * frame.shape[1]), 
                            int(landmarks[point].y * frame.shape[0])) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        if self.frame is not None and self.frame.size > 0:
            height, width = self.frame.shape[:2]
            self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.

        Arguments:
            landmarks (list): Facial landmarks from MediaPipe
            points (list): Points of an eye (from the MediaPipe landmarks)

        Returns:
            The computed ratio
        """
        left = (landmarks[points[0]].x, landmarks[points[0]].y)
        right = (landmarks[points[3]].x, landmarks[points[3]].y)
        top = self._middle_point((landmarks[points[1]].x, landmarks[points[1]].y),
                                 (landmarks[points[2]].x, landmarks[points[2]].y))
        bottom = self._middle_point((landmarks[points[4]].x, landmarks[points[4]].y),
                                    (landmarks[points[5]].x, landmarks[points[5]].y))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            frame (numpy.ndarray): Frame passed by the user
            landmarks (list): Facial landmarks from MediaPipe
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(frame, landmarks, points)

        if self.frame is not None and self.frame.size > 0:
            if not calibration.is_complete():
                calibration.evaluate(self.frame, side)

            threshold = calibration.threshold(side)
            self.pupil = Pupil(self.frame, threshold)