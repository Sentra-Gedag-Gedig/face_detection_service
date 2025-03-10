import cv2
import numpy as np
from pathlib import Path
from core.config import settings
from app.schemas.detection import DetectionResponse, Position


class FaceDetectionService:
    def __init__(self):
        self.clf = cv2.CascadeClassifier(str(
            Path(cv2.__file__).parent.absolute() / "data" /
            settings.cascade_path))
        if self.clf.empty():
            raise ValueError("Failed to load cascade classifier")

    def process(self, frame: np.ndarray) -> DetectionResponse:
        frame_height, frame_width = frame.shape[:2]
        center = {
            "x": frame_width // 2,
            "y": frame_height // 2
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        result = DetectionResponse(
            status="NO_FACE",
            instructions=[],
            face_position=None,
            face_size=None,
            frame_center=Position(**center),
            deviations=None
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            return self._analyze_face(x, y, w, h, frame_width, frame_height,
                                      center)

        return result

    def _analyze_face(self, x, y, w, h, fw, fh, center) -> DetectionResponse:
        face_center = {
            "x": x + w//2,
            "y": y + h//2
        }
        face_ratio = w / fw
        x_dev = (face_center["x"] - center["x"]) / fw
        y_dev = (face_center["y"] - center["y"]) / fh

        instructions = []
        if abs(x_dev) > settings.center_deviation:
            instructions.append("left" if x_dev < 0 else "right")
        if abs(y_dev) > settings.center_deviation:
            instructions.append("down" if y_dev < 0 else "up")
        if face_ratio < settings.min_face_ratio:
            instructions.append("closer")
        elif face_ratio > settings.max_face_ratio:
            instructions.append("back")

        status = "READY" if not instructions else "ADJUST"

        return DetectionResponse(
            status=status,
            instructions=instructions,
            face_position=Position(**face_center),
            face_size=face_ratio,
            frame_center=Position(**center),
            deviations={"x": x_dev, "y": y_dev} if x_dev is not None
            and y_dev is not None else None
        )
