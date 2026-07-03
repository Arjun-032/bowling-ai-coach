"""
pose_estimator.py
=================
Thin, stateful wrapper around MediaPipe Pose Landmarker (Tasks API).

Responsibilities
----------------
* Load the .task model file once at construction time.
* Run per-frame inference and return landmarks in pixel coordinates.
* Draw a clean skeleton overlay for visualisation.

MediaPipe landmark indices used
--------------------------------
NOSE=0, L_SHOULDER=11, R_SHOULDER=12,
L_ELBOW=13, R_ELBOW=14, L_WRIST=15, R_WRIST=16,
L_HIP=23,  R_HIP=24,   L_KNEE=25,  R_KNEE=26,
L_ANKLE=27, R_ANKLE=28
"""
from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from typing import Optional


# ── Landmark registry ─────────────────────────────────────────────────────────

LANDMARK_IDX: dict[str, int] = {
    "NOSE":           0,
    "LEFT_SHOULDER":  11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW":     13,
    "RIGHT_ELBOW":    14,
    "LEFT_WRIST":     15,
    "RIGHT_WRIST":    16,
    "LEFT_HIP":       23,
    "RIGHT_HIP":      24,
    "LEFT_KNEE":      25,
    "RIGHT_KNEE":     26,
    "LEFT_ANKLE":     27,
    "RIGHT_ANKLE":    28,
}

# Pairs to draw as bones
SKELETON_EDGES: list[tuple[str, str]] = [
    ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER",  "LEFT_ELBOW"),
    ("LEFT_ELBOW",     "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW",    "RIGHT_WRIST"),
    ("LEFT_SHOULDER",  "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP",       "RIGHT_HIP"),
    ("LEFT_HIP",       "LEFT_KNEE"),
    ("RIGHT_HIP",      "RIGHT_KNEE"),
    ("LEFT_KNEE",      "LEFT_ANKLE"),
    ("RIGHT_KNEE",     "RIGHT_ANKLE"),
]

LandmarkMap = dict[str, dict[str, float]]  # name → {x, y, z, visibility}


class PoseEstimator:
    """
    Wraps MediaPipe Pose Landmarker for per-frame inference.

    Parameters
    ----------
    model_path : str
        Path to pose_landmarker_lite.task (or full/heavy variant).
    min_pose_confidence : float
        Detection confidence threshold (0-1).
    """

    MODEL_DOWNLOAD_CMD = (
        "wget https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task -O models/pose_landmarker_lite.task"
    )

    def __init__(
        self,
        model_path: str,
        min_pose_confidence: float = 0.45,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at: {model_path}\n"
                f"Download it with:\n  {self.MODEL_DOWNLOAD_CMD}"
            )

        base_opts = python.BaseOptions(model_asset_path=str(path))
        opts = vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_pose_confidence,
            min_pose_presence_confidence=min_pose_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame_rgb: np.ndarray) -> Optional[LandmarkMap]:
        """
        Run pose detection on a single RGB frame.

        Returns a dict mapping landmark name → {"x", "y", "z", "visibility"}
        with x/y in pixel coordinates, or None if no pose is detected.
        """
        h, w = frame_rgb.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_img)

        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks[0]
        return {
            name: {
                "x":          float(lms[idx].x * w),
                "y":          float(lms[idx].y * h),
                "z":          float(lms[idx].z),
                "visibility": float(lms[idx].visibility),
            }
            for name, idx in LANDMARK_IDX.items()
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw_skeleton(
        self,
        frame_rgb: np.ndarray,
        landmarks: Optional[LandmarkMap],
    ) -> np.ndarray:
        """
        Return a copy of frame_rgb with a colour-coded skeleton overlay.

        Right-side joints → warm red;  Left-side joints → cool blue.
        """
        out = frame_rgb.copy()
        if landmarks is None:
            return out

        # Draw bones
        for a, b in SKELETON_EDGES:
            if a in landmarks and b in landmarks:
                p1 = (int(landmarks[a]["x"]), int(landmarks[a]["y"]))
                p2 = (int(landmarks[b]["x"]), int(landmarks[b]["y"]))
                cv2.line(out, p1, p2, (0, 230, 90), 2, cv2.LINE_AA)

        # Draw joints
        for name, lm in landmarks.items():
            pt = (int(lm["x"]), int(lm["y"]))
            color = (255, 80, 60) if name.startswith("RIGHT") else (60, 120, 255)
            cv2.circle(out, pt, 6, color, -1, cv2.LINE_AA)
            cv2.circle(out, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)  # white rim

        return out

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release MediaPipe internal resources."""
        self._landmarker.close()
