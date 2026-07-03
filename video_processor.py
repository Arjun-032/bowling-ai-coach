"""
video_processor.py
==================
Handles all video I/O: frame extraction, metadata, and annotated video export.

Design goal: pure I/O; no ML or biomechanics logic lives here.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Generator, Optional


class VideoProcessor:
    """
    Manages video I/O operations for the bowling analysis pipeline.

    Parameters
    ----------
    video_path : str
        Path to the input video (MP4, AVI, MOV, MKV).
    skip_frames : int
        Process every (skip_frames + 1)-th frame.
        0 = every frame; 2 = every 3rd frame (default, good balance).
    """

    def __init__(self, video_path: str, skip_frames: int = 2) -> None:
        self.video_path = str(video_path)
        self.skip_frames = max(0, skip_frames)
        self._info: Optional[dict] = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_video_info(self) -> dict:
        """
        Return a metadata dict with fps, frame count, resolution, duration.
        Result is cached after the first call.
        """
        if self._info is not None:
            return self._info

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self._info = {
            "fps": fps,
            "total_frames": total,
            "width": w,
            "height": h,
            "duration_s": total / fps,
        }
        return self._info

    # ------------------------------------------------------------------
    # Frame Extraction
    # ------------------------------------------------------------------

    def extract_frames(
        self,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Generator yielding (original_frame_index, RGB_frame) tuples.

        The original_frame_index allows downstream code to compute
        accurate timestamps via  time_s = frame_idx / fps.

        Parameters
        ----------
        progress_cb : callable, optional
            Called with a float in [0, 1] after each yielded frame.
            Useful for driving Streamlit progress bars.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        total = self.get_video_info()["total_frames"]
        idx = 0

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if idx % (self.skip_frames + 1) == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    yield idx, frame_rgb

                    if progress_cb:
                        progress_cb(min(idx / max(total, 1), 1.0))

                idx += 1
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Annotated-video export
    # ------------------------------------------------------------------

    @staticmethod
    def save_video(
        frames_bgr: list[np.ndarray],
        output_path: str,
        fps: float = 25.0,
    ) -> None:
        """
        Write a list of BGR frames to an MP4 file.

        Parameters
        ----------
        frames_bgr : list of np.ndarray
            Frames in BGR format (OpenCV standard).
        output_path : str
            Destination file path (will be created/overwritten).
        fps : float
            Output frame rate.
        """
        if not frames_bgr:
            return

        h, w = frames_bgr[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames_bgr:
            writer.write(frame)

        writer.release()

    # ------------------------------------------------------------------
    # Frame annotation helper (text overlay)
    # ------------------------------------------------------------------

    @staticmethod
    def annotate_frame(
        frame_rgb: np.ndarray,
        angles: Optional[dict],
        phase: str = "",
        is_release: bool = False,
    ) -> np.ndarray:
        """
        Burn angle values and phase label onto a copy of frame_rgb.
        Returns the annotated frame still in RGB.
        """
        out = frame_rgb.copy()
        h, w = out.shape[:2]

        # Semi-transparent overlay panel (top-left)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (220, 95), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        if angles:
            y = 20
            for label, key, color in [
                ("Elbow",    "elbow_angle",    (0, 200, 255)),
                ("Shoulder", "shoulder_angle", (255, 215, 0)),
                ("Back",     "back_angle",     (255, 100, 100)),
            ]:
                val = angles.get(key)
                if val is not None:
                    cv2.putText(
                        out,
                        f"{label}: {val:.1f}\u00b0",
                        (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                    y += 24

        if phase:
            cv2.putText(
                out,
                phase.upper(),
                (8, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        if is_release:
            cv2.putText(
                out,
                "★ RELEASE",
                (w // 2 - 60, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 180),
                2,
                cv2.LINE_AA,
            )

        return out
