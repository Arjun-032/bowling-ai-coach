"""
biomechanics.py
===============
Core cricket bowling biomechanics engine.

Responsibilities
----------------
1. Handedness detection   — which arm is the bowling arm?
2. Angle computation      — elbow, shoulder, back, trunk tilt per frame
3. DataFrame construction — tidy time-series with Savitzky-Golay smoothing
4. Release-point detection — heuristic: minimum wrist-Y in delivery half
5. Phase segmentation     — Run-up / Gather / Delivery / Release / Follow-through
6. Coaching feedback      — structured tips keyed to measured angles
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Optional

# Minimum landmark visibility to trust a reading (0-1 scale)
MIN_VISIBILITY = 0.30

LandmarkMap = dict[str, dict[str, float]]


class BiomechanicsAnalyzer:
    """
    Stateful analyzer that wraps a sequence of per-frame LandmarkMaps
    and produces a rich analysis DataFrame plus event metadata.
    """

    def __init__(self) -> None:
        self.bowling_side: str = "RIGHT"   # updated by detect_handedness()

    # ==================================================================
    # 1. Geometry helpers (static — no side-effects)
    # ==================================================================

    @staticmethod
    def _angle(a: tuple, b: tuple, c: tuple) -> float:
        """
        Angle at vertex b, in degrees, for the path a–b–c.
        Uses the dot-product formula; clipped to avoid float artefacts.
        """
        va = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        vc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
        na, nc = np.linalg.norm(va), np.linalg.norm(vc)
        if na < 1e-6 or nc < 1e-6:
            return 180.0
        cos_val = np.dot(va, vc) / (na * nc)
        return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

    @staticmethod
    def _xy(lm: LandmarkMap, name: str) -> tuple[float, float]:
        return (lm[name]["x"], lm[name]["y"])

    @staticmethod
    def _min_vis(lm: LandmarkMap, *names: str) -> float:
        return min(lm[n]["visibility"] for n in names)

    # ==================================================================
    # 2. Handedness detection
    # ==================================================================

    def detect_handedness(self, landmark_seq: list[Optional[LandmarkMap]]) -> str:
        """
        Infer the bowling arm by comparing cumulative wrist displacements.
        """
        right_dist = 0.0
        left_dist  = 0.0

        for i in range(1, len(landmark_seq)):
            curr, prev = landmark_seq[i], landmark_seq[i - 1]
            if curr is None or prev is None:
                continue

            for side in ("RIGHT", "LEFT"):
                key = f"{side}_WRIST"
                dx = curr[key]["x"] - prev[key]["x"]
                dy = curr[key]["y"] - prev[key]["y"]
                dist = float(np.hypot(dx, dy))
                if side == "RIGHT":
                    right_dist += dist
                else:
                    left_dist += dist

        self.bowling_side = "RIGHT" if right_dist >= left_dist else "LEFT"
        return self.bowling_side

    # ==================================================================
    # 3. Per-frame angle extraction
    # ==================================================================

    def compute_angles(self, lms: Optional[LandmarkMap]) -> Optional[dict]:
        """
        Compute bowling-relevant joint angles from a single frame.
        """
        if lms is None:
            return None

        s = self.bowling_side

        # Require minimum visibility on the 5 key joints
        required = [
            f"{s}_SHOULDER", f"{s}_ELBOW", f"{s}_WRIST",
            f"{s}_HIP",      f"{s}_KNEE",
        ]
        try:
            if self._min_vis(lms, *required) < MIN_VISIBILITY:
                return None

            shoulder = self._xy(lms, f"{s}_SHOULDER")
            elbow    = self._xy(lms, f"{s}_ELBOW")
            wrist    = self._xy(lms, f"{s}_WRIST")
            hip      = self._xy(lms, f"{s}_HIP")
            knee     = self._xy(lms, f"{s}_KNEE")
            l_sh     = self._xy(lms, "LEFT_SHOULDER")
            r_sh     = self._xy(lms, "RIGHT_SHOULDER")
        except KeyError:
            return None

        elbow_angle    = self._angle(shoulder, elbow, wrist)
        shoulder_angle = self._angle(hip, shoulder, elbow)
        back_angle     = self._angle(shoulder, hip, knee)

        # Lateral trunk tilt: angle of the shoulder line from horizontal
        dx = r_sh[0] - l_sh[0]
        dy = r_sh[1] - l_sh[1]
        trunk_tilt = float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-6)))

        return {
            "elbow_angle":    elbow_angle,
            "shoulder_angle": shoulder_angle,
            "back_angle":     back_angle,
            "trunk_tilt":     trunk_tilt,
            "wrist_x":        wrist[0],
            "wrist_y":        wrist[1],
        }

    # ==================================================================
    # 4. DataFrame construction & smoothing
    # ==================================================================

    def build_dataframe(
        self,
        frame_indices: list[int],
        angles_list: list[Optional[dict]],
        fps: float = 30.0,
    ) -> pd.DataFrame:
        """
        Merge per-frame angle dicts into a tidy DataFrame with Savitzky-Golay smoothing.
        """
        rows = [
            {"frame": fi, **a}
            for fi, a in zip(frame_indices, angles_list)
            if a is not None
        ]

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).reset_index(drop=True)
        df["time_s"] = df["frame"] / fps

        smooth_targets = ["elbow_angle", "shoulder_angle", "back_angle"]
        n = len(df)

        win = min(15, n)
        if win % 2 == 0:
            win = max(win - 1, 1)

        for col in smooth_targets:
            if col not in df.columns:
                continue
            series = df[col].ffill().bfill().to_numpy(dtype=float)
            if win >= 3 and n > win:
                smoothed = savgol_filter(series, win, polyorder=2)
            else:
                smoothed = series.copy()
            df[f"{col}_smooth"] = smoothed

        return df

    # ==================================================================
    # 5. Release-point detection
    # ==================================================================

    def detect_release_point(self, df: pd.DataFrame) -> int:
        """
        Detect the ball-release frame index (minimum y within delivery half).
        """
        if df.empty or "wrist_y" not in df.columns:
            return len(df) // 2

        search_start = int(len(df) * 0.45)
        segment      = df["wrist_y"].values[search_start:]

        if len(segment) == 0:
            return len(df) // 2

        local_idx = int(np.argmin(segment))
        return search_start + local_idx

    # ==================================================================
    # 6. Phase segmentation (heuristic)
    # ==================================================================

    def detect_phases(self, df: pd.DataFrame) -> dict[str, tuple[int, int]]:
        """
        Segment the bowling action into 5 canonical phases.
        """
        if df.empty or len(df) < 8:
            return {}

        n = len(df)
        r = self.detect_release_point(df)

        return {
            "Run-up":          (0,               int(n * 0.22)),
            "Gather":          (int(n * 0.22),   int(n * 0.44)),
            "Delivery Stride": (int(n * 0.44),   max(int(n * 0.44) + 1, r)),
            "Release":         (max(0, r - 1),   min(n - 1, r + 2)),
            "Follow-through":  (min(n - 1, r + 2), n - 1),
        }

    # ==================================================================
    # 7. AI coaching feedback
    # ==================================================================

    def generate_feedback(
        self,
        df: pd.DataFrame,
        release_idx: int,
    ) -> list[dict]:
        """
        Generate structured coaching tips based on angles at the release point.
        """
        if df.empty or release_idx >= len(df):
            return []

        tips: list[dict] = []
        row = df.iloc[release_idx]

        e_col  = "elbow_angle_smooth"    if "elbow_angle_smooth"    in df.columns else "elbow_angle"
        sh_col = "shoulder_angle_smooth" if "shoulder_angle_smooth" in df.columns else "shoulder_angle"
        bk_col = "back_angle_smooth"     if "back_angle_smooth"     in df.columns else "back_angle"

        e  = float(row.get(e_col,  160.0))
        sh = float(row.get(sh_col, 140.0))
        bk = float(row.get(bk_col, 160.0))

        # -- Elbow --
        if e < 145:
            tips.append({
                "severity_label": "HIGH", "category": "Elbow Extension",
                "message": (
                    f"Arm is heavily bent at release ({e:.0f} deg). "
                    "Work on pushing the arm through fully -- imagine trying to "
                    "reach past the batsman at the point of delivery."
                ),
                "severity": "high",
            })
        elif e < 158:
            tips.append({
                "severity_label": "MEDIUM", "category": "Elbow Extension",
                "message": (
                    f"Moderate elbow bend at release ({e:.0f} deg). "
                    "Drills focusing on a high release point and forearm rotation "
                    "should help open the arm further."
                ),
                "severity": "medium",
            })
        else:
            tips.append({
                "severity_label": "OK", "category": "Elbow Extension",
                "message": f"Good arm extension at release ({e:.0f} deg). Keep it up.",
                "severity": "low",
            })

        # -- Shoulder --
        if sh < 120:
            tips.append({
                "severity_label": "HIGH", "category": "Shoulder Rotation",
                "message": (
                    f"Shoulder not fully rotated at release ({sh:.0f} deg). "
                    "Hip-shoulder separation drills will dramatically increase "
                    "pace and consistency."
                ),
                "severity": "high",
            })
        elif sh < 140:
            tips.append({
                "severity_label": "MEDIUM", "category": "Shoulder Rotation",
                "message": (
                    f"Moderate shoulder rotation ({sh:.0f} deg). "
                    "Focus on leading with the non-bowling shoulder and "
                    "delaying arm rotation to build elastic energy."
                ),
                "severity": "medium",
            })
        else:
            tips.append({
                "severity_label": "OK", "category": "Shoulder Rotation",
                "message": f"Strong shoulder rotation at release ({sh:.0f} deg).",
                "severity": "low",
            })

        # -- Back / Trunk --
        if bk < 145:
            tips.append({
                "severity_label": "CRITICAL", "category": "Trunk Extension / Back Arch",
                "message": (
                    f"Extreme trunk extension detected ({bk:.0f} deg). "
                    "This significantly increases stress injury risk. "
                    "Consult a physiotherapist and prioritise core strengthening."
                ),
                "severity": "critical",
            })
        elif bk < 158:
            tips.append({
                "severity_label": "MEDIUM", "category": "Trunk Extension / Back Arch",
                "message": (
                    f"Moderate back arch at release ({bk:.0f} deg). "
                    "Monitor bowling load carefully and ensure adequate hip "
                    "drive to reduce lumbar stress."
                ),
                "severity": "medium",
            })
        else:
            tips.append({
                "severity_label": "OK", "category": "Trunk Extension / Back Arch",
                "message": f"Good upright trunk position ({bk:.0f} deg). Low injury risk.",
                "severity": "low",
            })

        return tips