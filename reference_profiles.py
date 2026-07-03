"""
reference_profiles.py
=====================
Synthetic biomechanics reference curves used for DTW comparison when
the user has not uploaded a second (pro) video.

Scientific basis
----------------
The "elite" profile approximates the joint-angle trajectories of a
fast-bowling archetype modelled on published cricket biomechanics
research (Portus et al. 2004; Elliott et al. 1993; Ferdinands et al. 2009):

  - Elbow remains near-straight (~158-165 deg) with minimal extension,
    consistent with a legal action and optimal pace transmission.
  - Shoulder angle rises rapidly from ~82 deg in the run-up to ~155 deg at
    release, driven by strong hip-shoulder separation.
  - Trunk (back) shows characteristic hyper-extension during the gather
    phase (~145-150 deg), recovering to ~160 deg at release -- typical of
    side-on fast bowling technique.

The "demo user" profile is deliberately imperfect: a legal but
borderline elbow extension (~14 deg) and slightly lower shoulder rotation,
so every feature of the app is visible without a real video.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# -- Private helpers --

def _smooth(arr: np.ndarray, window: int = 7) -> np.ndarray:
    """Simple symmetric moving-average smoother."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# -- Public API --

def get_reference_profile(n_frames: int = 90) -> pd.DataFrame:
    """
    Generate a reproducible elite fast-bowling reference profile.

    Parameters
    ----------
    n_frames : int
        Number of data points (analogous to analysed frames).

    Returns
    -------
    pd.DataFrame
        Columns: frame, elbow_angle_smooth, shoulder_angle_smooth,
                 back_angle_smooth.
        Schema matches the user analysis DataFrame so DTWComparator can
        compare them directly.
    """
    rng = _rng(42)
    t   = np.linspace(0, 1, n_frames)

    # -- Elbow: near-straight, tiny dip at gather, slight pre-release rise --
    elbow = (
        161.0
        + 4.0  * np.sin(np.pi * t)                         # gentle arc
        - 6.0  * np.exp(-((t - 0.46) ** 2) / 0.012)        # gather dip
        + 3.5  * t                                           # upward trend
        + rng.normal(0, 0.8, n_frames)
    )
    elbow = np.clip(elbow, 148.0, 178.0)

    # -- Shoulder: exponential rise toward full rotation at release --
    shoulder = (
        82.0
        + 76.0 * (1.0 - np.exp(-4.2 * t))
        - 14.0 * np.exp(-((t - 0.86) ** 2) / 0.014)        # slight peak-then-fall
        + rng.normal(0, 1.2, n_frames)
    )
    shoulder = np.clip(shoulder, 75.0, 172.0)

    # -- Back: hyper-extension at gather, recovery at release --
    back = (
        163.0
        - 19.0 * np.exp(-((t - 0.44) ** 2) / 0.013)        # hyper-extension
        + 9.0  * np.maximum(t - 0.44, 0.0)                  # recovery
        + rng.normal(0, 1.0, n_frames)
    )
    back = np.clip(back, 132.0, 178.0)

    return pd.DataFrame({
        "frame":                 np.arange(n_frames),
        "elbow_angle_smooth":    _smooth(elbow),
        "shoulder_angle_smooth": _smooth(shoulder),
        "back_angle_smooth":     _smooth(back),
    })


def get_demo_user_profile(n_frames: int = 80) -> pd.DataFrame:
    """
    Synthetic user profile for Demo Mode.

    Designed to trigger all app features:
      - Detectable release point (wrist dip at ~65 % of clip)
      - Borderline elbow extension (~14° delta) for coaching feedback
      - Moderate DTW similarity to the reference (~68–72 %)
      - Meaningful coaching feedback for all three angles
    """
    rng = _rng(7)
    t   = np.linspace(0, 1, n_frames)

    # Slightly more pronounced elbow dip → borderline legal
    elbow = (
        154.0
        - 12.0 * np.exp(-((t - 0.52) ** 2) / 0.018)
        + 4.5  * t
        + rng.normal(0, 1.0, n_frames)
    )
    elbow = np.clip(elbow, 138.0, 172.0)

    # Lower shoulder rotation → coaching tip fires
    shoulder = (
        80.0
        + 63.0 * (1.0 - np.exp(-3.4 * t))
        + rng.normal(0, 1.4, n_frames)
    )
    shoulder = np.clip(shoulder, 72.0, 162.0)

    # Moderate back arch
    back = (
        160.0
        - 21.0 * np.exp(-((t - 0.47) ** 2) / 0.017)
        + 7.0  * np.maximum(t - 0.44, 0.0)
        + rng.normal(0, 1.1, n_frames)
    )
    back = np.clip(back, 130.0, 175.0)

    # Wrist Y — dips (rises in image) to simulate arm coming overhead at ~65 %
    wrist_y = 410.0 - 130.0 * np.exp(-((t - 0.65) ** 2) / 0.009)

    # Approximate time axis (assume 2.5-second clip at 30 fps)
    time_s = np.linspace(0.0, 2.5, n_frames)

    return pd.DataFrame({
        "frame":                 np.arange(n_frames),
        "time_s":                time_s,
        "elbow_angle":           elbow,
        "shoulder_angle":        shoulder,
        "back_angle":            back,
        "wrist_y":               wrist_y,
        "elbow_angle_smooth":    _smooth(elbow),
        "shoulder_angle_smooth": _smooth(shoulder),
        "back_angle_smooth":     _smooth(back),
    })
