"""
dtw_comparator.py
=================
Dynamic Time Warping (DTW) engine for comparing a user's bowling
action against a reference (professional archetype or uploaded video).

Why DTW over Euclidean / Pearson?
----------------------------------
Bowling actions vary in tempo — a 5-second runup vs a 3-second one.
DTW warps the time axis to find the optimal alignment between two
sequences before measuring their similarity.  This gives a meaningful
comparison even when delivery speeds differ significantly.

Implementation
--------------
Standard O(N·M) DP-based DTW (Bellman, 1957), normalised by optimal
alignment path length so scores are comparable across different sequence
lengths.  For sequences of length ~100 frames this runs in < 50 ms in
pure NumPy; no external library required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class DTWComparator:
    """
    Multivariate DTW similarity scorer for bowling angle curves.

    The three angles (elbow, shoulder, back) are compared independently
    and combined as a weighted average reflecting their relative importance
    for technique assessment.

    Weights (biomechanics rationale)
    ----------------------------------
    elbow    0.45 — primary legality & pace determinant
    shoulder 0.30 — technique accuracy & consistency
    back     0.25 — injury-risk indicator
    """

    ANGLE_COLS = [
        "elbow_angle_smooth",
        "shoulder_angle_smooth",
        "back_angle_smooth",
    ]

    WEIGHTS = {
        "elbow_angle_smooth":    0.45,
        "shoulder_angle_smooth": 0.30,
        "back_angle_smooth":     0.25,
    }

    # ── Label thresholds for the overall similarity score ────────────────

    @staticmethod
    def score_to_grade(score: float) -> str:
        if score >= 88:
            return "Elite"
        elif score >= 74:
            return "Strong"
        elif score >= 58:
            return "Developing"
        else:
            return "Needs Work"

    # ------------------------------------------------------------------
    # Core DTW (pure NumPy — no external deps)
    # ------------------------------------------------------------------

    @staticmethod
    def _dtw_1d(s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Normalised DTW distance between two 1-D float sequences.

        Normalisation by (len(s1) + len(s2)) ensures the score is
        comparable regardless of sequence length.

        Returns
        -------
        float
            Normalised distance ∈ [0, ∞).  For min-max-normalised
            sequences the practical range is [0, 0.50].
            Lower → more similar.
        """
        n, m = len(s1), len(s2)
        # Infinity-bordered cost matrix (size (n+1) × (m+1))
        D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        D[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(float(s1[i - 1]) - float(s2[j - 1]))
                D[i, j] = cost + min(
                    D[i - 1, j],      # insertion
                    D[i, j - 1],      # deletion
                    D[i - 1, j - 1],  # match
                )

        return float(D[n, m]) / (n + m)

    @staticmethod
    def _min_max_normalize(seq: np.ndarray) -> np.ndarray:
        """
        Normalise seq to [0, 1] so that angles on different absolute
        scales can be compared fairly via DTW.
        """
        lo, hi = seq.min(), seq.max()
        if (hi - lo) < 1e-8:
            return np.zeros_like(seq, dtype=float)
        return (seq - lo) / (hi - lo)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_similarity(
        self,
        user_df: pd.DataFrame,
        ref_df: pd.DataFrame,
    ) -> dict:
        """
        Compare user's angle curves to a reference DataFrame using DTW.

        Both DataFrames must contain columns from ANGLE_COLS (or a subset).
        Missing columns are skipped gracefully.

        Returns
        -------
        dict
            overall_score : float  (0–100, higher = more similar)
            grade         : str    ("Elite" | "Strong" | "Developing" | "Needs Work")
            per_angle     : dict   {col: {"dtw_distance": float, "similarity": float}}
        """
        per_angle: dict[str, dict] = {}
        weighted_sum  = 0.0
        total_weight  = 0.0

        for col in self.ANGLE_COLS:
            if col not in user_df.columns or col not in ref_df.columns:
                continue

            u = user_df[col].dropna().to_numpy(dtype=float)
            r = ref_df[col].dropna().to_numpy(dtype=float)

            if len(u) < 3 or len(r) < 3:
                continue

            u_norm = self._min_max_normalize(u)
            r_norm = self._min_max_normalize(r)
            dist   = self._dtw_1d(u_norm, r_norm)

            # Convert distance → similarity score (0-100).
            # For normalised [0,1] sequences of similar length the
            # maximum practical DTW distance is ~0.40–0.45.
            # We use 0.45 as the ceiling so scores feel intuitive.
            MAX_PRACTICAL_DTW = 0.45
            sim = max(0.0, 1.0 - dist / MAX_PRACTICAL_DTW) * 100.0

            w = self.WEIGHTS.get(col, 1.0 / len(self.ANGLE_COLS))
            per_angle[col] = {
                "dtw_distance": round(dist, 4),
                "similarity":   round(sim, 1),
            }
            weighted_sum += w * sim
            total_weight += w

        overall = (
            round(weighted_sum / total_weight, 1)
            if total_weight > 0.0
            else 0.0
        )

        return {
            "overall_score": overall,
            "grade":         self.score_to_grade(overall),
            "per_angle":     per_angle,
        }

    # ------------------------------------------------------------------
    # DTW alignment path (for optional visualisation)
    # ------------------------------------------------------------------

    @staticmethod
    def alignment_path(
        s1: np.ndarray,
        s2: np.ndarray,
    ) -> list[tuple[int, int]]:
        """
        Return the optimal DTW warping path as a list of (i, j) index pairs.
        Useful for drawing the alignment between user and reference curves.
        """
        n, m = len(s1), len(s2)
        D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        D[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(float(s1[i - 1]) - float(s2[j - 1]))
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

        # Traceback from (n, m) → (1, 1)
        path: list[tuple[int, int]] = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            moves = {
                (i - 1, j):     D[i - 1, j],
                (i, j - 1):     D[i, j - 1],
                (i - 1, j - 1): D[i - 1, j - 1],
            }
            i, j = min(moves, key=moves.get)

        path.reverse()
        return path
