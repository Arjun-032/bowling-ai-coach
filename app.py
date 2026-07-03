"""
app.py
======
Cricket Bowling AI Coach — Streamlit web application.

Architecture overview
---------------------
  Upload / Demo → VideoProcessor (frame extraction)
               → PoseEstimator  (MediaPipe BlazePose per frame)
               → BiomechanicsAnalyzer (angles, smoothing, events)
               → DTWComparator  (vs elite reference profile)
               → Streamlit UI   (metrics, 3 analysis tabs)

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PATH OVERRIDE
# Inject the 'src' folder directly into sys.path to bypass package import issues
# ──────────────────────────────────────────────────────────────────────────────
SRC_PATH = str(Path(__file__).parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Imports with 'src.' prefix removed
from video_processor import VideoProcessor
from pose_estimator import PoseEstimator
from biomechanics import BiomechanicsAnalyzer
from dtw_comparator import DTWComparator
from reference_profiles import get_reference_profile, get_demo_user_profile

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# Phase colour palette (Plotly rgba strings)
PHASE_COLORS: dict[str, str] = {
    "Run-up":          "rgba(99, 110, 250, 0.13)",
    "Gather":          "rgba(239, 85,  59, 0.13)",
    "Delivery Stride": "rgba(0,  204, 150, 0.13)",
    "Release":         "rgba(255,161,  90, 0.32)",
    "Follow-through":  "rgba(171, 99, 250, 0.13)",
}

# Angle display config: col_name → (trace_colour, human label)
ANGLE_CFG: dict[str, tuple[str, str]] = {
    "elbow_angle_smooth":    ("#00C4FF", "Elbow Angle"),
    "shoulder_angle_smooth": ("#FFD700", "Shoulder Angle"),
    "back_angle_smooth":     ("#FF6B6B", "Back Angle"),
}

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# ──────────────────────────────────────────────────────────────────────────────
# Page config & global CSS
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cricket Bowling AI Coach",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Tighten the metric value font */
      div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }

      /* Tab bar */
      .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
        padding: 6px 18px;
      }

      /* Grade badge */
      .grade-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Cached resource: PoseEstimator (loaded once per Streamlit session)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_pose_estimator() -> PoseEstimator:
    return PoseEstimator(MODEL_PATH)


def _download_model() -> bool:
    """
    Download the MediaPipe Pose Landmarker model if it is not already present.
    Uses Python's built-in urllib so no external tools (wget/curl) are needed.
    Returns True on success, False on failure.
    """
    import urllib.request
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        return True
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("Downloading pose model (one-time, ~6 MB)..."):
            urllib.request.urlretrieve(MODEL_URL, str(model_path))
        return True
    except Exception as exc:
        st.error(
            f"Failed to download the MediaPipe model automatically: {exc}\n\n"
            "Please run `bash setup.sh` locally and re-deploy, or download manually:\n"
            f"{MODEL_URL}"
        )
        return False


def model_available() -> bool:
    if Path(MODEL_PATH).exists():
        return True
    return _download_model()

# ──────────────────────────────────────────────────────────────────────────────
# Core analysis pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(video_path: str, skip_frames: int) -> dict | None:
    """
    Full analysis pipeline from raw video path to a results dict.
    """
    processor = VideoProcessor(video_path, skip_frames=skip_frames)
    pose_est  = load_pose_estimator()
    analyzer  = BiomechanicsAnalyzer()

    info = processor.get_video_info()

    all_lms:        list = []
    all_frames_rgb: list = []
    frame_indices:  list = []

    # ── Step 1 & 2: Extract frames + detect pose ────────────────────────────
    prog = st.progress(0, text="Extracting pose landmarks...")

    for fi, frame_rgb in processor.extract_frames():
        lms = pose_est.detect(frame_rgb)
        all_lms.append(lms)
        all_frames_rgb.append(frame_rgb)
        frame_indices.append(fi)
        prog.progress(
            min(fi / max(info["total_frames"], 1), 0.95),
            text=f"Frame {fi} / {info['total_frames']}  |  "
                 f"Pose found: {sum(l is not None for l in all_lms)}",
        )

    prog.progress(1.0, text="Extraction complete")
    time.sleep(0.3)
    prog.empty()

    # Sanity check
    n_detected      = sum(1 for l in all_lms if l is not None)
    detection_rate  = n_detected / max(len(all_lms), 1)

    if detection_rate < 0.25:
        st.error(
            f"Warning: Pose detected in only {detection_rate:.0%} of frames. "
            "Ensure the bowler is fully visible, well-lit, and filmed from "
            "a side-on angle for best results."
        )
        return None

    # ── Steps 3–5: Handedness, angles, DataFrame ────────────────────────────
    bowling_side = analyzer.detect_handedness(all_lms)
    angles_list  = [analyzer.compute_angles(lm) for lm in all_lms]
    df           = analyzer.build_dataframe(frame_indices, angles_list, fps=info["fps"])

    if df.empty:
        st.error("No valid angle data could be extracted. Try a different video.")
        return None

    # ── Steps 6–8: Events, DTW, coaching ───────────────────────────────
    release_idx = analyzer.detect_release_point(df)
    phases      = analyzer.detect_phases(df)
    feedback    = analyzer.generate_feedback(df, release_idx)

    ref_df      = get_reference_profile(n_frames=100)
    dtw_results = DTWComparator().compute_similarity(df, ref_df)

    # ── Step 9: Annotated frames ────────────────────────────────────────────
    phase_map: dict[int, str] = {}
    for phase_name, (si, ei) in phases.items():
        for row in range(si, ei + 1):
            if row < len(df):
                phase_map[row] = phase_name

    annotated: list[np.ndarray] = []
    for row_idx, (frame_rgb, lms, raw_angles) in enumerate(
        zip(all_frames_rgb, all_lms, angles_list)
    ):
        skeletonised = pose_est.draw_skeleton(frame_rgb, lms)
        labelled     = VideoProcessor.annotate_frame(
            skeletonised,
            raw_angles,
            phase=phase_map.get(row_idx, ""),
            is_release=(row_idx == release_idx),
        )
        annotated.append(labelled)

    return {
        "df":             df,
        "ref_df":         ref_df,
        "info":           info,
        "bowling_side":   bowling_side,
        "release_idx":    release_idx,
        "phases":         phases,
        "feedback":       feedback,
        "dtw_results":    dtw_results,
        "annotated":      annotated,
        "detection_rate": detection_rate,
        "fps":            info["fps"],
    }

# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _add_phase_shading(fig: go.Figure, df: pd.DataFrame, phases: dict) -> None:
    """Add semi-transparent phase bands and labels to a Plotly figure."""
    for phase_name, (si, ei) in phases.items():
        if si >= len(df) or ei <= si:
            continue
        x0 = float(df["time_s"].iloc[min(si, len(df) - 1)])
        x1 = float(df["time_s"].iloc[min(ei, len(df) - 1)])
        if x1 <= x0:
            continue
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=PHASE_COLORS.get(phase_name, "rgba(128,128,128,0.1)"),
            layer="below",
            line_width=0,
            annotation_text=phase_name,
            annotation_position="top left",
            annotation=dict(font_size=9, font_color="#999"),
        )

def plot_timeline(df: pd.DataFrame, release_idx: int, phases: dict) -> go.Figure:
    """Interactive angle-over-time chart with phase shading & release marker."""
    fig = go.Figure()

    t           = df["time_s"].values
    release_t   = float(df["time_s"].iloc[release_idx])

    for col, (color, label) in ANGLE_CFG.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=t, y=df[col].values,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{label}</b>: %{{y:.1f}}°<br>t=%{{x:.2f}}s<extra></extra>",
            ))

    _add_phase_shading(fig, df, phases)

    fig.add_vline(
        x=release_t,
        line_dash="dot",
        line_color="#FF8C00",
        line_width=2.5,
        annotation_text="Release",
        annotation_position="top right",
        annotation=dict(font_color="#FF8C00", font_size=12),
    )

    fig.update_layout(
        title="Joint Angles Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Angle (°)",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=420,
        margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig

def plot_dtw(user_df: pd.DataFrame, ref_df: pd.DataFrame, dtw_results: dict) -> go.Figure:
    """3-row subplot: user vs reference for elbow, shoulder, back."""
    labels = {
        "elbow_angle_smooth":    "Elbow Angle",
        "shoulder_angle_smooth": "Shoulder Angle",
        "back_angle_smooth":     "Back Angle",
    }
    per_angle = dtw_results.get("per_angle", {})

    subplot_titles = []
    for col, label in labels.items():
        sim = per_angle.get(col, {}).get("similarity", 0)
        subplot_titles.append(f"{label}   (Similarity: {sim:.0f}%)")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.11,
    )

    for row, (col, label) in enumerate(labels.items(), start=1):
        if col not in user_df.columns or col not in ref_df.columns:
            continue

        user_t = np.linspace(0, 1, len(user_df))
        ref_t  = np.linspace(0, 1, len(ref_df))

        show_legend = row == 1

        fig.add_trace(go.Scatter(
            x=user_t, y=user_df[col].values,
            mode="lines", name="Your Action",
            line=dict(color="#00C4FF", width=2.5),
            showlegend=show_legend, legendgroup="user",
            hovertemplate=f"{label}: %{{y:.1f}}°<extra>Your Action</extra>",
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=ref_t, y=ref_df[col].values,
            mode="lines", name="Elite Reference",
            line=dict(color="#FFD700", dash="dash", width=2.0),
            showlegend=show_legend, legendgroup="ref",
            hovertemplate=f"{label}: %{{y:.1f}}°<extra>Reference</extra>",
        ), row=row, col=1)

        fig.update_yaxes(title_text="Degrees (°)", row=row, col=1)
        fig.update_xaxes(title_text="Normalised Time (0→1)", row=row, col=1)

    fig.update_layout(
        title="DTW Comparison: Your Action vs Elite Reference",
        template="plotly_dark",
        height=700,
        margin=dict(l=60, r=30, t=70, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Helper: safe column fetch at a row
# ──────────────────────────────────────────────────────────────────────────────

def _val(row: pd.Series, smooth_col: str, raw_col: str, default: float) -> float:
    if smooth_col in row.index and pd.notna(row[smooth_col]):
        return float(row[smooth_col])
    if raw_col in row.index and pd.notna(row[raw_col]):
        return float(row[raw_col])
    return default

# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style='text-align:center;margin-bottom:0;font-size:2.4rem'>
            Cricket Bowling AI Coach
        </h1>
        <p style='text-align:center;color:#999;margin-top:6px;font-size:1.05rem'>
            Upload a side-on bowling video → get biomechanics analysis
            &amp; pro-level comparison.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Session state init ────────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.session_state.results = None

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        demo_mode = st.toggle(
            "Demo Mode",
            value=False,
            help="Run with synthetic data — no video or model required.",
        )

        # ── Video upload guidance ──────────────────────────────────────
        if not demo_mode:
            with st.expander("Video requirements (read before uploading)", expanded=True):
                st.markdown(
                    """
**Ideal specs for accurate results:**

| Setting | Recommended |
|---|---|
| Duration | 5 – 15 seconds |
| Resolution | 720p (1280 x 720) |
| Format | MP4 (H.264) |
| Max file size | 50 MB |
| Camera angle | Side-on, 90 deg to bowler |
| Frame rate | 30 fps (no slow-motion) |

**Tips before uploading:**
- Trim to just the bowling action — a 10-second clip is ideal.
- Keep the full body in frame throughout the delivery.
- Good lighting and a plain background improve pose detection.
- File too large? Compress free with [HandBrake](https://handbrake.fr).
                    """
                )

        uploaded_file = st.file_uploader(
            "Upload Bowling Video",
            type=["mp4", "avi", "mov", "mkv"],
            disabled=demo_mode,
            help="MP4, 720p, 5-15 sec, under 50 MB. Side-on camera angle required.",
        )

        ref_video = st.file_uploader(
            "Custom Reference Video (optional)",
            type=["mp4", "avi", "mov"],
            disabled=demo_mode,
            help=(
                "Upload a pro bowler's video to compare against instead of the "
                "built-in synthetic reference. Same format rules apply (MP4, 720p, under 50 MB)."
            ),
        )

        process_every = st.select_slider(
            "Processing speed",
            options=["Every frame", "Every 2nd", "Every 3rd (default)", "Every 4th", "Every 5th"],
            value="Every 3rd (default)",
            help="Controls how many frames are analysed. 'Every 3rd' is the recommended balance of speed and accuracy. Use 'Every frame' only for short clips under 10 seconds.",
        )
        skip_map = {
            "Every frame": 0, "Every 2nd": 1, "Every 3rd (default)": 2,
            "Every 4th": 3, "Every 5th": 4,
        }
        skip_frames = skip_map[process_every]

        st.divider()

        with st.expander("How it works"):
            st.markdown(
                """
                **Pipeline (per uploaded video)**
                1. Frame extraction via OpenCV
                2. MediaPipe BlazePose — 33 keypoints per frame
                3. 3 joint angles computed: Elbow · Shoulder · Back
                4. Savitzky-Golay smoothing removes sensor jitter
                5. Release point detected as wrist-Y minimum
                6. Dynamic Time Warping compares your curves to elite
                7. Rule-based coaching tips generated from release angles

                **ML / Signal Processing Techniques**
                - Deep pose estimation (BlazePose CNN)
                - Savitzky-Golay filter (signal processing)
                - Dynamic Time Warping (sequence alignment)
                - Time-series feature extraction
                - Heuristic event detection
                """
            )

    # ── Demo Mode ─────────────────────────────────────────────────────────────
    if demo_mode:
        analyzer = BiomechanicsAnalyzer()
        analyzer.bowling_side = "RIGHT"

        demo_df = get_demo_user_profile(n_frames=80)
        ref_df  = get_reference_profile(n_frames=100)

        release_idx = analyzer.detect_release_point(demo_df)
        phases      = analyzer.detect_phases(demo_df)
        feedback    = analyzer.generate_feedback(demo_df, release_idx)
        dtw_results = DTWComparator().compute_similarity(demo_df, ref_df)

        st.session_state.results = {
            "df":             demo_df,
            "ref_df":         ref_df,
            "info":           {"fps": 30.0, "total_frames": 80, "width": 720, "height": 1280, "duration_s": 2.67},
            "bowling_side":   "Right",
            "release_idx":    release_idx,
            "phases":         phases,
            "feedback":       feedback,
            "dtw_results":    dtw_results,
            "annotated":      [],
            "detection_rate": 1.0,
            "fps":            30.0,
        }
        st.info("**Demo Mode** — showing synthetic bowling data. Toggle off to upload a real video.")

    # ── Real-video upload & analysis ──────────────────────────────────────────
    elif uploaded_file is not None:

        if not model_available():
            st.error(
                "**MediaPipe model not found.** "
                "Run `bash setup.sh` in your terminal to download it, "
                "then restart the app."
            )
            st.code("bash setup.sh", language="bash")
            st.stop()

        if st.button("Analyze Bowling Action", type="primary", use_container_width=True):
            suffix = Path(uploaded_file.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                vid_path = tmp.name

            try:
                results = run_pipeline(vid_path, skip_frames)

                if results and ref_video is not None:
                    ref_suffix = Path(ref_video.name).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ref_suffix) as ref_tmp:
                        ref_tmp.write(ref_video.read())
                        ref_path = ref_tmp.name

                    st.info("Processing reference video...")
                    ref_results = run_pipeline(ref_path, skip_frames)
                    if ref_results and not ref_results["df"].empty:
                        results["ref_df"]     = ref_results["df"]
                        results["dtw_results"] = DTWComparator().compute_similarity(
                            results["df"], ref_results["df"]
                        )
                    try:
                        os.unlink(ref_path)
                    except OSError:
                        pass

                if results:
                    st.session_state.results = results
                    st.rerun()

            finally:
                try:
                    os.unlink(vid_path)
                except OSError:
                    pass

    # ── Landing state ─────────────────────────────────────────────────────────
    else:
        if st.session_state.results is None:
            st.markdown(
                """
                <div style='text-align:center;padding:4rem 2rem;
                            background:#0e1117;border-radius:12px;
                            border:1px dashed #333;'>
                  <h3 style='margin-top:0.5rem'>Ready to analyse</h3>
                  <p style='color:#888'>
                    Upload a bowling video in the sidebar and click
                    <strong>Analyze Bowling Action</strong>,<br>
                    or toggle <strong>Demo Mode</strong> to explore with
                    sample data.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

    # ── Results rendering ─────────────────────────────────────────────────────
    r = st.session_state.results
    if not r:
        return

    df          = r["df"]
    ref_df      = r["ref_df"]
    release_idx = r["release_idx"]
    phases      = r["phases"]
    feedback    = r["feedback"]
    dtw_results = r["dtw_results"]

    rel_row = df.iloc[release_idx]
    e_val   = _val(rel_row, "elbow_angle_smooth",    "elbow_angle",    160.0)
    sh_val  = _val(rel_row, "shoulder_angle_smooth", "shoulder_angle", 140.0)
    bk_val  = _val(rel_row, "back_angle_smooth",     "back_angle",     160.0)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    st.subheader("Key Metrics at Release Point")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Bowling Arm", r["bowling_side"].title())
    with c2:
        st.metric("Elbow Angle", f"{e_val:.1f}°")
    with c3:
        st.metric("Shoulder Angle", f"{sh_val:.1f}°")
    with c4:
        st.metric("Back Angle", f"{bk_val:.1f}°")
    with c5:
        overall = dtw_results.get("overall_score", 0.0)
        grade   = dtw_results.get("grade", "N/A")
        st.metric("DTW Similarity", f"{overall:.0f}%", help=f"Grade: {grade}")

    det_rate = r.get("detection_rate", 1.0)
    if det_rate < 0.75:
        st.warning(
            f"Warning: Pose detected in {det_rate:.0%} of frames. "
            "Results may be less accurate — try a video with better lighting "
            "and a clear, unobstructed view of the bowler."
        )

    st.divider()

    # ── Analysis tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "Angle Timeline",
        "Pro Comparison (DTW)",
        "AI Coaching",
    ])

    # ── Tab 1: Angle Timeline ─────────────────────────────────────────────────
    with tab1:
        st.plotly_chart(plot_timeline(df, release_idx, phases), use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(
                "Joint angles extracted frame-by-frame via **MediaPipe BlazePose** "
                "(deep-learning pose estimator) and smoothed with a "
                "**Savitzky-Golay filter** (window 15, poly-order 2) to eliminate "
                "landmark jitter while preserving peak shape."
            )
        with col_r:
            angle_cols = [c for c in ["elbow_angle", "shoulder_angle", "back_angle"] if c in df.columns]
            if angle_cols:
                stats = df[angle_cols].describe().loc[["mean", "std", "min", "max"]].round(1)
                stats.index = ["Mean", "Std Dev", "Min", "Max"]
                st.dataframe(stats.rename(columns={
                    "elbow_angle": "Elbow (°)",
                    "shoulder_angle": "Shoulder (°)",
                    "back_angle": "Back (°)",
                }), use_container_width=True)

    # ── Tab 2: DTW Comparison ─────────────────────────────────────────────────
    with tab2:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Overall Similarity", f"{overall:.0f}%")
        with m2:
            per = dtw_results.get("per_angle", {})
            e_sim  = per.get("elbow_angle_smooth",    {}).get("similarity", 0)
            sh_sim = per.get("shoulder_angle_smooth",  {}).get("similarity", 0)
            bk_sim = per.get("back_angle_smooth",      {}).get("similarity", 0)
            st.metric("Elbow Similarity",    f"{e_sim:.0f}%")
        with m3:
            st.metric("Shoulder Similarity", f"{sh_sim:.0f}%")
        with m4:
            st.metric("Back Similarity",     f"{bk_sim:.0f}%")

        grade_color = {
            "Elite": "#00FF87", "Strong": "#7CFF6B",
            "Developing": "#FFB800", "Needs Work": "#FF4B4B",
        }.get(grade, "#aaa")
        st.markdown(
            f'<span class="grade-badge" '
            f'style="background:rgba(255,255,255,0.07);color:{grade_color};'
            f'border:1.5px solid {grade_color};">'
            f'{grade}</span>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(plot_dtw(df, ref_df, dtw_results), use_container_width=True)

        st.caption(
            "Your curves (solid blue) are compared to the elite reference (dashed gold) "
            "using **Dynamic Time Warping** — a sequence-alignment technique that warps "
            "the time axis to handle differences in delivery tempo before measuring "
            "shape-level similarity.  Similarity = 1 − (normalised DTW distance)."
        )

    # ── Tab 3: Coaching Feedback ──────────────────────────────────────────────
    with tab3:
        st.subheader("AI Coaching Analysis")

        if not feedback:
            st.info("No coaching data available — ensure the video has adequate pose detection coverage.")
        else:
            sorted_tips = sorted(feedback, key=lambda t: SEVERITY_ORDER.get(t.get("severity", "low"), 3))

            for tip in sorted_tips:
                cat  = tip["category"]
                msg  = tip["message"]
                sev  = tip.get("severity", "low")

                text = f"**{cat}** — {msg}"
                if sev in ("critical", "high"):
                    st.error(text)
                elif sev == "medium":
                    st.warning(text)
                else:
                    st.success(text)

        if phases:
            st.divider()
            st.markdown("### Phase Breakdown")
            rows = []
            for phase_name, (si, ei) in phases.items():
                if si >= len(df) or ei >= len(df):
                    continue
                t0 = float(df["time_s"].iloc[min(si, len(df) - 1)])
                t1 = float(df["time_s"].iloc[min(ei, len(df) - 1)])
                rows.append({
                    "Phase":       phase_name,
                    "Start (s)":   f"{t0:.2f}",
                    "End (s)":     f"{t1:.2f}",
                    "Duration (s)": f"{t1 - t0:.2f}",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Annotated video export (below tabs) ───────────────────────────────────
    if r.get("annotated"):
        st.divider()
        with st.expander("Export Annotated Video", expanded=False):
            st.markdown(
                "Download an MP4 with the skeleton overlay, "
                "angle labels, and release-point marker."
            )
            if st.button("Render Annotated Video"):
                with st.spinner("Rendering..."):
                    frames_bgr = [
                        cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                        for f in r["annotated"]
                    ]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_tmp:
                        VideoProcessor.save_video(
                            frames_bgr,
                            out_tmp.name,
                            fps=r.get("fps", 25.0),
                        )
                        out_path = out_tmp.name

                with open(out_path, "rb") as fh:
                    st.download_button(
                        "Download bowling_analysis.mp4",
                        data=fh,
                        file_name="bowling_analysis.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                try:
                    os.unlink(out_path)
                except OSError:
                    pass

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()