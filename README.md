# Cricket Bowling AI Coach

> full-stack sports analytics system combining
> deep-learning pose estimation, signal processing, Dynamic Time Warping, and an
> interactive Streamlit dashboard.

---

## What it does

Upload a side-on MP4 of any cricket bowler and receive:

| Feature | Detail |
|---|---|
| **Joint angle tracking** | Elbow, shoulder, and back angles extracted every frame via MediaPipe BlazePose |
| **Savitzky-Golay smoothing** | Removes landmark jitter while preserving biomechanically meaningful peaks |
| **Automatic release detection** | Heuristic detection of the ball-release frame (wrist-Y minima in delivery half) |
| **DTW pro comparison** | Dynamic Time Warping compares your angle curves to an elite-archetype reference; returns per-angle similarity scores (0–100 %) |
| **Coaching feedback** | Rule-based tips keyed to measured angles at release: elbow extension, shoulder rotation, trunk arch |
| **Handedness detection** | Automatically identifies the bowling arm from cumulative wrist displacement |
| **Annotated video export** | Downloadable MP4 with skeleton overlay, angle labels, phase tags, and release-point marker |

---

## Architecture

```
┌─ Streamlit UI (app.py) ───────────────────────────────────────────── ┐
│                                                                      │
│  Upload / Demo Mode                                                  │
│       │                                                              │
│       ▼                                                              │
│  VideoProcessor ──► frame generator (skip_frames configurable)       │
│       │                                                              │
│       ▼                                                              │
│  PoseEstimator ───► MediaPipe BlazePose (pose_landmarker_lite.task)  │
│       │              33 keypoints, pixel coordinates, visibility     │
│       ▼                                                              │
│  BiomechanicsAnalyzer                                                │
│    ├─ detect_handedness()   → cumulative wrist-displacement          │
│    ├─ compute_angles()      → elbow / shoulder / back per frame      │
│    ├─ build_dataframe()     → Pandas TS + Savitzky-Golay smooth      │
│    ├─ detect_release_point()→ argmin(wrist_y) in delivery half       │
│    ├─ detect_phases()       → 5-phase segmentation (heuristic)       │
│    └─ generate_feedback()   → structured coaching tips               │
│       │                                                              │
│       ▼                                                              │
│  DTWComparator ──► normalised O(N·M) DTW, weighted 3-angle score     │
│       │                                                              │
│       ▼                                                              │
│  Plotly Interactive Charts                                           │
│    ├─ Angle timeline with phase shading                              │
│    └─ DTW 3-subplot user vs reference comparison                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ML / Signal Processing Techniques

| Technique | Where used | Why |
|---|---|---|
| **Deep pose estimation** (BlazePose) | Per-frame keypoint extraction | Eliminates manual marker placement; works on casual video |
| **Savitzky-Golay filter** | Angle smoothing | Preserves peak shape & biomechanical events unlike moving average |
| **Dynamic Time Warping (DTW)** | Pro-comparison similarity score | Handles timing variation between deliveries fairly — a 3 s runup vs 5 s runup still produces a meaningful comparison |
| **Time-series feature extraction** | Release point, phase boundaries | Converts continuous signal into discrete coaching events |
| **Heuristic event detection** | Handedness, release frame | Simple, interpretable, no labelled training data required |
| **Rule-based expert system** | Coaching tips keyed to release angles | Encodes biomechanics domain knowledge for elbow, shoulder, and trunk guidance |

---

## Project structure

```
cricket_bowling_ai/
├── app.py                    # Streamlit app — orchestration + UI
├── biomechanics.py           # BiomechanicsAnalyzer — angles, events, coaching
├── dtw_comparator.py         # DTWComparator — multivariate DTW
├── pose_estimator.py         # PoseEstimator — MediaPipe Tasks API
├── reference_profiles.py     # Synthetic elite & demo bowling profiles
├── video_processor.py        # VideoProcessor — frame I/O
├── requirements.txt
├── setup.sh                  # One-command model download + pip install
├── .gitignore
├── LICENSE
└── models/
    └── pose_landmarker_lite.task  # Downloaded by setup.sh — NOT committed to git
```

---

## Setup & run

### Prerequisites

- Python >= 3.10
- `wget` or `curl` (for model download)

### Install

```bash
git clone <repo-url>
cd cricket_bowling_ai

# Creates virtual env (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# Download MediaPipe model + install dependencies
bash setup.sh
```

### Manual model download (alternative)

```bash
mkdir -p models
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task \
     -O models/pose_landmarker_lite.task
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage guide

1. **Sidebar → Upload Bowling Video** — MP4, AVI, MOV, MKV supported.
   Side-on (90°) camera angle gives best results.
2. *(Optional)* Upload a **Custom Reference Video** to use your own pro
   footage for DTW comparison instead of the built-in synthetic profile.
3. Choose a **processing speed** (every 3rd frame is the default balance).
4. Click **Analyze Bowling Action**.
5. Explore the three analysis tabs:
   - **Angle Timeline** — smooth angle curves with phase shading
   - **Pro Comparison (DTW)** — similarity score vs elite reference
   - **AI Coaching** — prioritised feedback sorted by severity
6. *(Optional)* Expand **Export Annotated Video** to download the skeleton-overlay MP4.

### Demo Mode

Toggle **Demo Mode** in the sidebar to explore all features instantly
using synthetic bowling data — no video or model file required.
Great for quick walkthroughs in presentations.

---

## Key design decisions

**Why Savitzky-Golay over a moving average?**
SG filtering fits a polynomial through a sliding window, which
means it preserves peak positions and heights — critical for detecting
the genuine maximum elbow angle during delivery without flattening it
into the noise floor.

**Why DTW over Pearson/Euclidean similarity?**
Two bowlers delivering at different pace produce temporally misaligned
angle curves even if their technique is identical.  DTW finds the optimal
time-axis alignment before measuring shape similarity, giving a fair
comparison regardless of delivery tempo.

**Why heuristic release detection instead of a trained classifier?**
Labelled frame-level bowling datasets are not publicly available.  The
wrist-Y minima heuristic is physically motivated (the arm is overhead at
release), transparent, and requires no training data.  A classifier could
replace or augment it given sufficient labelled data.

**Why the synthetic reference profile?**
Pro video footage is copyrighted.  The synthetic profile is derived from
published biomechanics research (Portus et al. 2004; Elliott et al. 1993)
and is fully reproducible (seeded RNG).  Users can always supply their
own reference video.

---

## Extending the project

Possible next steps for a production system:

- **LSTM / 1D-CNN phase classifier** — replace the heuristic phase
  segmentation with a model trained on labelled bowling sequences
- **Real-time webcam mode** — switch MediaPipe to `VIDEO` running mode
  with timestamp-based tracking
- **Injury risk score** — combine back arch + delivery load count into a
  session-level risk model
- **FastDTW** — drop-in replacement for the O(N·M) DTW implementation
  for longer sequences
- **Docker + cloud deploy** — `Dockerfile` already included in roadmap;
  deploy to Hugging Face Spaces (free tier) or Render

---

## References

- Elliott, B. C. et al. (1993). *A three-dimensional cinematographic
  analysis of the fastbowling delivery stride*. Journal of Sports Sciences.
- Berndt, D. J. & Clifford, J. (1994). *Using dynamic time warping to
  find patterns in time series*. KDD Workshop.
- Savitzky, A. & Golay, M. J. E. (1964). *Smoothing and differentiation
  of data by simplified least squares procedures*. Analytical Chemistry.
- MediaPipe Pose Landmarker:
  https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

---

## License

MIT — see `LICENSE` for details.
