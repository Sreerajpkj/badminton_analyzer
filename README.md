# 🏸 Badminton Match Analyzer

A computer-vision-powered badminton analysis dashboard built with
**Python, Streamlit, YOLOv8 Pose, OpenCV and Plotly**.

The app analyses match footage to estimate player movement, court
coverage, stroke activity and rally patterns. It provides an interactive
dashboard, an annotated video and downloadable analysis files.



------------------------------------------------------------------------

## ✨ Features

-   **Video input:** Upload a local video, use a local file path, or
    provide a YouTube URL.
-   **Court calibration:** Select the four court corners to map player
    positions onto a standard badminton court (6.1 m × 13.4 m).
-   **Player tracking and movement:** Estimate player speed, distance
    covered and movement intensity.
-   **Court coverage:** View front/mid/back occupancy, left/right
    distribution and player heatmaps.
-   **Stroke estimation:** Estimate stroke events from racket-arm
    movement; identify smash-like actions heuristically.
-   **Rally analysis:** Estimate rally count, duration, shots per rally,
    longest rally, rest time and work-to-rest ratio.
-   **Coach insights:** Generate rule-based observations about court
    positioning, recovery and movement trends.
-   **Visual dashboard:** Review KPIs, player comparisons, charts, rally
    timelines and annotated footage.
-   **Exports:** Download tracking, rally and stroke CSV files, a JSON
    summary and annotated video.

## 🧰 Tech stack

  Component          Technology
  ------------------ -------------------------
  Dashboard          Streamlit
  Language           Python
  Pose detection     Ultralytics YOLOv8 Pose
  Video processing   OpenCV
  Data processing    NumPy, Pandas
  Charts             Plotly
  Video download     yt-dlp

------------------------------------------------------------------------

## 🏗️ Application architecture

The project is split into a Streamlit user interface and a
computer-vision analysis engine.

``` mermaid
flowchart TD
    A[User] --> B[Streamlit Dashboard - app.py]
    B --> C{Select video source}
    C --> D[Upload local video]
    C --> E[Local video path]
    C --> F[YouTube URL]
    F --> G[yt-dlp download]
    D --> H[Video file]
    E --> H
    G --> H

    H --> I[Preview frame]
    I --> J[User selects 4 court corners]
    J --> K[Perspective transform / court calibration]
    K --> L[Analyzer engine - analyzer.py]

    L --> M[OpenCV frame processing]
    M --> N[YOLOv8 Pose detection]
    N --> O[Player positions and pose landmarks]
    O --> P[Tracking and movement calculations]
    P --> Q[Stroke event estimation]
    Q --> R[Rally segmentation and statistics]

    R --> S[Annotated video]
    R --> T[Player and court metrics]
    R --> U[Rally and stroke datasets]
    T --> V[Streamlit report dashboard]
    U --> V
    S --> V

    V --> W[CSV exports]
    V --> X[JSON summary]
    V --> Y[Annotated video export]
```

### Processing flow

1.  **Load footage** -- The dashboard accepts an uploaded file, a local
    path or a YouTube URL. YouTube videos are downloaded into the
    `downloads/` directory.
2.  **Preview and calibrate** -- A preview frame is displayed. Select
    the court corners in this order: near-left, near-right, far-right,
    far-left.
3.  **Map court coordinates** -- The selected points define a
    perspective transform. Player positions are mapped to a top-down
    court representation using the standard court dimensions.
4.  **Detect players and poses** -- OpenCV reads the video frames and
    YOLOv8 Pose estimates people and body landmarks.
5.  **Calculate movement** -- The engine tracks player positions and
    estimates speed, distance, movement intensity and court occupancy.
6.  **Estimate strokes and rallies** -- Heuristics based on racket-arm
    movement estimate stroke events. Rally segments are inferred from
    the estimated strokes.
7.  **Present and export** -- Streamlit displays the annotated video,
    metrics and charts, and provides downloadable output files.

------------------------------------------------------------------------

## 📁 Project structure

``` text
badminton-match-analyzer/
│
├── app.py                 # Streamlit dashboard and user interaction
├── analyzer.py            # Computer-vision and match-analysis engine
├── requirements.txt       # Python dependencies
├── yolov8n-pose.pt        # YOLOv8 pose weights (downloaded on first run)
│
├── downloads/             # Downloaded source videos (created at runtime)
├── outputs/               # Annotated videos and analysis exports (if configured)
│
└── README.md
```

The `downloads/` and `outputs/` directories may be created by the
application when it runs. Do not commit large video files or model
weights unless you intentionally want to distribute them.

------------------------------------------------------------------------

## 🚀 Getting started

### 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/<your-repository>.git
cd badminton_analyzer
```

Replace the repository URL and folder name with your actual GitHub
repository details.

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the dashboard

``` bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

``` text
http://localhost:8501
```

Open that URL in your browser.



------------------------------------------------------------------------

## 🕹️ How to use the app

### Step 1 --- Load a match video

Choose one of the supported input methods in the dashboard:

-   Upload a video from your computer.
-   Enter a path to a video available on the machine running Streamlit.
-   Paste a YouTube video URL.

For best results, use a continuous match recording with a fixed camera
and the entire court visible.

### Step 2 --- Calibrate the court

Pause on a frame where the court boundaries are clearly visible. Select
the four corners in this order:

1.  Near-left
2.  Near-right
3.  Far-right
4.  Far-left

Check the top-down calibration preview. If the corners are selected
incorrectly, redo the calibration before starting the analysis.

### Step 3 --- Analyse the footage

Start the analysis and monitor the annotated video and live metrics.
Depending on the configuration, the dashboard can show player
detections, movement trails, speed estimates, a mini-court and rally
information.

Use the sidebar's shot and rally tuning controls if the estimated stroke
or rally counts need adjustment.

------------------------------------------------------------------------

## 📊 Metrics and outputs

  -----------------------------------------------------------------------
  Area                                Metrics / outputs
  ----------------------------------- -----------------------------------
  Players                             Current, top and average speed;
                                      distance covered; distance per
                                      minute; time standing, walking,
                                      moving and sprinting

  Rallies                             Rally count; duration; shots per
                                      rally; longest rally; rest time;
                                      work-to-rest ratio; per-rally
                                      player movement

 






