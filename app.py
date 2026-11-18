"""
Badminton Match Analyzer - Streamlit app (styled dashboard)
Run:  streamlit run app.py
"""
import html
import json
import os
import re
import time

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import analyzer as az

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:  # optional dependency -> manual entry fallback
    streamlit_image_coordinates = None

DL_DIR = "downloads"
OUT_DIR = "outputs"
DISP_W = 900
PALETTE = ["#ff8c00", "#1fb6ff", "#e84393", "#2ecc71"]
ACCENT = "#22c55e"

st.set_page_config(page_title="Badminton Match Analyzer", page_icon="🏸", layout="wide")

# --------------------------------------------------------------------------------------
# Styling + graphics
# --------------------------------------------------------------------------------------
CSS = """
<style>
.block-container{padding-top:1.4rem;max-width:1300px}
.hero{position:relative;overflow:hidden;border-radius:22px;padding:30px 34px;margin-bottom:22px;color:#fff;
  background:linear-gradient(120deg,#052e16 0%,#065f46 45%,#0e7490 100%);box-shadow:0 10px 30px rgba(0,0,0,.25)}
.hero h1{margin:0;font-size:2.15rem;font-weight:800;letter-spacing:-.5px;color:#fff}
.hero p{margin:.4rem 0 0;opacity:.88;font-size:1.02rem;max-width:620px;color:#fff}
.hero .chips{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap}
.hero .chip{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.28);padding:4px 12px;
  border-radius:999px;font-size:.82rem;color:#fff}
.hero svg.court{position:absolute;right:-10px;top:-18px;height:150%;opacity:.28}
.hero svg.shuttle{position:absolute;right:150px;top:16px;width:110px;transform:rotate(28deg);
  filter:drop-shadow(0 6px 10px rgba(0,0,0,.35))}
.step{display:flex;align-items:center;gap:14px;margin:30px 0 12px}
.step .num{width:38px;height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;
  font-weight:800;color:#fff;background:linear-gradient(135deg,#22c55e,#0ea5e9);box-shadow:0 4px 12px rgba(14,165,233,.35)}
.step h2{margin:0;padding:0;font-size:1.45rem;line-height:1.2}
.step p{margin:0;opacity:.65;font-size:.9rem}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin:6px 0 16px}
.kpi{display:flex;gap:12px;align-items:center;padding:14px 16px;border-radius:16px;
  border:1px solid rgba(128,128,128,.22);border-left:5px solid var(--c);
  background:linear-gradient(135deg,color-mix(in srgb,var(--c) 16%,transparent),rgba(128,128,128,.05));
  transition:transform .15s ease,box-shadow .15s ease}
.kpi:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(0,0,0,.18)}
.kpi-ic{font-size:1.7rem;width:46px;height:46px;border-radius:14px;display:flex;align-items:center;justify-content:center;
  background:color-mix(in srgb,var(--c) 22%,transparent)}
.kpi-l{font-size:.74rem;text-transform:uppercase;letter-spacing:.7px;opacity:.68;font-weight:600}
.kpi-v{font-size:1.65rem;font-weight:800;line-height:1.15}
.kpi-s{font-size:.78rem;opacity:.65}
.pcard{border-radius:18px;padding:16px 18px;margin-bottom:12px;border:1px solid rgba(128,128,128,.22);
  background:linear-gradient(160deg,color-mix(in srgb,var(--c) 18%,transparent),rgba(128,128,128,.04));
  border-top:5px solid var(--c)}
.pcard .pc-h{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.15rem;margin-bottom:10px}
.pcard .dot{width:12px;height:12px;border-radius:50%;background:var(--c);box-shadow:0 0 0 4px color-mix(in srgb,var(--c) 30%,transparent)}
.pcard .badge{margin-left:auto;font-size:.7rem;font-weight:700;padding:3px 10px;border-radius:999px;
  background:color-mix(in srgb,var(--c) 28%,transparent);text-transform:uppercase;letter-spacing:.6px}
.pcard .pc-g{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center}
.pcard .pc-g b{display:block;font-size:1.25rem}
.pcard .pc-g span{font-size:.7rem;opacity:.65;text-transform:uppercase;letter-spacing:.5px}
.ins{display:flex;gap:14px;padding:14px 16px;margin-bottom:10px;border-radius:14px;border:1px solid rgba(128,128,128,.2);
  border-left:5px solid var(--c);background:linear-gradient(90deg,color-mix(in srgb,var(--c) 12%,transparent),transparent)}
.ins .ic{font-size:1.5rem}
.ins .tx{line-height:1.45}
.pill{display:inline-block;padding:3px 12px;border-radius:999px;font-size:.78rem;font-weight:700;
  background:rgba(34,197,94,.18);color:#22c55e;border:1px solid rgba(34,197,94,.35)}
div[data-testid="stTabs"] button[role="tab"]{font-weight:600;font-size:.98rem}
section[data-testid="stSidebar"] .logo{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.15rem;margin-bottom:6px}
</style>
"""

SHUTTLE_SVG = """
<svg class="shuttle" viewBox="0 0 100 140" xmlns="http://www.w3.org/2000/svg">
 <defs><linearGradient id="fg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#d1fae5"/></linearGradient></defs>
 <path d="M50 78 L8 6 Q50 -6 92 6 Z" fill="url(#fg)" stroke="#a7f3d0" stroke-width="2"/>
 <g stroke="#6ee7b7" stroke-width="1.6" fill="none">
  <path d="M50 78 L22 4"/><path d="M50 78 L36 0"/><path d="M50 78 L50 -1"/><path d="M50 78 L64 0"/><path d="M50 78 L78 4"/>
  <path d="M14 18 Q50 8 86 18"/><path d="M26 44 Q50 36 74 44"/></g>
 <ellipse cx="50" cy="96" rx="17" ry="20" fill="#f59e0b"/><ellipse cx="50" cy="90" rx="13" ry="10" fill="#fbbf24"/>
</svg>"""

COURT_SVG = """
<svg class="court" viewBox="0 0 200 440" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="#fff" stroke-width="3">
 <rect x="10" y="10" width="180" height="420"/><line x1="25" y1="10" x2="25" y2="430"/><line x1="175" y1="10" x2="175" y2="430"/>
 <line x1="10" y1="220" x2="190" y2="220" stroke-width="6"/><line x1="10" y1="155" x2="190" y2="155"/><line x1="10" y1="285" x2="190" y2="285"/>
 <line x1="100" y1="10" x2="100" y2="155"/><line x1="100" y1="285" x2="100" y2="430"/>
 <line x1="10" y1="34" x2="190" y2="34"/><line x1="10" y1="406" x2="190" y2="406"/></svg>"""

LOGO_SVG = """<svg width="30" height="30" viewBox="0 0 100 140" xmlns="http://www.w3.org/2000/svg">
 <path d="M50 78 L8 6 Q50 -6 92 6 Z" fill="#e5e7eb" stroke="#22c55e" stroke-width="5"/>
 <ellipse cx="50" cy="98" rx="19" ry="22" fill="#f59e0b"/></svg>"""

st.markdown(CSS, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def hero():
    st.markdown(
        f'<div class="hero">{COURT_SVG}{SHUTTLE_SVG}<h1>🏸 Badminton Match Analyzer</h1>'
        '<p>Paste a YouTube link, calibrate the court, and get live player speeds, rallies, strokes, '
        'heatmaps and coach-ready insights.</p><div class="chips"><span class="chip">⚡ Live speed tracking</span>'
        '<span class="chip">🔁 Rally detection</span><span class="chip">🔥 Court heatmaps</span>'
        '<span class="chip">🧠 Coach insights</span></div></div>', unsafe_allow_html=True)


def step(num, title, sub=""):
    st.markdown(f'<div class="step"><div class="num">{num}</div><div><h2>{esc(title)}</h2>'
                f'<p>{esc(sub)}</p></div></div>', unsafe_allow_html=True)


def kpi(label, value, sub="", icon="", color=ACCENT):
    s = f'<div class="kpi-s">{esc(sub)}</div>' if sub else ""
    return (f'<div class="kpi" style="--c:{color}"><div class="kpi-ic">{icon}</div><div>'
            f'<div class="kpi-l">{esc(label)}</div><div class="kpi-v">{esc(value)}</div>{s}</div></div>')


def kpi_row(items, target=None):
    (target or st).markdown('<div class="kpi-grid">' + "".join(items) + "</div>", unsafe_allow_html=True)


def hex_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def style(fig, h=360, title=None):
    fig.update_layout(
        height=h, title=dict(text=title, x=0.01, font=dict(size=16)) if title else None,
        margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.14, x=0), hoverlabel=dict(font_size=13))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.2)")
    return fig


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def show_image(target, rgb):
    try:
        target.image(rgb, width="stretch")
    except TypeError:  # older Streamlit
        target.image(rgb, use_container_width=True)


@st.cache_data(show_spinner=False)
def get_frame(path, t):
    return az.read_frame(path, t)


@st.cache_resource(show_spinner="Loading pose model (first run downloads weights)…")
def load_detector(model, imgsz):
    return az.PoseDetector(model, conf=0.3, imgsz=imgsz)


def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def add_court(fig, height=560):
    W, L = az.COURT_W, az.COURT_L
    lines = [(0, 0, W, 0), (W, 0, W, L), (W, L, 0, L), (0, L, 0, 0),
             (0.46, 0, 0.46, L), (W - 0.46, 0, W - 0.46, L),
             (0, L / 2 - 1.98, W, L / 2 - 1.98), (0, L / 2 + 1.98, W, L / 2 + 1.98),
             (0, 0.76, W, 0.76), (0, L - 0.76, W, L - 0.76),
             (W / 2, 0, W / 2, L / 2 - 1.98), (W / 2, L / 2 + 1.98, W / 2, L)]
    for x0, y0, x1, y1 in lines:
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=-0.3, y0=L / 2, x1=W + 0.3, y1=L / 2, line=dict(color="#222", width=5))
    fig.update_xaxes(range=[-0.8, W + 0.8], visible=False)
    fig.update_yaxes(range=[-1.2, L + 1.2], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(plot_bgcolor="#2e7d4f", paper_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=10, r=10, t=45, b=10))
    return fig


def heat_fig(df, title, color):
    W, L = az.COURT_W, az.COURT_L
    bx = np.arange(-0.6, W + 0.6 + 1e-6, 0.3)
    by = np.arange(-1.0, L + 1.0 + 1e-6, 0.3)
    hh, _, _ = np.histogram2d(df.x, df.y, bins=[bx, by])
    hh = cv2.GaussianBlur(hh.T.astype(np.float32), (0, 0), 1.2)
    if hh.sum() > 0:
        hh = hh / hh.sum() * 100
        hh[hh < hh.max() * 0.03] = np.nan
    scale = [[0, hex_rgba("#fef08a", 0.55)], [0.5, hex_rgba(color, 0.8)], [1, "#7f1d1d"]]
    fig = go.Figure(go.Heatmap(z=hh, x=(bx[:-1] + bx[1:]) / 2, y=(by[:-1] + by[1:]) / 2,
                               colorscale=scale, colorbar=dict(title="% time"),
                               hovertemplate="%{z:.2f}% of time<extra></extra>"))
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16)))
    return add_court(fig)


def pcolor(labels):
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}


def radar_fig(s, labels, colors):
    cats = ["Distance", "Top speed", "Strokes", "Smash share", "Net presence", "Sprint share", "Positioning"]
    raw = {}
    for lab in labels:
        if lab not in s.index:
            continue
        r = s.loc[lab]
        rec = r.recovery_dist_m
        raw[lab] = [r.distance_m, r.peak_speed_ms, r.shots, r.smash_pct, r.front_pct, r.sprint_pct,
                    0.0 if rec != rec else max(0.0, 1 - rec / 3)]
    mx = [max(v[i] for v in raw.values()) or 1 for i in range(len(cats))]
    fig = go.Figure()
    for lab, v in raw.items():
        vals = [100 * x / m for x, m in zip(v, mx)]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=lab, fill="toself",
                                      line=dict(color=colors[lab], width=3), fillcolor=hex_rgba(colors[lab], 0.28)))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 100],
                                                                          gridcolor="rgba(128,128,128,.3)")))
    return style(fig, 400, "Player profile (relative to best)")


def gauge(value, title, color, vmax, suffix=" m/s"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value) if value == value else 0, number=dict(suffix=suffix, valueformat=".1f"),
        title=dict(text=title, font=dict(size=14)),
        gauge=dict(axis=dict(range=[0, vmax]), bar=dict(color=color, thickness=0.28), bgcolor="rgba(128,128,128,.12)",
                   borderwidth=0, steps=[dict(range=[0, vmax * 0.5], color="rgba(128,128,128,.08)"),
                                         dict(range=[vmax * 0.5, vmax * 0.8], color="rgba(128,128,128,.16)"),
                                         dict(range=[vmax * 0.8, vmax], color="rgba(128,128,128,.26)")])))
    fig.update_layout(height=210, margin=dict(l=20, r=20, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def donut(values, names, colors, title, h=300, center=""):
    fig = go.Figure(go.Pie(values=values, labels=names, hole=0.62, marker=dict(colors=colors), sort=False,
                           textinfo="percent", hovertemplate="%{label}: %{value:.1f}%<extra></extra>"))
    if center:
        fig.add_annotation(text=center, showarrow=False, font=dict(size=15))
    fig.update_layout(title=dict(text=title, x=0.02, font=dict(size=15)), height=h, showlegend=True,
                      margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.05))
    return fig


def player_card(lab, r, color):
    rec = "–" if r.recovery_dist_m != r.recovery_dist_m else f"{r.recovery_dist_m:.1f}m"
    return (f'<div class="pcard" style="--c:{color}"><div class="pc-h"><span class="dot"></span>{esc(lab)}'
            f'<span class="badge">{esc(r.side)} side</span></div><div class="pc-g">'
            f'<div><b>{r.distance_m:.0f} m</b><span>distance</span></div>'
            f'<div><b>{r.peak_speed_ms:.1f}</b><span>top m/s</span></div>'
            f'<div><b>{int(r.shots)}</b><span>strokes</span></div>'
            f'<div><b>{rec}</b><span>recovery</span></div></div></div>')


def insight_icon(text):
    t = text.lower()
    table = [("fatigue", "🥵", "#ef4444"), ("recovery", "⏱️", "#f59e0b"), ("smash", "💥", "#e84393"),
             ("sprint", "⚡", "#eab308"), ("back court", "↩️", "#f97316"), ("net", "🕸️", "#22c55e"),
             ("work : rest", "⚖️", "#0ea5e9"), ("rally profile", "🔁", "#8b5cf6"), ("movement load", "🏃", "#06b6d4"),
             ("half", "↔️", "#a855f7")]
    for k, ic, c in table:
        if k in t:
            return ic, c
    return "💡", "#22c55e"


def fmt_or_dash(v, f="{:.0f}", suf=""):
    return "–" if v is None or v != v else f.format(v) + suf


# --------------------------------------------------------------------------------------
# Sidebar - settings
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div class="logo">{LOGO_SVG} Match Analyzer</div>', unsafe_allow_html=True)
    st.header("⚙️ Settings")
    mode = st.radio("Match type", ["singles", "doubles"], horizontal=True)
    a1 = a2 = b1 = b2 = ""
    if mode == "singles":
        n1 = st.text_input("Near-side player (closest to camera)", "Player A")
        n2 = st.text_input("Far-side player", "Player B")
    else:
        n1 = st.text_input("Team A name (near side, closest to camera)", "Team A")
        c1_, c2_ = st.columns(2)
        a1 = c1_.text_input("A · player 1 (left)", "Player A1")
        a2 = c2_.text_input("A · player 2 (right)", "Player A2")
        n2 = st.text_input("Team B name (far side)", "Team B")
        c3_, c4_ = st.columns(2)
        b1 = c3_.text_input("B · player 1 (left)", "Player B1")
        b2 = c4_.text_input("B · player 2 (right)", "Player B2")
        st.caption("Charts show Team A and Team B only. Player names appear on the annotated video by court "
                   "position (left / right as seen from the camera) and may swap if partners rotate.")
    st.caption("Stats are tracked per *side of the court*. If players switch ends, analyse each game "
               "separately using the clip range below.")
    st.subheader("Clip range (seconds)")
    c1, c2 = st.columns(2)
    clip_start = c1.number_input("Start", 0.0, value=0.0, step=5.0)
    clip_end = c2.number_input("End (0 = to the end)", 0.0, value=0.0, step=5.0)
    st.subheader("Speed vs accuracy")
    sample_fps = st.slider("Analysed frames per second", 5, 25, 12,
                           help="Higher = more accurate shots & speed, slower processing")
    model = st.selectbox("Pose model", ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt"],
                         help="n = fastest, m = most accurate (use a GPU for m)")
    imgsz = st.select_slider("Detection resolution", [640, 960, 1280], value=960,
                             help="Far-side players are small - higher helps")
    save_video = st.checkbox("Save annotated video", True)
    with st.expander("🎯 Shot & rally tuning"):
        hit_thr = st.slider("Stroke threshold (wrist speed)", 4, 40, 12,
                            help="Lower = more strokes detected (and more false ones)")
        smash_thr = st.slider("Smash-like threshold", 15, 80, 35)
        rally_gap = st.slider("Max gap inside a rally (s)", 1.5, 6.0, 3.0, 0.5)
        min_shots = st.slider("Min shots per rally", 2, 8, 3)
    st.divider()
    st.caption("⚠️ Estimates from video - best with a fixed, wide view of the whole court.")

hero()

# --------------------------------------------------------------------------------------
# Step 1 - load video
# --------------------------------------------------------------------------------------
step(1, "Load video", "YouTube link, uploaded file or a file on this computer")
src = st.radio("Source", ["YouTube link", "Upload file", "Local file path"], horizontal=True,
               label_visibility="collapsed")


def register_video(path, meta=None):
    path = az.ensure_readable(path)
    st.session_state.video_path = path
    st.session_state.video_info = az.video_info(path)
    st.session_state.video_meta = meta or {}
    st.session_state.corners = []
    st.session_state.result = None
    st.session_state.video_out = None
    st.session_state.calib_nonce = ss("calib_nonce", 0) + 1
    st.session_state.last_click = None


if src == "YouTube link":
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…")
    res = st.selectbox("Max download resolution", [480, 720, 1080], index=1,
                       help="720p is a good balance for analysis")
    if st.button("⬇️ Download & load", type="primary", disabled=not url):
        bar = st.progress(0.0, text="Downloading…")
        try:
            path, meta = az.download_youtube(url, DL_DIR, res,
                                             lambda p: bar.progress(p, text=f"Downloading… {p * 100:.0f}%"))
            register_video(path, meta)
            bar.empty()
            st.success(f"Downloaded: {meta.get('title', os.path.basename(path))}")
        except Exception as e:
            bar.empty()
            st.error(f"Download failed: {e}")
elif src == "Upload file":
    up = st.file_uploader("Video file", type=["mp4", "mov", "mkv", "avi", "webm"])
    if up is not None and st.button("Load uploaded video", type="primary"):
        os.makedirs(DL_DIR, exist_ok=True)
        p = os.path.join(DL_DIR, up.name)
        with open(p, "wb") as f:
            f.write(up.getbuffer())
        register_video(p, {"title": up.name})
else:
    lp = st.text_input("Path to video file on this computer")
    if st.button("Load file", type="primary", disabled=not lp):
        if os.path.exists(lp):
            register_video(lp, {"title": os.path.basename(lp)})
        else:
            st.error("File not found.")

vpath = st.session_state.get("video_path")
if not vpath:
    st.info("Load a video to continue.")
    st.stop()

vinfo = st.session_state.video_info
kpi_row([kpi("Video", st.session_state.video_meta.get("title", os.path.basename(vpath))[:28], icon="🎞️", color="#8b5cf6"),
         kpi("Resolution", f"{vinfo['width']}×{vinfo['height']}", icon="🖥️", color="#0ea5e9"),
         kpi("Frame rate", f"{vinfo['fps']:.0f} fps", icon="⏩", color="#f59e0b"),
         kpi("Length", az.fmt_t(vinfo["duration"]), icon="⏱️", color="#22c55e")])

# --------------------------------------------------------------------------------------
# Step 2 - court calibration
# --------------------------------------------------------------------------------------
step(2, "Calibrate the court", "Click 4 corners: ① near-left → ② near-right → ③ far-right → ④ far-left (outer doubles lines)")

corners = ss("corners", [])
t_prev = st.slider("Preview frame at (seconds)", 0.0, float(max(vinfo["duration"] - 1, 1)),
                   float(min(vinfo["duration"] * 0.1, 30)), step=1.0)
frame = get_frame(vpath, t_prev)
if frame is None:
    st.error("Could not read that frame.")
    st.stop()
fh, fw = frame.shape[:2]
scale = fw / DISP_W
disp = cv2.resize(frame, (DISP_W, int(fh / scale)))
for i, (cx, cy) in enumerate(corners):
    p = (int(cx / scale), int(cy / scale))
    cv2.circle(disp, p, 7, (0, 0, 255), -1)
    cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
if len(corners) >= 2:
    pts = np.int32([(c[0] / scale, c[1] / scale) for c in corners]).reshape(-1, 1, 2)
    cv2.polylines(disp, [pts], len(corners) == 4, (0, 255, 0), 2)
disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

cc1, cc2 = st.columns([3, 1])
with cc1:
    if streamlit_image_coordinates is not None and len(corners) < 4:
        click = streamlit_image_coordinates(Image.fromarray(disp_rgb), width=DISP_W,
                                            key=f"calib_{ss('calib_nonce', 0)}_{len(corners)}")
        if click and click != st.session_state.get("last_click"):
            st.session_state.last_click = click
            st.session_state.corners.append((click["x"] * scale, click["y"] * scale))
            st.rerun()
    else:
        show_image(st, disp_rgb)
with cc2:
    done = len(corners) == 4
    st.markdown(f'<span class="pill">{"✅ Calibrated" if done else f"Points {len(corners)}/4"}</span>',
                unsafe_allow_html=True)
    st.write("")
    if st.button("↩️ Undo last point", disabled=not corners):
        st.session_state.corners.pop()
        st.session_state.last_click = None
        st.session_state.calib_nonce += 1
        st.rerun()
    if st.button("🗑️ Reset points", disabled=not corners):
        st.session_state.corners = []
        st.session_state.last_click = None
        st.session_state.calib_nonce += 1
        st.rerun()
    if done:
        H = az.compute_homography(corners)
        top = cv2.warpPerspective(frame, np.diag([50, 50, 1]) @ H,
                                  (int(az.COURT_W * 50), int(az.COURT_L * 50)))
        top = cv2.flip(top, 0)
        st.caption("Top-down check (should look like a straight court):")
        show_image(st, cv2.cvtColor(top, cv2.COLOR_BGR2RGB))

with st.expander("Manual corner entry (pixel coordinates)"):
    st.caption(f"Original frame size is {fw}×{fh}. One `x,y` per line, in the order ① ② ③ ④.")
    txt = st.text_area("Corners", "\n".join(f"{c[0]:.0f},{c[1]:.0f}" for c in corners), height=100)
    if st.button("Apply manual corners"):
        try:
            pts = [tuple(float(v) for v in line.split(",")) for line in txt.strip().splitlines() if line.strip()]
            assert len(pts) == 4 and all(len(p) == 2 for p in pts)
            st.session_state.corners = pts
            st.session_state.calib_nonce += 1
            st.rerun()
        except Exception:
            st.error("Enter exactly 4 lines of `x,y`.")

if len(corners) != 4:
    st.info("Select all 4 court corners to continue.")
    st.stop()

# --------------------------------------------------------------------------------------
# Step 3 - run analysis with live view
# --------------------------------------------------------------------------------------
step(3, "Analyse", "Live annotated video, speeds and rally counter while the match is processed")
end_val = clip_end if clip_end > clip_start else None
if st.button("▶️ Start analysis", type="primary"):
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = az.Config(mode=mode, sample_fps=sample_fps, hit_thr=hit_thr, smash_thr=smash_thr,
                    rally_gap=rally_gap, min_shots=min_shots)
    detector = load_detector(model, imgsz)
    if mode == "singles":
        names = [n1.strip() or "Player A", n2.strip() or "Player B"]
    else:
        ta, tb = n1.strip() or "Team A", n2.strip() or "Team B"
        if ta == tb:
            tb += " (2)"
        names = [ta, tb, a1.strip() or "Player A1", a2.strip() or "Player A2",
                 b1.strip() or "Player B1", b2.strip() or "Player B2"]
        st.session_state.roster = f"**{ta}:** {names[2]} / {names[3]}  ·  **{tb}:** {names[4]} / {names[5]}"
    out_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(vpath))[0] + "_analysed.mp4") \
        if save_video else None

    prog = st.progress(0.0, text="Starting…")
    col_v, col_s = st.columns([3, 2])
    frame_ph = col_v.empty()
    with col_s:
        st.subheader("📡 Live stats")
        stats_ph = st.empty()
        chart_ph = st.empty()
    live_rows, result, n_up = [], None, 0
    live_colors = None
    try:
        for up in az.analyze_video(vpath, cfg, corners, names, detector, clip_start, end_val, out_path):
            if up.done:
                result, st.session_state.video_out = up.result, up.video_path
                break
            n_up += 1
            prog.progress(up.progress, text=f"Analysing… {up.progress * 100:.0f}% · ETA {up.eta:.0f}s")
            show_image(frame_ph, cv2.cvtColor(up.frame, cv2.COLOR_BGR2RGB))
            snap = up.snap
            if live_colors is None:
                live_colors = pcolor(list(snap["players"].keys()))
            items = [kpi("Match time", az.fmt_t(snap["t"]), icon="⏱️", color="#0ea5e9"),
                     kpi("Rallies", snap["rallies"], icon="🔁", color="#8b5cf6"),
                     kpi("State", "Rally" if snap["in_rally"] else "Break",
                         f"{snap['cur_shots'] if snap['in_rally'] else 0} shots",
                         "🟢" if snap["in_rally"] else "⚪", "#22c55e" if snap["in_rally"] else "#94a3b8")]
            for lab, p in snap["players"].items():
                items.append(kpi(lab, f"{p['speed']:.1f} m/s", f"top {p['vmax']:.1f} · {p['dist']:.0f} m · "
                                 f"{p['hits']} strokes", "🏃", live_colors[lab]))
            kpi_row(items, stats_ph)
            row = {"t": snap["t"]}
            row.update({lab: p["speed"] for lab, p in snap["players"].items()})
            live_rows.append(row)
            ldf = pd.DataFrame(live_rows).tail(250)
            lf = go.Figure()
            for lab in snap["players"]:
                lf.add_scatter(x=ldf.t, y=ldf[lab], name=lab, mode="lines", line=dict(color=live_colors[lab], width=2.5),
                               fill="tozeroy", fillcolor=hex_rgba(live_colors[lab], 0.12))
            style(lf, 230, "Live speed (m/s)")
            chart_ph.plotly_chart(lf, key=f"live_{n_up}")
    except Exception as e:
        st.error(f"Analysis failed: {e}")
    if result is not None:
        st.session_state.result = result
        prog.progress(1.0, text="Done ✅")
        time.sleep(0.4)
        st.rerun()

result = st.session_state.get("result")
if result is None:
    st.stop()

# --------------------------------------------------------------------------------------
# Step 4 - report
# --------------------------------------------------------------------------------------
step(4, "Match report", "KPIs, player profiles, rallies, heatmaps and coaching notes")
R, meta = result, result.meta
labels = meta["labels"]
colors = pcolor(labels)
if R.summary.empty:
    for m in R.insights:
        st.warning(m)
    st.stop()

if meta["valid_ratio"] < 0.6:
    st.warning(f"Players were found on both sides in only {meta['valid_ratio'] * 100:.0f}% of analysed frames "
               "(replays/zooms or wrong calibration?). Numbers may be incomplete.")

tabs = st.tabs(["📊 Overview", "🏃 Players", "🔁 Rallies", "🔥 Heatmaps", "🎯 Shots", "🧠 Coach insights",
                "🎬 Video & export"])
M, S = R.match, R.summary

# ---------------- Overview ----------------
with tabs[0]:
    if meta.get("mode") == "doubles" and st.session_state.get("roster"):
        st.caption(st.session_state.roster)
    kpi_row([kpi("Analysed", az.fmt_t(M["clip_duration"]), icon="🕒", color="#0ea5e9"),
             kpi("Rallies", M["rallies"], icon="🔁", color="#8b5cf6"),
             kpi("Avg rally", fmt_or_dash(M.get("avg_rally_s"), "{:.1f}", " s"), icon="⏳", color="#22c55e"),
             kpi("Longest rally", fmt_or_dash(M.get("longest_rally_s"), "{:.1f}", " s"),
                 f"{M.get('max_shots', 0)} shots", "🏆", "#f59e0b")])
    kpi_row([kpi("Total shots", M.get("total_shots", 0), icon="🏸", color="#e84393"),
             kpi("Shots / rally", fmt_or_dash(M.get("avg_shots"), "{:.1f}"), icon="📈", color="#06b6d4"),
             kpi("Smash-like", M.get("total_smashes", 0), icon="💥", color="#ef4444"),
             kpi("Avg rest", fmt_or_dash(M.get("avg_rest_s"), "{:.0f}", " s"), "between rallies", "😮‍💨", "#64748b")])

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(radar_fig(S, labels, colors))
    with c2:
        act = M.get("active_ratio")
        if act == act and act is not None:
            act = min(max(act, 0), 1)
            st.plotly_chart(donut([act * 100, (1 - act) * 100], ["Rally (work)", "Rest / other"],
                                  ["#22c55e", "#94a3b8"], "Time in play", 400, f"<b>{act * 100:.0f}%</b><br>active"))
    cols = st.columns(max(len(labels), 1))
    for col, lab in zip(cols, labels):
        if lab in S.index:
            col.markdown(player_card(lab, S.loc[lab], colors[lab]), unsafe_allow_html=True)

    show = pd.DataFrame({
        "Distance (m)": S.distance_m.round(0), "Avg speed (m/s)": S.avg_speed_ms.round(2),
        "Rally speed (m/s)": S.avg_rally_speed_ms.round(2), "Peak p99 (m/s)": S.peak_speed_ms.round(2),
        "Strokes": S.shots, "Smash-like": S.smashes, "Recovery (m)": S.recovery_dist_m.round(2)})
    st.dataframe(show, column_config={
        "Distance (m)": st.column_config.ProgressColumn("Distance (m)", min_value=0, max_value=float(max(S.distance_m.max(), 1)), format="%.0f"),
        "Peak p99 (m/s)": st.column_config.ProgressColumn("Peak p99 (m/s)", min_value=0, max_value=9.0, format="%.1f")})

# ---------------- Players ----------------
with tabs[1]:
    for lab in labels:
        if lab not in S.index:
            continue
        r = S.loc[lab]
        st.markdown(player_card(lab, r, colors[lab]), unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        g1.plotly_chart(gauge(r.peak_speed_ms, "Peak speed", colors[lab], 9), key=f"g1{lab}")
        g2.plotly_chart(gauge(r.avg_rally_speed_ms, "Avg speed in rally", colors[lab], 4), key=f"g2{lab}")
        g3.plotly_chart(gauge(r.sprint_pct, "Sprint share", colors[lab], 40, "%"), key=f"g3{lab}")
        rec = r.recovery_dist_m if r.recovery_dist_m == r.recovery_dist_m else 0
        g4.plotly_chart(gauge(rec, "Recovery distance", colors[lab], 4, " m"), key=f"g4{lab}")

    tr = R.tracking
    win = max(int(2 * meta["eff_fps"]), 3) * (2 if meta.get("mode") == "doubles" else 1)
    fig = go.Figure()
    for lab in labels:
        g = tr[tr.player == lab]
        if len(g):
            fig.add_scatter(x=g.t, y=g.speed.rolling(win, min_periods=1).mean(), name=lab, mode="lines",
                            line=dict(color=colors[lab], width=2.5, shape="spline"),
                            fill="tozeroy", fillcolor=hex_rgba(colors[lab], 0.14))
    for r_ in R.rallies.head(150).itertuples():
        fig.add_vrect(x0=r_.start, x1=r_.end, fillcolor="#22c55e", opacity=0.10, line_width=0, layer="below")
    style(fig, 380, "Speed timeline (2 s rolling avg · green bands = rallies)")
    fig.update_xaxes(title="video time (s)")
    fig.update_yaxes(title="m/s")
    st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    with c1:
        tr2 = tr.copy()
        tr2["minute"] = ((tr2.t - meta["clip_start"]) // 60).astype(int) + 1
        pm = tr2.groupby(["minute", "player"]).step.sum().reset_index()
        fig = go.Figure()
        for lab in labels:
            g = pm[pm.player == lab]
            fig.add_bar(x=g.minute, y=g.step, name=lab, marker_color=colors[lab], marker_cornerradius=6)
        fig.update_layout(barmode="group")
        style(fig, 340, "Distance per minute (m) · workload / fatigue")
        fig.update_xaxes(title="minute", dtick=1)
        st.plotly_chart(fig)
    with c2:
        fig = go.Figure()
        for key, name, col in [("stand_pct", "Standing <0.5", "#94a3b8"), ("walk_pct", "Walking 0.5–2", "#38bdf8"),
                               ("jog_pct", "Moving 2–4", "#fbbf24"), ("sprint_pct", "Sprint >4", "#ef4444")]:
            fig.add_bar(y=S.index, x=S[key], name=name, marker_color=col, orientation="h",
                        text=S[key].round(0).astype(int).astype(str) + "%", textposition="inside")
        fig.update_layout(barmode="stack")
        style(fig, 340, "Movement intensity (% of rally time, m/s)")
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for key, name, col in [("front_pct", "Front", "#22c55e"), ("mid_pct", "Mid", "#facc15"),
                               ("back_pct", "Back", "#f97316")]:
            fig.add_bar(y=S.index, x=S[key], name=name, marker_color=col, orientation="h",
                        text=S[key].round(0).astype(int).astype(str) + "%", textposition="inside")
        fig.update_layout(barmode="stack")
        style(fig, 300, "Court depth occupancy (% of rally time)")
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(fig)
    with c2:
        fig = go.Figure()
        fig.add_bar(y=S.index, x=S.left_pct, name="Left half", marker_color="#a855f7", orientation="h",
                    text=S.left_pct.round(0).astype(int).astype(str) + "%", textposition="inside")
        fig.add_bar(y=S.index, x=S.right_pct, name="Right half", marker_color="#14b8a6", orientation="h",
                    text=S.right_pct.round(0).astype(int).astype(str) + "%", textposition="inside")
        fig.update_layout(barmode="stack")
        style(fig, 300, "Left / right split (% of rally time)")
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(fig)
    st.caption("Recovery distance = average distance from the 'base' position (centre of own half) when the "
               "opponent strikes the shuttle. Lower is better.")

# ---------------- Rallies ----------------
with tabs[2]:
    rd = R.rallies
    if rd.empty:
        st.info("No rallies detected - lower the stroke threshold / min shots in the sidebar and re-run.")
    else:
        top = rd.sort_values("duration", ascending=False).head(3)
        medals = ["🥇", "🥈", "🥉"]
        kpi_row([kpi(f"Rally #{int(r.id)} · at {az.fmt_t(r.start)}", f"{r.duration:.1f} s", f"{int(r.shots)} shots · {int(r.smashes)} smash-like",
                     medals[i], ["#f59e0b", "#94a3b8", "#b45309"][i]) for i, r in enumerate(top.itertuples())])
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure(go.Scatter(
                x=rd.start, y=rd.duration, mode="markers",
                marker=dict(size=np.clip(rd.shots * 3.2, 10, 48), color=rd.shots, colorscale="Turbo", showscale=True,
                            colorbar=dict(title="shots"), line=dict(color="white", width=1.5), opacity=0.9),
                customdata=np.c_[rd.id, rd.shots, rd.smashes],
                hovertemplate="Rally %{customdata[0]}<br>%{y:.1f}s · %{customdata[1]} shots<br>%{customdata[2]} smash-like<extra></extra>"))
            style(fig, 360, "Rallies through the match (size = shots)")
            fig.update_xaxes(title="video time (s)")
            fig.update_yaxes(title="duration (s)")
            st.plotly_chart(fig)
        with c2:
            fig = go.Figure(go.Histogram(x=rd.shots, nbinsx=14, marker=dict(color="#0ea5e9", cornerradius=6)))
            style(fig, 360, "Shots per rally")
            fig.update_xaxes(title="shots")
            st.plotly_chart(fig)
        t = rd.copy()
        t["start"] = t.start.map(az.fmt_t)
        t["end"] = t.end.map(az.fmt_t)
        cols = ["id", "start", "end", "duration", "shots", "smashes"] + \
               [c for c in t.columns if c.startswith(("shots_", "dist_", "spd_"))]
        st.dataframe(t[cols].round(2), column_config={
            "duration": st.column_config.ProgressColumn("duration", min_value=0, max_value=float(rd.duration.max()), format="%.1f s"),
            "shots": st.column_config.ProgressColumn("shots", min_value=0, max_value=int(rd.shots.max()), format="%d")})

# ---------------- Heatmaps ----------------
with tabs[3]:
    st.caption("Heatmaps use rally time only. Court shown from the near baseline (bottom) to the far baseline (top).")
    cols = st.columns(2)
    for i, lab in enumerate(labels):
        base = R.tracking[R.tracking.player == lab]
        g = base[base.in_rally] if base.in_rally.any() else base
        if len(g):
            with cols[i % 2]:
                st.plotly_chart(heat_fig(g, f"{lab} · position heatmap", colors[lab]), key=f"heat{lab}")

# ---------------- Shots ----------------
with tabs[4]:
    h = R.hits[R.hits.rally_id > 0]
    if h.empty:
        st.info("No strokes in rallies detected.")
    else:
        cnt = h.groupby("player").agg(strokes=("t", "count"), smashes=("smash", "sum"), avg_intensity=("v", "mean"))
        kpi_row([kpi(lab, f"{int(r.strokes)} strokes", f"{int(r.smashes)} smash-like · intensity {r.avg_intensity:.0f}",
                     "🏸", colors.get(lab, ACCENT)) for lab, r in cnt.iterrows()])
        fig = go.Figure()
        for lab in labels:
            g = h[h.player == lab]
            if len(g):
                fig.add_scatter(x=g.x, y=g.y, mode="markers", name=lab,
                                marker=dict(color=colors[lab], size=np.where(g.smash, 18, 10),
                                            symbol=np.where(g.smash, "star", "circle"),
                                            line=dict(color="white", width=1.2)),
                                hovertemplate="stroke intensity %{customdata:.0f}<extra></extra>", customdata=g.v)
        fig.update_layout(title=dict(text="Where strokes were played (★ = smash-like)", x=0.01, font=dict(size=16)))
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(add_court(fig, 600))
        with c2:
            st.plotly_chart(donut(cnt.strokes.tolist(), cnt.index.tolist(), [colors.get(l, ACCENT) for l in cnt.index],
                                  "Share of strokes", 280, f"<b>{int(cnt.strokes.sum())}</b><br>strokes"))
            fig2 = go.Figure()
            for lab in labels:
                g = h[h.player == lab]
                if len(g):
                    fig2.add_histogram(x=g.v, name=lab, marker_color=colors[lab], opacity=0.7)
            fig2.update_layout(barmode="overlay")
            style(fig2, 280, "Stroke intensity distribution")
            st.plotly_chart(fig2)
        st.caption("Strokes are inferred from racket-arm (wrist) speed normalised by torso length - an approximation. "
                   "Tune thresholds in the sidebar if you see too many / too few.")

# ---------------- Coach insights ----------------
with tabs[5]:
    st.subheader("Automatic coaching notes")
    for m in R.insights:
        ic, c = insight_icon(m)
        body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(m))
        st.markdown(f'<div class="ins" style="--c:{c}"><div class="ic">{ic}</div><div class="tx">{body}</div></div>',
                    unsafe_allow_html=True)
    st.subheader("Questions to explore with the player")
    for q in ["Which rallies were longest, and how did the lower-moving player win or lose them?",
              "Does recovery distance rise late in the match? (compare speed per rally)",
              "Are opponent strokes forcing one player into one corner? (Heatmaps tab)",
              "Is the smash count consistent with the game plan?"]:
        st.markdown(f'<div class="ins" style="--c:#64748b"><div class="ic">❓</div><div class="tx">{esc(q)}</div></div>',
                    unsafe_allow_html=True)

# ---------------- Video & export ----------------
with tabs[6]:
    vo = st.session_state.get("video_out")
    if vo and os.path.exists(vo):
        st.video(vo)
        with open(vo, "rb") as f:
            st.download_button("⬇️ Download annotated video", f, file_name=os.path.basename(vo), type="primary")
    else:
        st.info("No annotated video was saved (enable it in the sidebar).")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("📄 Tracking CSV", R.tracking.to_csv(index=False), "tracking.csv")
    c2.download_button("📄 Rallies CSV", R.rallies.to_csv(index=False), "rallies.csv")
    c3.download_button("📄 Strokes CSV", R.hits.to_csv(index=False), "strokes.csv")
    report = dict(match=M, players=json.loads(S.to_json(orient="index")), insights=R.insights, meta=dict(meta))
    c4.download_button("🧾 Summary JSON", json.dumps(report, indent=2, default=str), "summary.json")
    st.caption(f"Processed {meta['samples']} frames in {meta['processing_seconds']:.0f}s "
               f"({meta['eff_fps']:.1f} analysed fps).")
