"""
analyzer.py - Badminton match analytics engine (OpenCV + YOLO-pose)

Pipeline
--------
1. Download a YouTube video locally (yt-dlp).
2. Player detection + body keypoints with YOLOv8-pose (OpenCV handles all video I/O and drawing).
3. A 4-point court calibration -> homography -> every player position is converted to *metres*
   on a standard 6.1 m x 13.4 m court.
4. Metrics: speed, distance, court coverage, zones, recovery position, shot (hit) detection from
   wrist speed, smash-like shots, rally segmentation, rest time, fatigue trend, coach insights.

Honest limitations
------------------
* Needs a mostly static camera showing the whole court (rear / high-angle view works best).
  Broadcast footage with replays / zooms is handled by ignoring frames where the court view is lost.
* The shuttle is not tracked. Hits are estimated from racket-arm (wrist) speed and rallies are
  built from alternating hits. Point winner / score is NOT detected.
* All numbers are estimates - treat them as coaching indicators, not official statistics.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Court geometry (metres).  x: 0..6.1 (left->right seen from near baseline), y: 0 = near
# baseline, 13.4 = far baseline, net at y = 6.7
# --------------------------------------------------------------------------------------
COURT_W = 6.10
COURT_L = 13.40
NET_Y = COURT_L / 2
SHORT_SERVICE = 1.98
FRONT_LIM = 1.98   # distance from net: front court
MID_LIM = 4.20     # mid court up to here, then back court
COURT_DST = np.float32([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]])

NEAR_COLOR = (0, 140, 255)   # BGR orange
FAR_COLOR = (255, 180, 0)    # BGR cyan-ish
COLORS = {"near": NEAR_COLOR, "far": FAR_COLOR}


def fmt_t(sec: float) -> str:
    sec = int(round(max(sec, 0)))
    return f"{sec // 60}:{sec % 60:02d}"


# --------------------------------------------------------------------------------------
# Video helpers
# --------------------------------------------------------------------------------------
def ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def download_youtube(url: str, out_dir: str = "downloads", max_height: int = 720, progress_cb=None):
    """Download video-only stream (no audio needed) so no ffmpeg merge is required."""
    import yt_dlp

    os.makedirs(out_dir, exist_ok=True)

    def hook(d):
        if not progress_cb:
            return
        if d.get("status") == "downloading":
            tot = d.get("total_bytes") or d.get("total_bytes_estimate")
            if tot:
                progress_cb(min(d.get("downloaded_bytes", 0) / tot, 1.0))
        elif d.get("status") == "finished":
            progress_cb(1.0)

    h = int(max_height)
    fmt = (f"bestvideo[height<={h}][vcodec^=avc1]/bestvideo[height<={h}][ext=mp4]/"
           f"best[height<={h}][ext=mp4]/best[height<={h}]/best")
    opts = {
        "format": fmt,
        "outtmpl": os.path.join(out_dir, "%(id)s_%(height)sp.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    if not os.path.exists(path):
        base = os.path.splitext(path)[0]
        for ext in ("mp4", "mkv", "webm"):
            if os.path.exists(f"{base}.{ext}"):
                path = f"{base}.{ext}"
                break
    meta = {"title": info.get("title", ""), "duration": info.get("duration"), "id": info.get("id")}
    return path, meta


def transcode_h264(src: str, dst: str) -> bool:
    exe = ffmpeg_exe()
    if not exe:
        return False
    cmd = [exe, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-crf", "23", "-movflags", "+faststart", "-an", dst]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(dst)
    except Exception:
        return False


def ensure_readable(path: str) -> str:
    """If OpenCV can't decode the file (e.g. AV1/VP9 build issues) transcode it to H.264."""
    cap = cv2.VideoCapture(path)
    ok, _ = cap.read()
    cap.release()
    if ok:
        return path
    dst = os.path.splitext(path)[0] + "_h264.mp4"
    if not os.path.exists(dst) and not transcode_h264(path, dst):
        raise RuntimeError("OpenCV cannot decode this video and ffmpeg is unavailable to convert it.")
    return dst


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "width": w, "height": h, "duration": n / fps if fps else 0.0}


def read_frame(path: str, t_sec: float):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------
def compute_homography(corners_px) -> np.ndarray:
    """corners order: near-left, near-right, far-right, far-left (outer doubles lines)."""
    return cv2.getPerspectiveTransform(np.float32(corners_px), COURT_DST)


def apply_h(H: np.ndarray, pt) -> np.ndarray:
    return cv2.perspectiveTransform(np.float32([[pt]]), H)[0, 0]


# --------------------------------------------------------------------------------------
# Pose detector (YOLOv8-pose). Anything with .detect(frame)->list[dict] can be injected.
# --------------------------------------------------------------------------------------
class PoseDetector:
    def __init__(self, model: str = "yolov8n-pose.pt", conf: float = 0.3, imgsz: int = 960):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame) -> List[dict]:
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        if res.keypoints is not None and res.keypoints.xy is not None:
            kxy = res.keypoints.xy.cpu().numpy()
            kc = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
                  else np.ones(kxy.shape[:2]))
        else:
            kxy = np.zeros((len(boxes), 17, 2))
            kc = np.zeros((len(boxes), 17))
        return [dict(box=b, conf=float(c), kp=k, kpc=q) for b, c, k, q in zip(boxes, confs, kxy, kc)]


def foot_point(d: dict) -> np.ndarray:
    kp, kc = d["kp"], d["kpc"]
    ank = [kp[i] for i in (15, 16) if kc[i] > 0.3]
    if ank:
        return np.mean(ank, axis=0)
    x1, y1, x2, y2 = d["box"]
    return np.array([(x1 + x2) / 2, y2], dtype=float)


def torso_len(d: dict) -> float:
    kp, kc = d["kp"], d["kpc"]
    ls = []
    for s, h in ((5, 11), (6, 12)):
        if kc[s] > 0.3 and kc[h] > 0.3:
            ls.append(np.linalg.norm(kp[s] - kp[h]))
    if ls:
        return max(float(np.mean(ls)), 5.0)
    x1, y1, x2, y2 = d["box"]
    return max(0.3 * (y2 - y1), 5.0)


def wrists(d: dict) -> np.ndarray:
    kp, kc = d["kp"], d["kpc"]
    w = np.full((2, 2), np.nan)
    for i, idx in enumerate((9, 10)):
        if kc[idx] > 0.3:
            w[i] = kp[idx]
    return w


# --------------------------------------------------------------------------------------
# Config / state containers
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    mode: str = "singles"          # "singles" | "doubles"
    sample_fps: float = 12.0       # analysed frames per second of video
    hit_thr: float = 12.0          # wrist speed (torso-lengths / s) that counts as a stroke
    smash_thr: float = 35.0        # ... that counts as a smash-like stroke
    hit_debounce: float = 0.5      # min seconds between two hits of the same side
    rally_gap: float = 3.0         # max seconds between hits inside one rally
    min_shots: int = 3             # min hits for a valid rally
    max_speed: float = 8.5         # m/s cap (physical limit guard)
    out_width: int = 960           # annotated video width


@dataclass
class Slot:
    key: str
    side: str
    label: str
    base: np.ndarray
    pos: Optional[np.ndarray] = None
    t: float = -1e9
    hist: deque = field(default_factory=lambda: deque(maxlen=80))
    speed: float = 0.0
    dist: float = 0.0
    vmax: float = 0.0
    hits: int = 0
    wprev: Optional[np.ndarray] = None
    wt: float = -1e9
    wv: deque = field(default_factory=lambda: deque(maxlen=3))
    trail: deque = field(default_factory=lambda: deque(maxlen=45))
    flash_until: float = -1.0
    flash_text: str = ""
    name: str = ""          # individual name shown on the video (label = team in doubles)


@dataclass
class Update:
    progress: float
    frame: Optional[np.ndarray] = None
    snap: Optional[dict] = None
    eta: float = 0.0
    done: bool = False
    result: Optional["Result"] = None
    video_path: Optional[str] = None


@dataclass
class Result:
    tracking: pd.DataFrame
    hits: pd.DataFrame
    rallies: pd.DataFrame
    summary: pd.DataFrame
    match: dict
    insights: List[str]
    meta: dict


# --------------------------------------------------------------------------------------
# Core analyzer
# --------------------------------------------------------------------------------------
class MatchAnalyzer:
    def __init__(self, cfg: Config, corners_px, names: List[str], detector, eff_fps: float = 12.0):
        self.cfg = cfg
        self.H = compute_homography(corners_px)
        self.Hinv = np.linalg.inv(self.H)
        self.corners = np.int32(corners_px)
        self.detector = detector
        self.eff_fps = eff_fps
        self.slots = self._make_slots(names)
        self.rows: list = []
        self.hits: list = []
        self.ready_rows: list = []
        self.rallies: list = []
        self.cur: Optional[dict] = None
        self.last_hit_t = {"near": -1e9, "far": -1e9}
        self.valid = False
        self.last_valid_t = -1e9
        self.prev_hist = None
        self.n_samples = 0
        self.valid_samples = 0

    # ---- setup ---------------------------------------------------------------------
    def _make_slots(self, names):
        """singles: names = [A, B].  doubles: names = [TeamA, TeamB, A1, A2, B1, B2] (or just 2 team names).
        In doubles both players of a side share the team label, so every chart shows Team A / Team B only."""
        a, b = names[0], names[1]
        d = 3.0
        if self.cfg.mode == "singles":
            return {
                "near": Slot("near", "near", a, np.array([COURT_W / 2, NET_Y - d]), name=a),
                "far": Slot("far", "far", b, np.array([COURT_W / 2, NET_Y + d]), name=b),
            }
        xl, xr = COURT_W * 0.25, COURT_W * 0.75
        pn = list(names[2:6]) if len(names) >= 6 else [f"{a} (L)", f"{a} (R)", f"{b} (L)", f"{b} (R)"]
        return {
            "near_L": Slot("near_L", "near", a, np.array([xl, NET_Y - d]), name=pn[0]),
            "near_R": Slot("near_R", "near", a, np.array([xr, NET_Y - d]), name=pn[1]),
            "far_L": Slot("far_L", "far", b, np.array([xl, NET_Y + d]), name=pn[2]),
            "far_R": Slot("far_R", "far", b, np.array([xr, NET_Y + d]), name=pn[3]),
        }

    # ---- player selection ----------------------------------------------------------
    def _select(self, dets, frame_area) -> Dict[str, dict]:
        n_per_side = 1 if self.cfg.mode == "singles" else 2
        cands = {"near": [], "far": []}
        for d in dets:
            c = apply_h(self.H, foot_point(d))
            if not (-1.2 <= c[0] <= COURT_W + 1.2 and -2.0 <= c[1] <= COURT_L + 2.0):
                continue
            inside = 0 <= c[0] <= COURT_W and 0 <= c[1] <= COURT_L
            x1, y1, x2, y2 = d["box"]
            area = (x2 - x1) * (y2 - y1) / frame_area
            d["court"] = c
            d["foot_px"] = foot_point(d)
            d["score"] = d["conf"] + (0.5 if inside else 0.0) + min(area, 0.05) * 5
            side = "near" if c[1] < NET_Y else "far"
            d["side"] = side
            cands[side].append(d)
        sel = {}
        for side, lst in cands.items():
            lst = sorted(lst, key=lambda z: -z["score"])[:n_per_side]
            if self.cfg.mode == "singles":
                if lst:
                    sel[side] = lst[0]
            else:
                lst = sorted(lst, key=lambda z: z["court"][0])
                if len(lst) == 2:
                    sel[f"{side}_L"], sel[f"{side}_R"] = lst
                elif len(lst) == 1:
                    sel[f"{side}_L" if lst[0]["court"][0] < COURT_W / 2 else f"{side}_R"] = lst[0]
        return sel

    # ---- per-slot update -----------------------------------------------------------
    def _update_slot(self, s: Slot, det: dict, t: float):
        xy = det["court"].astype(float)
        step = 0.0
        if s.pos is None or t - s.t > 1.0 or np.linalg.norm(xy - s.pos) > 3.5:
            s.pos = xy.copy()
            s.hist.clear()
            s.wprev = None
            s.wv.clear()
            s.speed = 0.0
        else:
            new = 0.4 * xy + 0.6 * s.pos
            step = float(np.linalg.norm(new - s.pos))
            if step < 0.03:          # dead-band against detection jitter
                step = 0.0
            s.pos = new
        s.t = t
        s.hist.append((t, s.pos.copy()))
        # speed over ~0.5 s window
        ref = None
        for (t0, p0) in s.hist:
            if t - t0 <= 0.6:
                ref = (t0, p0)
                break
        if ref is not None and t - ref[0] >= 0.25:
            s.speed = min(float(np.linalg.norm(s.pos - ref[1]) / (t - ref[0])), self.cfg.max_speed)
        s.dist += step
        s.vmax = max(s.vmax, s.speed)
        s.trail.append(det["foot_px"].copy())
        self.rows.append((t, s.key, s.label, s.side, float(s.pos[0]), float(s.pos[1]), s.speed, step))

        # ---- stroke detection from wrist speed ----
        w = wrists(det)
        torso = torso_len(det)
        v = 0.0
        if s.wprev is not None and 0 < t - s.wt <= 0.5:
            dif = np.linalg.norm(w - s.wprev, axis=1)
            if np.isfinite(dif).any():
                v = float(np.nanmax(dif) / (t - s.wt) / torso)
        s.wprev, s.wt = w, t
        s.wv.append((t, v))
        if len(s.wv) == 3:
            (_, v0), (t1, v1), (_, v2) = s.wv
            if v1 >= self.cfg.hit_thr and v1 >= v0 and v1 >= v2:
                self._register_hit(s, t1, v1)

    def _register_hit(self, s: Slot, t: float, v: float):
        if t - self.last_hit_t[s.side] < self.cfg.hit_debounce:
            return
        self.last_hit_t[s.side] = t
        smash = v >= self.cfg.smash_thr
        hit = dict(t=t, slot=s.key, player=s.label, side=s.side, v=v, smash=smash,
                   x=float(s.pos[0]), y=float(s.pos[1]))
        self.hits.append(hit)
        s.hits += 1
        s.flash_until = t + 0.4
        s.flash_text = "SMASH" if smash else "HIT"
        for o in self.slots.values():          # where is the opponent when the shuttle is struck?
            if o.side != s.side and o.pos is not None and t - o.t < 0.5:
                self.ready_rows.append((t, o.key, o.label, float(np.linalg.norm(o.pos - o.base))))
        # rally state machine
        c = self.cur
        if c and t - c["last"] > self.cfg.rally_gap:
            self._close_rally()
            c = None
        if c is None:
            self.cur = {"start": t, "last": t, "hits": [hit]}
        else:
            c["hits"].append(hit)
            c["last"] = t

    def _close_rally(self):
        c, self.cur = self.cur, None
        if not c:
            return
        sides = {h["side"] for h in c["hits"]}
        if len(c["hits"]) >= self.cfg.min_shots and len(sides) == 2:
            self.rallies.append(dict(id=len(self.rallies) + 1, start=c["start"], end=c["last"],
                                     shots=len(c["hits"]),
                                     smashes=int(sum(h["smash"] for h in c["hits"]))))

    @property
    def in_rally(self) -> bool:
        return bool(self.cur and len(self.cur["hits"]) >= 2)

    # ---- one frame -----------------------------------------------------------------
    def process(self, frame, t: float, draw: bool = True):
        self.n_samples += 1
        small = cv2.resize(frame, (64, 36))
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        cv2.normalize(hist, hist)
        cut = False
        if self.prev_hist is not None and cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL) < 0.6:
            cut = True
            for s in self.slots.values():
                s.pos = None
                s.wprev = None
        self.prev_hist = hist

        sel = {}
        if not cut:
            h, w = frame.shape[:2]
            sel = self._select(self.detector.detect(frame), float(h * w))
            if self.cfg.mode == "singles":
                self.valid = "near" in sel and "far" in sel
            else:
                self.valid = any(k.startswith("near") for k in sel) and any(k.startswith("far") for k in sel)
        else:
            self.valid = False

        if self.valid:
            self.valid_samples += 1
            self.last_valid_t = t
            for k, d in sel.items():
                self._update_slot(self.slots[k], d, t)
        if self.cur and t - self.cur["last"] > self.cfg.rally_gap:
            self._close_rally()
        return self._draw(frame, sel, t) if draw else None

    # ---- drawing -------------------------------------------------------------------
    def _draw(self, frame, sel, t):
        img = frame.copy()
        h, w = img.shape[:2]
        fs = max(0.5, h / 900 * 0.8)
        th = max(1, int(h / 500))
        cv2.polylines(img, [self.corners.reshape(-1, 1, 2)], True, (80, 255, 120), th)
        for k, d in sel.items():
            s = self.slots[k]
            col = COLORS[s.side]
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, th)
            lab = f"{s.name or s.label} {s.speed:.1f} m/s"
            cv2.putText(img, lab, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th + 1, cv2.LINE_AA)
            fp = tuple(int(v) for v in d["foot_px"])
            cv2.circle(img, fp, th * 3, col, -1)
            if len(s.trail) > 1:
                cv2.polylines(img, [np.int32(list(s.trail)).reshape(-1, 1, 2)], False, col, th)
            if t <= s.flash_until:
                cv2.putText(img, s.flash_text, (x1, y2 + int(25 * fs)), cv2.FONT_HERSHEY_SIMPLEX,
                            fs * 1.2, (0, 0, 255) if s.flash_text == "SMASH" else (255, 255, 255),
                            th + 1, cv2.LINE_AA)
        self._minimap(img, sel)
        # HUD
        bar = img.copy()
        cv2.rectangle(bar, (0, 0), (int(w * 0.62), int(38 * fs) + 10), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.55, img, 0.45, 0, img)
        state = "RALLY" if self.in_rally else "BREAK"
        shots = len(self.cur["hits"]) if self.in_rally else 0
        txt = f"{fmt_t(t)}  Rallies:{len(self.rallies)}  {state}  shots:{shots}"
        cv2.putText(img, txt, (10, int(28 * fs) + 4), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 255, 0) if self.in_rally else (255, 255, 255), th + 1, cv2.LINE_AA)
        if not self.valid:
            cv2.putText(img, "court view / players not found", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (0, 0, 255), th + 1, cv2.LINE_AA)
        if w > self.cfg.out_width:
            sc = self.cfg.out_width / w
            img = cv2.resize(img, (self.cfg.out_width, int(h * sc)), interpolation=cv2.INTER_AREA)
        return img

    def _minimap(self, img, sel):
        h, w = img.shape[:2]
        sc = max(10, int(h / 40))
        mw, mh, pad = int(COURT_W * sc), int(COURT_L * sc), 6
        x0, y0 = w - mw - 2 * pad - 10, 10
        ov = img.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + mw + 2 * pad, y0 + mh + 2 * pad), (40, 110, 60), -1)
        cv2.addWeighted(ov, 0.75, img, 0.25, 0, img)

        def m(p):
            return int(x0 + pad + p[0] * sc), int(y0 + pad + (COURT_L - p[1]) * sc)

        for (a, b) in [((0, 0), (COURT_W, 0)), ((COURT_W, 0), (COURT_W, COURT_L)),
                       ((COURT_W, COURT_L), (0, COURT_L)), ((0, COURT_L), (0, 0)),
                       ((0.46, 0), (0.46, COURT_L)), ((COURT_W - 0.46, 0), (COURT_W - 0.46, COURT_L)),
                       ((0, NET_Y - SHORT_SERVICE), (COURT_W, NET_Y - SHORT_SERVICE)),
                       ((0, NET_Y + SHORT_SERVICE), (COURT_W, NET_Y + SHORT_SERVICE)),
                       ((COURT_W / 2, 0), (COURT_W / 2, NET_Y - SHORT_SERVICE)),
                       ((COURT_W / 2, NET_Y + SHORT_SERVICE), (COURT_W / 2, COURT_L))]:
            cv2.line(img, m(a), m(b), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, m((0, NET_Y)), m((COURT_W, NET_Y)), (255, 255, 255), 3)
        for s in self.slots.values():
            if s.pos is not None and k_in(sel, s.key):
                cv2.circle(img, m(s.pos), max(4, sc // 3), COLORS[s.side], -1, cv2.LINE_AA)

    # ---- live snapshot ---------------------------------------------------------------
    def snapshot(self, t: float) -> dict:
        players: dict = {}
        for s in self.slots.values():          # doubles: both players of a team are combined
            p = players.setdefault(s.label, dict(speed=0.0, dist=0.0, vmax=0.0, hits=0, side=s.side, _n=0))
            p["speed"] += s.speed
            p["_n"] += 1
            p["dist"] += s.dist
            p["vmax"] = max(p["vmax"], s.vmax)
            p["hits"] += s.hits
        for p in players.values():
            p["speed"] /= max(p.pop("_n"), 1)
        return dict(t=t, rallies=len(self.rallies), in_rally=self.in_rally,
                    cur_shots=len(self.cur["hits"]) if self.cur else 0, valid=self.valid, players=players)

    # ---- final aggregation -----------------------------------------------------------
    def finalize(self, meta: dict) -> Result:
        self._close_rally()
        labels = list(dict.fromkeys(s.label for s in self.slots.values()))
        track = pd.DataFrame(self.rows, columns=["t", "slot", "player", "side", "x", "y", "speed", "step"])
        hits = pd.DataFrame(self.hits, columns=["t", "slot", "player", "side", "v", "smash", "x", "y"])
        ready = pd.DataFrame(self.ready_rows, columns=["t", "slot", "player", "dist"])
        rallies = pd.DataFrame(self.rallies, columns=["id", "start", "end", "shots", "smashes"])
        rallies["duration"] = rallies["end"] - rallies["start"]
        meta = dict(meta, labels=labels, valid_ratio=self.valid_samples / max(self.n_samples, 1))

        if track.empty:
            return Result(track, hits, rallies, pd.DataFrame(), {}, [
                "No valid frames were analysed. Check the court calibration (corner order!), that the camera "
                "shows the whole court, and that the chosen time range contains live play."], meta)

        # in-rally flags
        def mask(ts):
            m = np.zeros(len(ts), bool)
            for r in rallies.itertuples():
                m |= (ts >= r.start - 0.4) & (ts <= r.end + 0.4)
            return m

        track["in_rally"] = mask(track["t"].to_numpy())
        if not ready.empty:
            ready = ready[mask(ready["t"].to_numpy())]
        hits["rally_id"] = 0
        for r in rallies.itertuples():
            hits.loc[(hits.t >= r.start) & (hits.t <= r.end), "rally_id"] = r.id
        rh = hits[hits.rally_id > 0]

        # per-rally detail
        det_rows = []
        for r in rallies.itertuples():
            row = dict(id=r.id, start=r.start, end=r.end, duration=r.duration, shots=r.shots, smashes=r.smashes)
            seg = track[(track.t >= r.start - 0.4) & (track.t <= r.end + 0.4)]
            for lab in labels:
                g = seg[seg.player == lab]
                row[f"dist_{lab}"] = float(g.step.sum())
                row[f"spd_{lab}"] = float(g.speed.mean()) if len(g) else np.nan
                row[f"shots_{lab}"] = int(((rh.rally_id == r.id) & (rh.player == lab)).sum())
            det_rows.append(row)
        rallies_det = pd.DataFrame(det_rows)

        # per-player summary (zones / bands / heat use in-rally samples when available)
        base = track[track.in_rally] if track.in_rally.any() else track
        sm = {}
        for lab in labels:
            slot = next(x for x in self.slots.values() if x.label == lab)
            ga, g = track[track.player == lab], base[base.player == lab]
            if ga.empty:
                continue
            depth = (g.y - NET_Y).abs()
            n = max(len(g), 1)
            sp = g.speed
            rd = ready[ready.player == lab]["dist"] if not ready.empty else pd.Series(dtype=float)
            ph = rh[rh.player == lab]
            sm[lab] = {
                "side": slot.side,
                "distance_m": float(ga.step.sum()),
                "avg_speed_ms": float(ga.speed.mean()),
                "avg_rally_speed_ms": float(g.speed.mean()) if len(g) else np.nan,
                "peak_speed_ms": float(ga.speed.quantile(0.99)),
                "shots": int(len(ph)),
                "smashes": int(ph.smash.sum()) if len(ph) else 0,
                "front_pct": 100 * float((depth < FRONT_LIM).sum()) / n,
                "mid_pct": 100 * float(((depth >= FRONT_LIM) & (depth < MID_LIM)).sum()) / n,
                "back_pct": 100 * float((depth >= MID_LIM).sum()) / n,
                "left_pct": 100 * float((g.x < COURT_W / 2).sum()) / n,
                "right_pct": 100 * float((g.x >= COURT_W / 2).sum()) / n,
                "stand_pct": 100 * float((sp < 0.5).sum()) / n,
                "walk_pct": 100 * float(((sp >= 0.5) & (sp < 2)).sum()) / n,
                "jog_pct": 100 * float(((sp >= 2) & (sp < 4)).sum()) / n,
                "sprint_pct": 100 * float((sp >= 4).sum()) / n,
                "recovery_dist_m": float(rd.mean()) if len(rd) else np.nan,
            }
        summary = pd.DataFrame(sm).T
        summary["smash_pct"] = np.where(summary.shots > 0, 100 * summary.smashes / summary.shots.clip(lower=1), 0.0)

        # match level
        clip_dur = float(meta.get("clip_end", 0) - meta.get("clip_start", 0))
        match = dict(clip_duration=clip_dur, rallies=int(len(rallies)))
        if len(rallies):
            rest = (rallies.start.shift(-1) - rallies.end).dropna()
            match.update(
                rally_time=float(rallies.duration.sum()),
                avg_rally_s=float(rallies.duration.mean()),
                longest_rally_s=float(rallies.duration.max()),
                avg_shots=float(rallies.shots.mean()),
                max_shots=int(rallies.shots.max()),
                total_shots=int(rallies.shots.sum()),
                total_smashes=int(rallies.smashes.sum()),
                avg_rest_s=float(rest.mean()) if len(rest) else np.nan,
                work_rest=float(rallies.duration.sum() / max(rest.sum(), 1e-6)) if len(rest) else np.nan,
                active_ratio=float(rallies.duration.sum() / clip_dur) if clip_dur > 0 else np.nan,
                short_rally_pct=100 * float((rallies.shots <= 4).mean()),
            )
        insights = build_insights(summary, rallies_det, match, labels)
        return Result(track, hits, rallies_det, summary, match, insights, meta)


def k_in(sel, key):
    return key in sel


# --------------------------------------------------------------------------------------
# Coach insights (rule based)
# --------------------------------------------------------------------------------------
def build_insights(summary: pd.DataFrame, rallies: pd.DataFrame, match: dict, labels: List[str]) -> List[str]:
    out: List[str] = []
    if summary.empty:
        return out
    if match.get("rallies", 0) == 0:
        return ["No complete rallies were detected. Try lowering the 'stroke threshold' or 'min shots per rally' "
                "in the sidebar, or use a clip with a clearer view of the players' racket arms."]

    out.append(f"**Rally profile:** {match['rallies']} rallies, average {match['avg_rally_s']:.1f}s / "
               f"{match['avg_shots']:.1f} shots, longest {match['longest_rally_s']:.1f}s "
               f"({match['max_shots']} shots). {match['short_rally_pct']:.0f}% of rallies were 4 shots or fewer "
               f"(serve / return dominated).")
    if not np.isnan(match.get("avg_rest_s", np.nan)):
        out.append(f"**Work : rest** ≈ 1 : {1 / max(match['work_rest'], 1e-6):.1f} "
                   f"(average {match['avg_rest_s']:.0f}s between rallies). Long rests (>25 s) suggest slow "
                   f"recovery routines or breaks in the footage.")

    d = summary["distance_m"]
    if len(d) >= 2:
        hi, lo = d.idxmax(), d.idxmin()
        if d[lo] > 0:
            out.append(f"**Movement load:** {hi} covered {d[hi]:.0f} m vs {d[lo]:.0f} m for {lo} "
                       f"({100 * (d[hi] / d[lo] - 1):.0f}% more) - the player running more is usually the one "
                       f"being pulled around the court.")
    for lab, r in summary.iterrows():
        if r.back_pct > 55:
            out.append(f"**{lab}** spends {r.back_pct:.0f}% of rally time in the back court - likely being "
                       f"pushed back; work on attacking clears / drops to regain the net.")
        if r.front_pct > 35:
            out.append(f"**{lab}** is very active at the net ({r.front_pct:.0f}% front court) - net game is a weapon.")
        side = max(r.left_pct, r.right_pct)
        if side > 62:
            which = "left" if r.left_pct > r.right_pct else "right"
            out.append(f"**{lab}** favours the {which} half ({side:.0f}% of time) - possible court imbalance "
                       f"the opponent can exploit.")
        if not np.isnan(r.recovery_dist_m) and r.recovery_dist_m > 1.8:
            out.append(f"**{lab}** is on average {r.recovery_dist_m:.1f} m from the base position when the "
                       f"opponent hits - recovery to centre is late.")
        if r.shots >= 8 and r.smash_pct >= 15:
            out.append(f"**{lab}** hit {r.smashes} smash-like strokes ({r.smash_pct:.0f}% of shots) - a strong attacker.")
        if r.sprint_pct > 12:
            out.append(f"**{lab}** spent {r.sprint_pct:.0f}% of rally time sprinting (>4 m/s) - high explosive demand.")

    if len(rallies) >= 6:
        k = max(len(rallies) // 3, 2)
        for lab in labels:
            col = f"spd_{lab}"
            if col in rallies and rallies[col].notna().sum() >= 2 * k:
                first, last = rallies[col].head(k).mean(), rallies[col].tail(k).mean()
                if first > 0 and (last / first - 1) < -0.10:
                    out.append(f"**{lab}** average rally speed dropped {100 * (1 - last / first):.0f}% from the "
                               f"first to the last third of rallies - possible fatigue.")
    return out


# --------------------------------------------------------------------------------------
# Main generator used by the Streamlit app
# --------------------------------------------------------------------------------------
def analyze_video(path: str, cfg: Config, corners_px, names: List[str], detector,
                  start: float = 0.0, end: Optional[float] = None, out_path: Optional[str] = None,
                  ui_every: int = 6) -> Iterator[Update]:
    info = video_info(path)
    fps = info["fps"]
    stride = max(1, int(round(fps / cfg.sample_fps)))
    eff_fps = fps / stride
    start_f = int(start * fps)
    end_f = int(end * fps) if end else info["frames"]
    end_f = min(end_f, info["frames"]) if info["frames"] else end_f
    total = max(1, (end_f - start_f) // stride)

    an = MatchAnalyzer(cfg, corners_px, names, detector, eff_fps)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    writer = None
    tmp_out = None
    idx, n, t0, last_t = start_f, 0, time.time(), start
    while idx < end_f:
        if not cap.grab():
            break
        if (idx - start_f) % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            last_t = idx / fps
            ann = an.process(frame, last_t, draw=True)
            n += 1
            if out_path:
                if writer is None:
                    tmp_out = out_path + ".tmp.mp4"
                    writer = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), eff_fps,
                                             (ann.shape[1], ann.shape[0]))
                writer.write(ann)
            if n % ui_every == 0:
                el = time.time() - t0
                prog = min(n / total, 1.0)
                yield Update(prog, ann, an.snapshot(last_t), eta=el / prog * (1 - prog) if prog > 0 else 0.0)
        idx += 1
    cap.release()
    video_path = None
    if writer is not None:
        writer.release()
        video_path = out_path if transcode_h264(tmp_out, out_path) else tmp_out
        if video_path == out_path and os.path.exists(tmp_out):
            os.remove(tmp_out)
    meta = dict(fps=fps, eff_fps=eff_fps, stride=stride, clip_start=start, clip_end=last_t, mode=cfg.mode,
                processing_seconds=time.time() - t0, samples=n)
    result = an.finalize(meta)
    yield Update(1.0, None, an.snapshot(last_t), 0.0, True, result, video_path)
