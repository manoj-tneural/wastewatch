"""
WasteWatch - roadside littering detection engine.

Pipeline (per frame):
  1. YOLO + ByteTrack follow every person / bicycle / motorcycle / car / bus / truck.
     For every track we keep its position history and its best (largest, clearest)
     snapshot, so evidence survives after the vehicle has left the frame.
  2. A background model of the watched zone finds NEW objects. The opening scene,
     including bushes and litter already lying there, is learned first and is not
     an event. Vehicles and people are masked out so they are not painted into the
     verge. A later dump in the same spot is a new event once it is no longer part
     of the same throw (different colour, or after `fragment_sec`).
  3. Litter is recorded only when someone actually dumped it. A new object that
     stays still, or a real object that then disappears into bushes, becomes an
     event only if a person or vehicle let go of it. Bushes, wind, and waste that
     was already in the verge are learned into the background instead. We look
     BACK in time (`lookback_sec`) to name the culprit.
  4. A rolling frame buffer lets us build a GIF that starts `pre_roll_sec` before the
     waste appeared (so the throw itself is included) and ends `post_roll_sec` after.
  5. Optional second camera (plate_cam.py): a zoomed view of the exit path. When an
     event is confirmed, the culprit is found again in that view by type, timing and
     colour, and its plate is read on several frames. A confident plate-camera read
     replaces any read from the overview.
"""
from __future__ import annotations

import collections
import json
import math
import os
import threading
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np
from PIL import Image

from .plate_cam import (Calibration, LiveReader, PlateCamMatcher, is_stream, open_plate_source, osd_start,
                        resolve_offset)
from .plates import PlateReader, display_plate

PERSON = 0
VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TWO_WHEELERS = {1, 3}
TRACK_CLASSES = [PERSON] + list(VEHICLES)
CLASS_NAMES = {PERSON: "person", **VEHICLES}


@dataclass
class Config:
    model: str = "yolo11n.pt"          # any ultralytics detection model (yolo11s.pt = more accurate)
    conf: float = 0.30                 # detection confidence
    imgsz: int = 640
    device: str = ""                   # "" auto, "cpu", "0" for GPU
    detect_stride: int = 1             # run YOLO every N frames (2 = ~2x faster)
    zone: list | None = None           # polygon [[x,y],...] normalised 0..1, None = whole frame
    proc_width: int = 640              # resolution of the background model
    diff_thresh: int = 30              # pixel difference to count as "changed"
    min_area_frac: float = 0.00007     # smallest waste blob (fraction of frame); bottles and covers are small
    max_area_frac: float = 0.03        # largest waste blob
    persist_sec: float = 0.8           # object must stay still this long to be confirmed
    warmup_sec: float = 3.0            # median of the opening scene; no events in this time
    baseline_sec: float = 2.0          # then paint bushes and litter already there into the background
    texture_ncc: float = 0.6           # same texture as the background above this = shadow/light change, not waste
    lookback_sec: float = 4.0          # how far back to search for the culprit
    fragment_sec: float = 1.5          # same-spot scraps inside this window count as one throw
    pre_roll_sec: float = 4.0          # GIF starts this long before the waste appeared
    post_roll_sec: float = 1.5         # GIF continues this long after confirmation
    buffer_sec: float = 15.0           # rolling memory of frames
    gif_width: int = 480
    gif_fps: int = 8
    snapshot_width: int = 960          # resolution stored in the frame buffer
    # second camera (number plates). Leave plate_source empty for single-camera mode.
    plate_source: str | None = None    # plate-camera clip path, or rtsp:// URL when live
    plate_offset: float = 0.0          # clips only: seconds the plate clip started AFTER the overview clip
    plate_before_sec: float = 1.0      # margin before the culprit entered the overview
    plate_after_sec: float = 6.0       # ...until this long after the culprit left the overview
    plate_lookback_max_sec: float = 30.0  # search back to when the culprit entered the overview, at most this long
    plate_verge_side: str = "same"     # where the verge is in the plate view: same as overview / left / right / opposite
    plate_order_match: bool = True     # match by vehicle order + lane (False: best single match only)
    live_fps: float = 10.0             # live only: frames per second to process (0 = every camera frame)


# --------------------------------------------------------------------------- helpers
def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def _point_box_dist(px, py, b):
    dx = max(b[0] - px, 0, px - b[2])
    dy = max(b[1] - py, 0, py - b[3])
    return math.hypot(dx, dy)


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _fmt_t(sec):
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _jpeg(img, q=85):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def _unjpeg(buf):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


COLORS = {"vehicle": (255, 170, 40), "person": (120, 220, 120), "culprit": (40, 40, 255),
          "waste": (0, 215, 255), "zone": (255, 120, 255), "cand": (0, 255, 255)}

# How different a new blob must look from waste already recorded in that spot
# before it is treated as another dump rather than a scrap of the same one.
NEW_DUMP_APPEARANCE = 40.0


def appearance_dist(a, b):
    """Distance between two mean HSV colours. Ignores shadow-sized brightness shifts."""
    dh = abs(float(a[0] - b[0]))
    dh = min(dh, 180.0 - dh)
    ds = abs(float(a[1] - b[1]))
    dv = abs(float(a[2] - b[2]))
    hue_w = min(float(a[1]), float(b[1])) / 255.0
    return hue_w * dh * 3.0 + ds * 0.5 + max(0.0, dv - 40.0) * 0.8


def same_spot_suppressed(litter, box, idx, fps, fragment_sec, strength, diff_thresh, appearance):
    """True when `box` is leftover from waste we already reported.

    `litter` entries are (box, frame, mean_hsv). A clearly different colour is a new
    dump even inside the merge window. After the window, only faint flicker is ignored.
    """
    window = fragment_sec * fps
    near_any = False
    near_recent = False
    for lb, fr, _col in litter:
        if not WasteDetector._near_litter(box, lb):
            continue
        near_any = True
        if idx - fr <= window:
            near_recent = True
    if not near_any:
        return False
    if near_recent:
        return appearance is None or appearance < NEW_DUMP_APPEARANCE
    return strength < diff_thresh * 1.35


# --------------------------------------------------------------------------- core
class Track:
    __slots__ = ("tid", "cls_votes", "history", "best_score", "best_crop", "best_frame",
                 "last_seen", "first_seen", "rider", "rider_frame")

    def __init__(self, tid):
        self.tid = tid
        self.cls_votes = collections.Counter()
        self.history = collections.deque(maxlen=2000)   # (frame_idx, evidence_box, conf)
        self.best_score = -1.0
        self.best_crop = None
        self.best_frame = -1
        self.last_seen = -1
        self.first_seen = -1
        self.rider = False
        self.rider_frame = -100000

    def riding_at(self, frame, fps):
        """True only while they are actually on or in the vehicle, not after they step off."""
        return 0 <= frame - self.rider_frame <= int(0.7 * fps)

    @property
    def cls(self):
        return self.cls_votes.most_common(1)[0][0] if self.cls_votes else PERSON

    @property
    def is_vehicle(self):
        return self.cls in VEHICLES

    def window(self, lo, hi):
        """History entries with lo <= frame <= hi, oldest first. Walks back from the newest
        entry and stops at lo, instead of scanning the whole history."""
        out = []
        for item in reversed(self.history):
            if item[0] < lo:
                break
            if item[0] <= hi:
                out.append(item)
        out.reverse()
        return out

    def box_at(self, idx, tol=3):
        best = None
        for f, b, _ in reversed(self.history):
            if f < idx - tol:
                break
            if abs(f - idx) <= tol:
                best = b
        return best


class Candidate:
    _next = 1

    def __init__(self, box, cx, cy, idx):
        self.id = Candidate._next
        Candidate._next += 1
        self.box, self.cx, self.cy = box, cx, cy
        self.sx, self.sy = cx, cy            # where it first appeared, nearer the road
        self.ax, self.ay = cx, cy            # anchor: where it last came to rest
        self.born = idx                      # first frame, not reset when it keeps moving
        self.first = idx
        self.last = idx
        self.hits = 1
        self.seen = 1
        self.confirmed = False
        self.real = False          # passed the new-object test while it was still visible
        self.maxd = 0.0            # furthest it travelled from where it first appeared (proc px)
        self.lost = False          # vanished (bushes, cover) before the still-time was up


class WasteDetector:
    def __init__(self, cfg: Config, out_dir: str, on_event=None, log=print):
        from ultralytics import YOLO

        self.cfg = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.model = YOLO(cfg.model)
        self.on_event = on_event or (lambda e: None)
        self.log = log
        self.tracks: dict[int, Track] = {}
        self.cands: list[Candidate] = []
        self._rider_arm: dict[int, list] = {}
        self._rider_done: set[int] = set()
        self._car_win: dict[int, dict] = {}
        self.events: list[dict] = []
        self.pending_gifs: list[dict] = []
        self.litter: list[tuple] = []          # (box, frame, mean hsv) confirmed waste, proc coords
        self.bg = None
        self.bg_seen = None
        self.preview = None
        self.stop_flag = False
        self.plates = PlateReader()
        self.plates.start()
        self.live = False
        self.stamps = collections.deque(maxlen=25 * 180)    # (frame idx, wall-clock) in live mode
        self.pc_src = None                                    # plate camera source
        self.platecam = None                                  # PlateCamMatcher
        self._pc_pool = None
        self._json_lock = threading.Lock()
        self.clock0 = None                                    # clips: time of day of the first frame (camera clock)

    # ------------------------------------------------------------------ setup
    def _setup(self, w, h, fps):
        c = self.cfg
        self.W, self.H, self.fps = w, h, fps
        self.ps = c.proc_width / w                       # full -> proc scale
        self.pw, self.ph = c.proc_width, int(round(h * self.ps))
        self.ss = min(1.0, c.snapshot_width / w)         # full -> snapshot scale
        self.zone_mask = np.zeros((self.ph, self.pw), np.uint8)
        if c.zone and len(c.zone) >= 3:
            pts = np.array([[x * self.pw, y * self.ph] for x, y in c.zone], np.int32)
            cv2.fillPoly(self.zone_mask, [pts], 255)
            self.zone_full = np.array([[x * w, y * h] for x, y in c.zone], np.int32)
        else:
            self.zone_mask[:] = 255
            self.zone_full = None
        area = self.pw * self.ph
        self.min_area = max(10, c.min_area_frac * area)
        self.max_area = c.max_area_frac * area
        self.persist_n = max(3, int(c.persist_sec * fps))
        self.warmup_n = int(c.warmup_sec * fps)
        self.baseline_n = self.warmup_n + max(1, int(c.baseline_sec * fps))
        self._warm = []
        self.buffer = collections.deque(maxlen=int(c.buffer_sec * fps))
        self.k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))

    # ------------------------------------------------------------------ detection
    def _detect(self, frame, idx):
        c = self.cfg
        if idx % c.detect_stride != 0 and hasattr(self, "_last_dets"):
            return self._last_dets
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=c.conf,
                             imgsz=c.imgsz, classes=TRACK_CLASSES, verbose=False,
                             device=c.device or None)[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(cls)
            for b, k, cf, t in zip(xyxy, cls, conf, ids):
                dets.append({"tid": int(t), "cls": int(k), "conf": float(cf), "box": b.tolist()})
        self._last_dets = dets
        return dets

    def _update_tracks(self, frame, dets, idx):
        persons = [d for d in dets if d["cls"] == PERSON]
        vehicles = [d for d in dets if d["cls"] in VEHICLES]
        # riders: a person sitting on a two-wheeler joins the vehicle's evidence box
        for v in vehicles:
            v["ebox"] = list(v["box"])
            if v["cls"] in TWO_WHEELERS:
                vb = v["box"]
                for p in persons:
                    pb = p["box"]
                    pcx = (pb[0] + pb[2]) / 2
                    if vb[0] - 0.1 * (vb[2] - vb[0]) <= pcx <= vb[2] + 0.1 * (vb[2] - vb[0]) \
                            and pb[3] > vb[1] and pb[1] < vb[1] + 0.3 * (vb[3] - vb[1]):
                        v["ebox"] = _union(v["ebox"], pb)
                        p["rider"] = True
            elif v["cls"] in (2, 5, 7):
                for p in persons:
                    if _iou(p["box"], v["box"]) > 0 and \
                            (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]) < \
                            0.6 * (v["box"][2] - v["box"][0]) * (v["box"][3] - v["box"][1]):
                        p["rider"] = True   # occupant visible through window
        for p in persons:
            p["ebox"] = list(p["box"])

        for d in dets:
            if d["tid"] < 0:
                continue
            t = self.tracks.get(d["tid"])
            if t is None:
                t = self.tracks[d["tid"]] = Track(d["tid"])
            t.cls_votes[d["cls"]] += 1
            if t.first_seen < 0:
                t.first_seen = idx
            t.last_seen = idx
            t.rider = bool(d.get("rider"))
            if t.rider:
                t.rider_frame = idx
            eb = d["ebox"]
            t.history.append((idx, eb, d["conf"]))
            # best snapshot: big, confident, not cut by the frame border
            bw, bh = eb[2] - eb[0], eb[3] - eb[1]
            border = eb[0] < 4 or eb[1] < 4 or eb[2] > self.W - 4 or eb[3] > self.H - 4
            score = bw * bh * d["conf"] * (0.35 if border else 1.0)
            if score > t.best_score and (idx - t.best_frame >= 2 or t.best_frame < 0):
                pad = 0.08
                x1 = int(max(0, eb[0] - pad * bw)); y1 = int(max(0, eb[1] - pad * bh))
                x2 = int(min(self.W, eb[2] + pad * bw)); y2 = int(min(self.H, eb[3] + pad * bh))
                if x2 - x1 > 8 and y2 - y1 > 8:
                    t.best_score = score
                    t.best_crop = _jpeg(frame[y1:y2, x1:x2], 92)
                    t.best_frame = idx

    # ------------------------------------------------------------------ background / waste
    def _object_mask(self, dets):
        m = np.zeros((self.ph, self.pw), np.uint8)
        for d in dets:
            b = d.get("ebox", d["box"])
            x1, y1, x2, y2 = [v * self.ps for v in b]
            w, h = x2 - x1, y2 - y1
            # Cover the body and its shadow. Keep the side margin small so a bag
            # that has just left the rider is still visible.
            x1 -= 0.12 * w; x2 += 0.12 * w; y1 -= 0.10 * h; y2 += 0.22 * h
            cv2.rectangle(m, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
        return m

    def _enhance_litter(self, small):
        """Sharpen and lift local contrast before the litter check.

        A bag against the verge is only a few pixels wide on this camera. CLAHE
        on the brightness channel, then a light unsharp mask, makes that edge
        visible without changing the photo saved as evidence.
        """
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        soft = cv2.GaussianBlur(out, (0, 0), 0.8)
        return cv2.addWeighted(out, 1.35, soft, -0.35, 0)

    def _waste_step(self, small, dets, idx):
        c = self.cfg
        small = self._enhance_litter(small)
        blur = cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)
        objm = self._object_mask(dets)
        free = objm == 0
        self._objm, self._blur = objm, blur
        self._fgm = None
        if self.bg is None:
            # Warm-up: median of the verge where no vehicle is covering it. A bus in the
            # opening frames is not the background, and litter already lying there is.
            if idx < self.warmup_n:
                if idx % 2 == 0:
                    self._warm.append((blur, free.copy()))
                return []
            samples = self._warm + [(blur, free.copy())]
            self._warm = []
            if len(samples) < 3:
                self.bg = blur.copy()
                self.bg_seen = free.copy()
            else:
                acc = np.stack([b for b, _ in samples]).astype(np.float32)
                masks = np.stack([m for _, m in samples])
                seen = masks.any(axis=0)
                filled = np.where(masks[..., None], acc, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(filled, axis=0)
                bg = np.zeros_like(acc[0])
                bg[seen] = np.nan_to_num(med, nan=0.0)[seen]
                self.bg = bg.astype(np.float32)
                self.bg_seen = seen
            return []
        # Opening scene: bushes and litter already in the verge become the background.
        # A vehicle is masked, so it is not painted onto the shoulder. Ground that was
        # hidden for the whole opening is learned the first time it is actually seen.
        if idx < self.baseline_n:
            fresh = free & ~self.bg_seen
            if fresh.any():
                self.bg[fresh] = blur[fresh]
                self.bg_seen |= fresh
            learn = free & self.bg_seen
            self.bg[learn] += 0.5 * (blur[learn] - self.bg[learn])
            if idx + 1 == self.baseline_n:
                self.log("[baseline] opening scene learned; litter already there is ignored")
            return []
        # first clean look at pixels that were hidden behind a vehicle at start
        newly = free & ~self.bg_seen
        if newly.any():
            self.bg[newly] = blur[newly]
            self.bg_seen |= newly

        diff = np.abs(blur - self.bg).max(axis=2)
        fg = (diff > c.diff_thresh)

        # shadow suppression: darker but same colour
        hsv_c = cv2.cvtColor(blur.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv_b = cv2.cvtColor(self.bg.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int16)
        ratio = (hsv_c[..., 2] + 1) / (hsv_b[..., 2] + 1)
        dh = np.abs(hsv_c[..., 0] - hsv_b[..., 0]); dh = np.minimum(dh, 180 - dh)
        shadow = (ratio > 0.25) & (ratio < 0.90) & (np.abs(hsv_c[..., 1] - hsv_b[..., 1]) < 45) & (dh < 18)
        fg &= ~shadow

        # background learning: fast where unchanged, very slow where changed, none under objects
        upd_fast = free & ~fg
        upd_slow = free & fg
        self.bg[upd_fast] += 0.04 * (blur[upd_fast] - self.bg[upd_fast])
        self.bg[upd_slow] += 0.0015 * (blur[upd_slow] - self.bg[upd_slow])

        fgm = (fg & free).astype(np.uint8) * 255
        fgm &= self.zone_mask
        fgm = cv2.morphologyEx(fgm, cv2.MORPH_OPEN, self.k3)
        fgm = cv2.morphologyEx(fgm, cv2.MORPH_CLOSE, self.k7)
        n, _, stats, cents = cv2.connectedComponentsWithStats(fgm, connectivity=8)
        blobs = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < self.min_area or a > self.max_area:
                continue
            if w > 0.35 * self.pw or h > 0.35 * self.ph:
                continue
            blobs.append(([float(x), float(y), float(x + w), float(y + h)], cents[i][0], cents[i][1]))
        self._objm = objm
        self._blur = blur
        self._fgm = fgm
        return blobs

    def _is_real_object(self, box, blur):
        """Edge test: a new object adds edges that the background didn't have.
        A 'ghost' (something that left) has edges in the background instead."""
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = 3
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(self.pw, x2 + pad), min(self.ph, y2 + pad)
        cur = cv2.cvtColor(blur[y1:y2, x1:x2].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(self.bg[y1:y2, x1:x2].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        g = lambda im: np.hypot(cv2.Sobel(im, cv2.CV_32F, 1, 0), cv2.Sobel(im, cv2.CV_32F, 0, 1)).mean()
        if g(cur) < 0.85 * g(bg):
            return False
        return not self._is_light_change(box, blur)

    def _is_light_change(self, box, blur, pad=2):
        """True when the 'new object' is the same ground under different light.

        A shadow moving off gravel, or sun on existing litter, makes the patch brighter
        or darker by roughly the same factor everywhere and keeps its texture. Real waste
        changes the pattern itself.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(self.pw, x2 + pad), min(self.ph, y2 + pad)
        cur = blur[y1:y2, x1:x2].astype(np.float32) + 4.0
        bg = self.bg[y1:y2, x1:x2].astype(np.float32) + 4.0
        if cur.size < 3 * 9:
            return False
        # judge the changed pixels, not the unchanged grass/gravel around them
        fgm = getattr(self, "_fgm", None)
        m = fgm[y1:y2, x1:x2] > 0 if fgm is not None else None
        fill = float(m.mean()) if m is not None and m.shape == cur.shape[:2] else 1.0
        sel = m if m is not None and m.shape == cur.shape[:2] and int(m.sum()) >= 4 else np.ones(cur.shape[:2], bool)
        ratio = (cur / bg)[sel]
        per_ch = ratio.reshape(-1, 3).mean(axis=0)
        gain = float(per_ch.mean())
        spread = float(per_ch.max() - per_ch.min()) / max(gain, 1e-3)    # colour shift
        cv = float(ratio.std()) / max(gain, 1e-3)                       # uneven change
        if abs(gain - 1.0) > 0.08 and spread < 0.10 and cv < 0.14:
            return True
        # texture check on patches big enough to have texture
        if (x2 - x1) >= 6 and (y2 - y1) >= 6 and fill >= 0.5:
            a = cv2.cvtColor(np.clip(cur - 4, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            b = cv2.cvtColor(np.clip(bg - 4, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            a -= a.mean(); b -= b.mean()
            den = float(np.sqrt((a * a).sum() * (b * b).sum()))
            if den > 1e-3 and float((a * b).sum()) / den > self.cfg.texture_ncc and spread < 0.15:
                return True
        return False

    def _update_candidates(self, blobs, idx):
        objm = self._objm
        matched = set()
        for box, cx, cy in blobs:
            diag = np.hypot(box[2] - box[0], box[3] - box[1])
            # A thrown bag moves much faster than a bag sitting in the verge.
            reach = max(12.0, 0.55 * diag, 0.07 * self.pw)
            best, bd = None, 1e9
            for cd in self.cands:
                d = np.hypot(cd.cx - cx, cd.cy - cy)
                if d < reach and d < bd:
                    best, bd = cd, d
            if best is None:
                if self._is_grass_flutter(box) or self._is_known_fragment(box, idx):
                    self._absorb(box)
                    continue
                self.cands.append(Candidate(box, cx, cy, idx))
            else:
                if best.id in matched:
                    continue
                matched.add(best.id)
                drift = np.hypot(cx - best.ax, cy - best.ay)
                if drift > max(4, 0.3 * diag):        # still falling into the bush
                    best.hits = 1
                    best.ax, best.ay = cx, cy
                else:
                    best.hits += 1
                best.seen += 1
                best.maxd = max(best.maxd, float(np.hypot(cx - best.sx, cy - best.sy)))
                best.box, best.cx, best.cy, best.last = box, cx, cy, idx
                # Latch on any clear look. Do not require it to sit still first:
                # a throw into bushes is gone before persist_sec.
                if not self._is_grass_flutter(best.box) and (
                        self._is_real_object(best.box, self._blur)
                        or (best.seen >= max(3, int(0.12 * self.fps))
                            and not self._is_light_change(best.box, self._blur))):
                    best.real = True

        keep, confirmed = [], []
        for cd in self.cands:
            if cd.confirmed:
                continue
            gap = idx - cd.last
            hidden = objm[int(min(cd.cy, self.ph - 1)), int(min(cd.cx, self.pw - 1))] > 0
            # Someone pouring often stands over the spot. Keep the candidate while they do.
            # Someone who gets out and stands over the bag can hide it for a while.
            # A bag that was a real new object, then vanished into bushes, is still a throw
            # when a person or vehicle was beside that spot.
            gone = gap > int(0.35 * self.fps) and not hidden
            covered_too_long = hidden and gap > int(18.0 * self.fps)
            # A throw moves. Wind in a bush, and litter that was already there, do not.
            span = max(float(np.hypot(cd.cx - cd.sx, cd.cy - cd.sy)), cd.maxd)
            diag = float(np.hypot(cd.box[2] - cd.box[0], cd.box[3] - cd.box[1]))
            travelled = span >= max(8.0, 0.45 * diag)
            flew = span >= max(16.0, 0.035 * self.pw)
            culprit = self._find_culprit(cd) if cd.real and (travelled or cd.hits >= self.persist_n) else None
            close = culprit is not None and culprit.get("score", 9) < 0.65
            if cd.real and travelled and (gone or covered_too_long) and culprit is not None:
                # A bag that clearly left someone is still a throw when it lands in grass.
                # A bush that only shivers beside a passing bus is not.
                if flew or (close and not self._is_grass_flutter(cd.box)):
                    cd.lost = True
                    cd.confirmed = True
                    confirmed.append(cd)
                else:
                    self._absorb(cd.box)
                continue
            if gap > ((18.0 if hidden else 0.5) * self.fps):
                if not hidden:
                    self._absorb(cd.box)
                continue
            if cd.last == idx and cd.hits >= self.persist_n:
                if self._touches_object(cd.box):
                    keep.append(cd)           # wait until vehicle/person (and its shadow) moves clear
                    continue
                if self._is_grass_flutter(cd.box):
                    self._absorb(cd.box)
                    continue
                # Still for the time the user asked, and no longer on a person or vehicle.
                # The rider has often already moved on by the time the kit sits.
                real = self._is_real_object(cd.box, self._blur) and not self._is_grass_flutter(cd.box)
                moved = False
                if culprit is not None:
                    lo = getattr(cd, "born", cd.first) - int(self.cfg.lookback_sec * self.fps)
                    moved = self._track_moved(culprit["track"], lo, idx)
                if real and culprit is not None and moved and (close or culprit.get("score", 9) < 1.1):
                    cd.confirmed = True
                    confirmed.append(cd)
                else:
                    self._absorb(cd.box)
                continue
            keep.append(cd)
        self.cands = keep
        return confirmed

    def _touches_object(self, box, margin=6):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(self.pw, x2 + margin), min(self.ph, y2 + margin)
        return bool(self._objm[y1:y2, x1:x2].any())

    @staticmethod
    def _near_litter(box, lb):
        size = max(lb[2] - lb[0], lb[3] - lb[1], box[2] - box[0], box[3] - box[1])
        gap = max(lb[0] - box[2], box[0] - lb[2], lb[1] - box[3], box[1] - lb[3], 0)
        return gap < 0.8 * size + 6

    def _fg_in(self, box):
        """Changed pixels inside `box` (the object itself, not the grass around it), or None."""
        fgm = getattr(self, "_fgm", None)
        if fgm is None:
            return None
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.pw, x2), min(self.ph, y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        m = fgm[y1:y2, x1:x2] > 0
        return m if int(m.sum()) >= 4 else None

    def _mean_hsv(self, box, fg_only=False):
        if self._blur is None:
            return None
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.pw, x2), min(self.ph, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        patch = np.clip(self._blur[y1:y2, x1:x2], 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
        m = self._fg_in(box) if fg_only else None
        if m is not None and m.shape == hsv.shape[:2]:
            return hsv[m].mean(axis=0)
        return hsv.reshape(-1, 3).mean(axis=0)

    def _change_strength(self, box):
        if self.bg is None or self._blur is None:
            return 0.0
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.pw, x2), min(self.ph, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        diff = np.abs(self._blur[y1:y2, x1:x2] - self.bg[y1:y2, x1:x2]).max(axis=2)
        return float(diff.mean()) if diff.size else 0.0

    def _appearance_against_litter(self, box):
        col = self._mean_hsv(box)
        if col is None:
            return None
        best = None
        for lb, _fr, lc in self.litter:
            if lc is None or not self._near_litter(box, lb):
                continue
            d = appearance_dist(col, lc)
            if best is None or d < best:
                best = d
        return best

    def _is_known_fragment(self, box, idx):
        return same_spot_suppressed(
            self.litter, box, idx, self.fps, self.cfg.fragment_sec,
            self._change_strength(box), self.cfg.diff_thresh, self._appearance_against_litter(box))

    def _ring_hsv(self, box, pad=8):
        if self._blur is None:
            return None
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
        ox2, oy2 = min(self.pw, x2 + pad), min(self.ph, y2 + pad)
        if ox2 - ox1 < 4 or oy2 - oy1 < 4:
            return None
        ring = np.clip(self._blur[oy1:oy2, ox1:ox2], 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV)
        mask = np.ones(hsv.shape[:2], np.uint8)
        mask[max(0, y1 - oy1):max(0, y2 - oy1), max(0, x1 - ox1):max(0, x2 - ox1)] = 0
        if int(mask.sum()) < 8:
            return None
        return hsv[mask > 0].astype(np.float32).mean(axis=0)

    def _is_grass_flutter(self, box):
        """Wind in the verge. A bag, bottle or cover is a different colour from the ground around it.
        Judged on the changed pixels only: a small white bag in the grass is not grass."""
        col = self._mean_hsv(box, fg_only=True)
        if col is None:
            return False
        hue, sat = float(col[0]), float(col[1])
        # Only plant-coloured patches. A bag can share the ground's brightness
        # and still be waste, so colour alone is not enough.
        if not (32 <= hue <= 98 and sat >= 28):
            return False
        ring = self._ring_hsv(box)
        if ring is None:
            return sat >= 45
        return appearance_dist(col, ring) < 30

    def _remember_litter(self, box, frame):
        self.litter.append((list(box), frame, self._mean_hsv(box)))

    def _absorb(self, box, pad=4):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(self.pw, x2 + pad), min(self.ph, y2 + pad)
        self.bg[y1:y2, x1:x2] = self._blur[y1:y2, x1:x2]

    # ------------------------------------------------------------------ attribution
    def _was_thrown(self, cd: Candidate):
        """True when a person or vehicle let go of this object.

        The object has to start beside them. A bus driving past the verge is not
        a dump, and neither is litter that was already lying there.
        """
        f0 = getattr(cd, "born", cd.first)
        lo = f0 - int(self.cfg.lookback_sec * self.fps)
        hi = f0 + int(0.8 * self.fps)
        sx, sy = cd.sx / self.ps, cd.sy / self.ps
        net = float(np.hypot(cd.cx - cd.sx, cd.cy - cd.sy))
        flew = max(net, cd.maxd) >= max(14.0, 0.04 * self.pw)
        need = max(3, int(0.25 * self.fps))
        for t in self.tracks.values():
            beside = leaving = 0
            for f, b, _ in t.window(lo, hi):
                bw, bh = max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)
                if (b[0] - 0.45 * bw) <= sx <= (b[2] + 0.45 * bw) and (b[1] - 0.35 * bh) <= sy <= (b[3] + 0.55 * bh):
                    beside += 1
                # still against the vehicle: this is where a thrown bag first appears
                if (b[0] - 0.15 * bw) <= sx <= (b[2] + 0.15 * bw) and (b[1] - 0.15 * bh) <= sy <= (b[3] + 0.25 * bh):
                    leaving += 1
            if beside <= 0 and leaving <= 0:
                continue
            if t.is_vehicle:
                if self._track_moved(t, lo, hi):
                    if flew and leaving >= 1:
                        return True
                elif beside >= 2:
                    return True          # stopped vehicle, someone put it down beside it
            elif self._track_moved(t, lo, hi, 0.10):
                if flew and beside >= 2:
                    return True          # thrown while they were walking
            elif beside >= need:
                return True              # they stopped and put it down
        return False

    def _zone_side(self, box):
        """-1 when the watched zone is to the left of the vehicle, +1 when it is to the right."""
        if getattr(self, "zone_full", None) is None:
            return -1
        zc = float(self.zone_full[:, 0].mean())
        return -1 if zc < (box[0] + box[2]) / 2 else 1

    def _gap_to_zone(self, box, side):
        """Horizontal gap, in full-frame pixels, from the vehicle's zone side to the verge."""
        y1 = int(max(0, box[1] * self.ps))
        y2 = int(min(self.ph, box[3] * self.ps))
        if y2 - y1 < 2:
            return 1e9
        cols = np.where(self.zone_mask[y1:y2].any(axis=0))[0]
        if len(cols) == 0:
            return 1e9
        if side < 0:
            edge = box[0] * self.ps
            nearer = cols[cols <= edge + 2]
            col = float(nearer.max()) if len(nearer) else float(cols.max())
            return max(0.0, edge - col) / max(self.ps, 1e-6)
        edge = box[2] * self.ps
        nearer = cols[cols >= edge - 2]
        col = float(nearer.min()) if len(nearer) else float(cols.min())
        return max(0.0, col - edge) / max(self.ps, 1e-6)

    def _watch_car_window(self, dets, idx):
        """A bag pushed out of a car window toward the verge, then let go.

        The bag stays inside the car until the arm reaches past the door, so the
        verge never sees it land. The reach past the door is the throw.
        """
        confirmed = []
        hsv = None
        for d in dets:
            if d.get("cls") != 2 or d.get("tid", -1) < 0:
                continue
            tid = d["tid"]
            if tid in self._rider_done:
                continue
            box = d["box"]
            bw, bh = box[2] - box[0], box[3] - box[1]
            # A window is only readable on a car this close. A distant van beside the hedge is not.
            if bw < 160 or bh < 120 or box[0] < 4 or box[2] > self.W - 4 or box[1] < 4:
                continue
            side = self._zone_side(box)
            if self._gap_to_zone(box, side) > 1.6 * bw:
                continue
            # The open window, just outside the door. The arm in this view sits high
            # on the door and is only a few frames wide, so the strip has to cover it.
            if side < 0:
                rx1, rx2 = box[0] - 0.22 * bw, box[0]
            else:
                rx1, rx2 = box[2], box[2] + 0.22 * bw
            ry1, ry2 = box[1], box[1] + 0.50 * bh
            ix1, iy1 = int(max(0, rx1)), int(max(0, ry1))
            ix2, iy2 = int(min(self.W - 1, rx2)), int(min(self.H - 1, ry2))
            st = self._car_win.setdefault(tid, {"n": 0, "blob": None, "last": -1, "first": idx, "car": list(box)})
            blob = None
            if ix2 - ix1 >= 8 and iy2 - iy1 >= 8:
                if hsv is None:
                    hsv = cv2.cvtColor(self._frame, cv2.COLOR_BGR2HSV)
                sub = hsv[iy1:iy2, ix1:ix2]
                mask = ((sub[:, :, 0] >= 35) & (sub[:, :, 0] <= 95) & (sub[:, :, 1] >= 70) & (sub[:, :, 2] >= 50)).astype(np.uint8) * 255
                n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
                best = None
                for k in range(1, n):
                    x, y, w, h, a = stats[k]
                    if a < 40 or a > 600 or w > 0.22 * bw or h < 8 or h > 0.40 * bh:
                        continue
                    if best is None or a > best[0]:
                        best = (int(a), int(x), int(y), int(w), int(h))
                if best:
                    _a, x, y, w, h = best
                    # The arm has to come out of the door. A bush with a gap in front of it does not.
                    if side < 0:
                        touches = (ix1 + x + w) >= box[0] - 0.12 * bw
                    else:
                        touches = (ix1 + x) <= box[2] + 0.12 * bw
                    if touches:
                        blob = (ix1 + x, iy1 + y, ix1 + x + w, iy1 + y + h)
            if blob is not None:
                if st["n"] == 0:
                    st["first"] = idx
                st["n"] += 1
                st["blob"] = blob
                st["last"] = idx
                st["car"] = list(box)
                continue
            if st["n"] < 2 or st["last"] < 0 or idx - st["last"] < max(3, int(0.2 * self.fps)):
                continue
            # A verge the car drives along stays green the whole way. The arm is out for a moment.
            if st["last"] - st["first"] > int(0.40 * self.fps):
                st["n"] = 0
                continue
            t = self.tracks.get(tid)
            if t is None or not self._track_moved(t, max(0, idx - int(self.fps)), idx):
                st["n"] = 0
                continue
            x1, y1, x2, y2 = st["blob"]
            pb = [v * self.ps for v in (x1, y1, x2, y2)]
            cd = Candidate(pb, (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2, st["last"])
            cd.born = st["last"]
            cd.last = st["last"]
            cd.force_tid = tid
            cd.force_box = st["car"]
            cd.lost = False
            self._rider_done.add(tid)
            confirmed.append(cd)
            self.log(f"[car] bag leaves the window of car #{tid} at {self._tstr(st['last'])}")
        return confirmed

    def _watch_rider_throws(self, dets, idx):
        """A bag held out from a scooter toward the verge.

        At night the headlight lights the whole shoulder, so the bag never becomes
        its own blob. What shows is a pale shape that appears on the verge side of
        the rider and was not there when the scooter first came alongside.
        """
        if self.bg is None or idx < getattr(self, "baseline_n", 0) or self._frame is None:
            return []
        # Daylight asphalt is pale, so this night test would mark every scooter.
        # A coloured kit still on the rider is not litter either.
        if float(self._frame.mean()) > 100:
            return self._watch_car_window(dets, idx)
        confirmed = []
        hsv = None
        for d in dets:
            if d.get("cls") not in TWO_WHEELERS or d.get("tid", -1) < 0:
                continue
            tid = d["tid"]
            if tid in self._rider_done:
                continue
            box = d.get("ebox", d["box"])
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < 70 or bh < 40:
                continue
            side = self._zone_side(box)
            gap = self._gap_to_zone(box, side)
            # Sample while the scooter is still coming alongside. The bag is out
            # before the bike is tight against the verge.
            if gap > 2.8 * bw:
                continue
            reach = 0.45 * bw
            if side < 0:
                rx1, rx2 = box[0] - reach, box[0] + 0.15 * bw
            else:
                rx1, rx2 = box[2] - 0.15 * bw, box[2] + reach
            ry1, ry2 = box[1], box[1] + 0.62 * bh
            ix1, iy1 = int(max(0, rx1)), int(max(0, ry1))
            ix2, iy2 = int(min(self.W, rx2)), int(min(self.H, ry2))
            if ix2 - ix1 < 12 or iy2 - iy1 < 12:
                continue
            if hsv is None:
                hsv = cv2.cvtColor(self._frame, cv2.COLOR_BGR2HSV)
            patch = hsv[iy1:iy2, ix1:ix2]
            pale = (patch[:, :, 2] > 155) & (patch[:, :, 1] < 110)
            frac = float(pale.mean()) if pale.size else 0.0
            owner = tid
            for ot, prev in self._rider_arm.items():
                if ot == tid or not prev or ot in self._rider_done:
                    continue
                if idx - prev[-1][0] > int(0.7 * self.fps):
                    continue
                if _iou(box, prev[-1][3]) > 0.15:
                    owner = ot
                    break
            if owner in self._rider_done:
                continue
            seq = self._rider_arm.setdefault(owner, [])
            seq.append((idx, frac, (ix1, iy1, ix2, iy2), list(box)))
            if len(seq) > 40:
                del seq[:-40]
            if len(seq) < 4:
                continue
            early = min(s[1] for s in seq[:3])
            usable = [s for s in seq if s[1] <= 0.80]
            if not usable:
                continue
            peak_i, peak_f, peak_b, _peak_box = max(usable, key=lambda s: s[1])
            if early > 0.40 or peak_f < 0.45 or peak_f < early + 0.14:
                continue
            if peak_i < seq[0][0] + int(0.15 * self.fps):
                continue
            t = self.tracks.get(tid)
            if t is None or not self._track_moved(t, seq[0][0], idx):
                continue
            # Crop the verge side of the rider, where the bag is held out.
            vx1, vy1, vx2, vy2 = _peak_box
            vbw, vbh = vx2 - vx1, vy2 - vy1
            if side < 0:
                fx1, fx2 = vx1 - 0.20 * vbw, vx1 + 0.55 * vbw
            else:
                fx1, fx2 = vx2 - 0.55 * vbw, vx2 + 0.20 * vbw
            fy1, fy2 = vy1, vy1 + 0.75 * vbh
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(self.W, fx2), min(self.H, fy2)
            pb = [fx1 * self.ps, fy1 * self.ps, fx2 * self.ps, fy2 * self.ps]
            cd = Candidate(pb, (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2, peak_i)
            cd.born = peak_i
            cd.last = peak_i
            cd.force_tid = tid
            cd.lost = True
            self._rider_done.add(tid)
            self._rider_done.add(owner)
            confirmed.append(cd)
            self.log(f"[rider] pale object leaves two-wheeler #{tid} toward the verge at {self._tstr(peak_i)}")
        return confirmed

    def _find_culprit(self, cd: Candidate):
        forced = getattr(cd, "force_tid", None)
        if forced is not None and forced in self.tracks:
            t = self.tracks[forced]
            b = getattr(cd, "force_box", None) or t.box_at(cd.last, tol=8) or t.box_at(getattr(cd, "born", cd.first), tol=15)
            if b is not None:
                return {"track": t, "frame": cd.last, "box": b, "score": 0.0}
        f0 = getattr(cd, "born", cd.first)
        lo = f0 - int(self.cfg.lookback_sec * self.fps)
        hi = max(cd.last, f0) + int(0.5 * self.fps)
        # The bag is closest to the rider when it is thrown, and deepest in the
        # bush at the end. Try both spots.
        spots = [(cd.cx / self.ps, cd.cy / self.ps),
                 (getattr(cd, "sx", cd.cx) / self.ps, getattr(cd, "sy", cd.cy) / self.ps)]
        ranked = []
        for t in self.tracks.values():
            best = None
            for f, b, _ in t.window(lo, hi):
                diag = np.hypot(b[2] - b[0], b[3] - b[1])
                bh = max(b[3] - b[1], 1)
                for wx, wy in spots:
                    nd = _point_box_dist(wx, wy, b) / max(diag, 1)
                    beside = (b[1] - 0.45 * bh) <= wy <= (b[3] + 0.45 * bh)
                    if wx < b[0]:
                        hgap = b[0] - wx
                    elif wx > b[2]:
                        hgap = wx - b[2]
                    else:
                        hgap = 0
                    # The verge is deep in this view, so a rider in the near lane
                    # can be well to the side of where the bag finishes.
                    from_lane = beside and hgap < 0.36 * self.W
                    if nd >= 1.8 and not from_lane:
                        continue
                    dt = abs(f - f0) / self.fps
                    if nd < 1.8:
                        score = nd + 0.25 * dt
                    else:
                        score = 0.85 + hgap / max(self.W, 1) + 0.25 * dt
                    if t.is_vehicle:
                        score -= 0.15
                    elif t.riding_at(f, self.fps):
                        score += 0.45
                    if best is None or score < best[0]:
                        best = (score, f, b)
            if best:
                ranked.append((best[0], t, best[1], best[2]))
        if not ranked:
            return None
        ranked.sort(key=lambda r: r[0])
        score, t, f, b = ranked[0]
        # A parked auto beside the verge should not beat the rider who actually passed.
        if not self._track_moved(t, lo, hi):
            for sc, ot, of, ob in ranked[1:]:
                if sc > score + 0.8:
                    break
                if self._track_moved(ot, lo, hi):
                    score, t, f, b = sc, ot, of, ob
                    break
        t, f, b = self._vehicle_for_rider(t, f, b)
        return {"track": t, "frame": f, "box": b, "score": score}

    def _track_moved(self, track, lo, hi, min_frac=0.04):
        boxes = [b for f, b, _ in track.window(lo, hi)]
        if len(boxes) < 2:
            return False
        a, z = boxes[0], boxes[-1]
        shift = np.hypot((a[0] + a[2]) / 2 - (z[0] + z[2]) / 2,
                         (a[1] + a[3]) / 2 - (z[1] + z[3]) / 2)
        return shift > min_frac * self.W

    def _vehicle_for_rider(self, track, frame, box):
        """The plate is on the scooter, even when the person box is the closer one."""
        if track.is_vehicle:
            return track, frame, box
        pb = track.box_at(frame, tol=6) or box
        best = None
        for other in self.tracks.values():
            if other.cls not in TWO_WHEELERS:
                continue
            vb = other.box_at(frame, tol=6)
            if vb is None:
                continue
            pcx = (pb[0] + pb[2]) / 2
            if vb[0] - 0.2 * (vb[2] - vb[0]) <= pcx <= vb[2] + 0.2 * (vb[2] - vb[0]):
                best = other, frame, vb
                break
        return best if best else (track, frame, box)

    # ------------------------------------------------------------------ evidence
    def _is_duplicate(self, cd, culprit):
        """Same culprit, same place, a few seconds apart -> a second fragment of one throw."""
        if culprit is None:
            return False
        for ev in self.events:
            if ev.get("track_id") != culprit["track"].tid:
                continue
            if abs(ev["waste_frame"] - getattr(cd, "born", cd.first)) > 3 * self.fps:
                continue
            wb = [v * self.ps for v in ev["waste_box"]]
            d = np.hypot((wb[0] + wb[2]) / 2 - cd.cx, (wb[1] + wb[3]) / 2 - cd.cy)
            if d < 0.15 * self.pw:
                return True
        return False

    def _make_event(self, cd: Candidate, idx, hidden=False):
        culprit = self._find_culprit(cd)
        if self._is_duplicate(cd, culprit):
            self._remember_litter(cd.box, cd.first)
            self._absorb(cd.box)
            return
        ev_no = len(self.events) + 1
        ev_dir = os.path.join(self.out_dir, f"event_{ev_no:02d}")
        os.makedirs(ev_dir, exist_ok=True)
        wbox_full = [v / self.ps for v in cd.box]
        appeared = getattr(cd, "born", cd.first)
        ev = {
            "id": ev_no, "dir": os.path.basename(ev_dir),
            "detected_frame": idx, "waste_frame": appeared,
            "time": round(appeared / self.fps, 2), "time_str": self._tstr(appeared),
            "clock": self._clock(self._T(appeared)),
            "waste_box": [round(v, 1) for v in wbox_full],
            "culprit_type": "unknown", "track_id": None, "vehicle_class": None,
            "plate_text": None, "plate_norm": None, "plate_conf": None, "plate_status": "pending",
            "hidden": bool(hidden),
            "files": {}, "gif_ready": False,
            "platecam": {"status": "waiting"} if self.cfg.plate_source else None,
        }
        # Prefer the last frame the bag was actually visible. Once bushes cover it,
        # the live frame no longer shows the waste.
        src = self._buffer_frame(cd.last)
        scale = self.ss if src is not None else 1.0
        if src is None:
            src = self._frame
        x1, y1, x2, y2 = [int(v * scale) for v in wbox_full]
        ih, iw = src.shape[:2]
        pw_, ph_ = max(40, x2 - x1), max(40, y2 - y1)
        cx1, cy1 = max(0, x1 - pw_), max(0, y1 - ph_)
        cx2, cy2 = min(iw, x2 + pw_), min(ih, y2 + ph_)
        wcrop = src[cy1:cy2, cx1:cx2].copy()
        if wcrop.size:
            cv2.rectangle(wcrop, (max(0, x1 - cx1), max(0, y1 - cy1)),
                          (min(wcrop.shape[1] - 1, x2 - cx1), min(wcrop.shape[0] - 1, y2 - cy1)),
                          COLORS["waste"], 2)
        cv2.imwrite(os.path.join(ev_dir, "waste.jpg"), wcrop if wcrop.size else src)
        ev["files"]["waste"] = "waste.jpg"

        if culprit:
            t = culprit["track"]
            ev.update(track_id=t.tid, vehicle_class=CLASS_NAMES.get(t.cls, "object"),
                      culprit_type="vehicle" if t.is_vehicle else ("rider" if t.rider else "pedestrian"),
                      culprit_frame=culprit["frame"],
                      culprit_time_str=self._tstr(culprit["frame"]),
                      culprit_clock=self._clock(self._T(culprit["frame"])))
            self._write_snapshot(ev, t)
            self._maybe_plate(ev, t)
            # scene: the moment the culprit was closest to the waste spot
            scene = self._buffer_frame(culprit["frame"])
            if scene is not None:
                s = self.ss
                b = culprit["box"]
                cv2.rectangle(scene, (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)), COLORS["culprit"], 3)
                cv2.rectangle(scene, (int(wbox_full[0] * s), int(wbox_full[1] * s)),
                              (int(wbox_full[2] * s), int(wbox_full[3] * s)), COLORS["waste"], 2)
                self._label(scene, f"{ev['vehicle_class']} #{t.tid}", (int(b[0] * s), int(b[1] * s)), COLORS["culprit"])
                cv2.imwrite(os.path.join(ev_dir, "scene.jpg"), scene)
                ev["files"]["scene"] = "scene.jpg"

        self.events.append(ev)
        self._remember_litter(cd.box, appeared)
        self._absorb(cd.box)
        start = max(0, appeared - int(self.cfg.pre_roll_sec * self.fps))
        if culprit:
            start = min(start, culprit["frame"] - int(1.0 * self.fps))
        self.pending_gifs.append({"ev": ev, "start": max(0, start),
                                  "end": idx + int(self.cfg.post_roll_sec * self.fps),
                                  "wbox": wbox_full, "first": appeared,
                                  "tid": ev["track_id"]})
        self._save_json()
        why = " hidden after it landed" if hidden else ""
        self.log(f"[event {ev_no}] waste at {ev['time_str']} -> {ev['culprit_type']} "
                 f"{ev['vehicle_class'] or ''} track {ev['track_id']}{why}")
        self.on_event(ev)

    def _write_snapshot(self, ev, t: Track):
        if t.best_crop is None:
            return
        p = os.path.join(self.out_dir, ev["dir"], "vehicle.jpg")
        with open(p, "wb") as f:
            f.write(t.best_crop)
        ev["files"]["vehicle"] = "vehicle.jpg"
        ev["snapshot_frame"] = t.best_frame
        ev["snapshot_version"] = ev.get("snapshot_version", 0) + 1

    @staticmethod
    def _platecam_owns(ev):
        """A confident plate-camera read is final. The overview must not overwrite it."""
        return ev.get("plate_from") == "plate_cam" and ev.get("plate_status") == "ok"

    def _maybe_plate(self, ev, track, force=False):
        if track is None or track.best_crop is None or self._platecam_owns(ev):
            return
        if not force and ev.get("plate_frame") == track.best_frame:
            return
        if not force and ev.get("plate_status") == "ok" and track.best_score < ev.get("plate_score", 0) * 1.35:
            return
        try:
            img = _unjpeg(track.best_crop)
            hit = self.plates.read(img)
        except Exception as e:  # noqa: BLE001 - a bad plate read must not drop the event
            self.log(f"plate read failed: {e}")
            if ev.get("plate_status") != "ok":
                ev["plate_status"] = "unreadable"
            return
        ev["plate_frame"] = track.best_frame
        ev["plate_score"] = track.best_score
        if not hit or hit.get("crop") is None:
            if ev.get("plate_status") != "ok":
                ev["plate_text"] = None
                ev["plate_norm"] = None
                ev["plate_conf"] = None
                ev["plate_status"] = self.plates.status if self.plates.status != "ready" else "unreadable"
            return
        # A later, less confident read must not wipe a number already accepted.
        if hit.get("display") or ev.get("plate_status") != "ok":
            cv2.imwrite(os.path.join(self.out_dir, ev["dir"], "plate.jpg"), hit["crop"])
            ev["files"]["plate"] = "plate.jpg"
        if hit.get("display"):
            ev["plate_text"] = hit["display"]
            ev["plate_norm"] = hit["norm"]
            ev["plate_conf"] = hit["conf"]
            ev["plate_status"] = "ok"
            self.log(f"[event {ev['id']}] plate {hit['display']} ({hit['conf']:.0%})")
        elif ev.get("plate_status") != "ok":
            ev["plate_text"] = None
            ev["plate_norm"] = None
            ev["plate_conf"] = None
            ev["plate_status"] = "unreadable"
            self.log(f"[event {ev['id']}] plate photo saved, number not confident")

    def _buffer_frame(self, idx):
        for f, buf in self.buffer:
            if f == idx:
                return _unjpeg(buf)
        return None

    @staticmethod
    def _label(img, text, org, color):
        x, y = org
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y = max(th + 6, y)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
        cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _apply_plate(self, ev, hit, source):
        if not hit or hit.get("crop") is None or self._platecam_owns(ev):
            return False
        if hit.get("display") or ev.get("plate_status") != "ok":
            cv2.imwrite(os.path.join(self.out_dir, ev["dir"], "plate.jpg"), hit["crop"])
            ev["files"]["plate"] = "plate.jpg"
        if hit.get("display"):
            ev["plate_text"] = hit["display"]
            ev["plate_norm"] = hit["norm"]
            ev["plate_conf"] = hit["conf"]
            ev["plate_status"] = "ok"
            ev["plate_from"] = source
            self.log(f"[event {ev['id']}] plate {hit['display']} ({hit['conf']:.0%}) from {source}")
            return True
        if ev.get("plate_status") != "ok":
            ev["plate_text"] = None
            ev["plate_norm"] = None
            ev["plate_conf"] = None
            ev["plate_status"] = "unreadable"
            ev["plate_from"] = source
        return False

    def _scan_plate_clip(self, g):
        """Read the plate from the stored clip, after the vehicle has passed.

        A scooter plate is sharp on only a few frames while it is moving. The
        single live snapshot is often the wrong one, so this looks through the
        frames already kept for the GIF and keeps the read that agrees.
        """
        ev = g["ev"]
        if ev.get("plate_status") == "ok":
            return
        tid = g.get("tid")
        track = self.tracks.get(tid) if tid is not None else None
        if track is None or not track.is_vehicle:
            return
        step = max(1, int(round(self.fps / 6)))
        tol = max(4, self.cfg.detect_stride + 1)
        ranked = []
        for f, buf in self.buffer:
            if f < g["start"] or f > g["end"] or (f - g["start"]) % step:
                continue
            box = track.box_at(f, tol=tol)
            if box is None:
                continue
            img = _unjpeg(buf)
            s = self.ss
            x1, y1 = int(max(0, box[0] * s)), int(max(0, box[1] * s))
            x2, y2 = int(min(img.shape[1], box[2] * s)), int(min(img.shape[0], box[3] * s))
            if x2 - x1 < 48 or y2 - y1 < 36:
                continue
            crop = img[y1:y2, x1:x2]
            lower = crop[int(crop.shape[0] * 0.40):]
            gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            ranked.append((sharp, crop))
        ranked.sort(key=lambda item: -item[0])
        best_photo = None
        try:
            for _sharp, crop in ranked[:8]:
                hit = self.plates.read(crop)
                if not hit:
                    continue
                if hit.get("display"):
                    self._apply_plate(ev, hit, "clip")
                    return
                if best_photo is None and hit.get("crop") is not None:
                    best_photo = hit
        except Exception as e:  # noqa: BLE001 - plate search must not drop the clip
            self.log(f"plate clip read failed: {e}")
            return
        if best_photo is not None and ev.get("plate_status") != "ok":
            self._apply_plate(ev, best_photo, "clip")
            self.log(f"[event {ev['id']}] plate photo from clip, number not confident")

    # ------------------------------------------------------------------ second camera
    def _tstr(self, idx):
        """Clip position (mm:ss.ss) for recorded video, clock time (HH:MM:SS) when live."""
        if not self.live:
            return _fmt_t(idx / self.fps)
        t = self._T(idx)
        return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 100):02d}"

    def _clock(self, t):
        """Time of day for a point on the shared timeline, or None when unknown."""
        if self.live:
            return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 10)}"
        if self.clock0 is None:
            return None
        s = (self.clock0 + t) % 86400
        return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:04.1f}"

    def _T(self, idx):
        """Time of overview frame `idx` on the shared timeline (clip seconds, or wall-clock when live)."""
        if not self.live:
            return idx / self.fps
        if self.stamps:
            first_idx = self.stamps[0][0]
            k = idx - first_idx
            if 0 <= k < len(self.stamps) and self.stamps[k][0] == idx:
                return self.stamps[k][1]
            last_idx, last_t = self.stamps[-1]
            return last_t - (last_idx - idx) / self.fps
        return time.time()

    def _pc_match(self, *args, **kw):
        try:
            return self.platecam.match(*args, **kw)
        except Exception as e:  # noqa: BLE001 - a plate-camera failure must not drop the event
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def _platecam_job(self, g):
        """Start looking for the culprit in the plate camera. Returns a Future, or None if not applicable."""
        if self.pc_src is None or self.platecam is None:
            return None
        ev = g["ev"]
        tid = g.get("tid")
        track = self.tracks.get(tid) if tid is not None else None
        if track is not None and track.is_vehicle:
            want_cls = track.cls
        elif track is not None and ev.get("culprit_type") == "rider":
            want_cls = 3          # rider whose bike was not tracked separately: look for a two-wheeler
        else:
            ev["platecam"] = {"status": "skipped", "reason": "culprit is not a vehicle"}
            return None
        c = self.cfg
        at_spot = ev.get("culprit_frame", g["first"])
        tc = self._T(at_spot)
        tl = self._T(max(track.last_seen, at_spot))
        # The plate camera may watch the road before the dumping spot or after it. Search
        # from when the culprit first appeared in the overview (it may have stopped at the
        # spot for a while) to a few seconds after it left. At most 30 s back.
        tf = self._T(track.first_seen) if track.first_seen >= 0 else tc
        t_from = max(tf, tc - c.plate_lookback_max_sec) - c.plate_before_sec
        args = (self.pc_src, track.best_crop, want_cls,
                t_from, tl + c.plate_after_sec,
                (t_from, tl + 0.5 * c.plate_after_sec),
                os.path.join(self.out_dir, ev["dir"]))
        kw = {"calib": getattr(self, "_calib", None)}
        if c.plate_order_match:
            lo_f = max(track.first_seen if track.first_seen >= 0 else at_spot,
                       at_spot - int(c.plate_lookback_max_sec * self.fps)) - int(c.plate_before_sec * self.fps)
            hi_f = max(track.last_seen, at_spot) + int(c.plate_after_sec * self.fps)
            kw["context"] = self._overview_context(track, want_cls, at_spot, lo_f, hi_f)
            ref_box = track.box_at(at_spot, tol=8) or (track.history[-1][1] if track.history else [0, 0, 1, 1])
            side = self._zone_side(ref_box)
            kw["verge_side"] = {"left": -1, "right": 1, "opposite": -side}.get(c.plate_verge_side, side)
        ev["platecam"] = {"status": "searching"}
        if self._pc_pool is not None:                # live: wait for the camera without blocking detection
            return self._pc_pool.submit(self._pc_match, *args, **kw)
        fut = Future()
        fut.set_result(self._pc_match(*args, **kw))   # clip: the frames are already on disk
        return fut

    def _overview_context(self, culprit, want_cls, at_spot, lo_f, hi_f, limit=14):
        """Every vehicle the overview saw in the search window, for order + lane matching.

        For each: its type, when it passed the dumping spot (closest to the watched zone),
        how far from the verge it was then (in vehicle widths), and its best snapshot.
        """
        rows = []
        # reference row: the middle of the watched zone (or of the frame when there is none)
        zf = getattr(self, "zone_full", None)
        y_ref = float(zf[:, 1].mean()) if zf is not None else self.H * 0.6
        for t in self.tracks.values():
            is_culprit = t is culprit
            if not (t.is_vehicle or is_culprit):
                continue
            hist = [(f, b) for f, b, _ in t.history if lo_f <= f <= hi_f]
            if len(hist) < 3 and not is_culprit:
                continue
            # Every vehicle, the culprit included, is timed the same way: the moment the
            # bottom of its box crosses the row of the dumping spot. Each vehicle crosses
            # it once, whatever lane it is in. (The throw itself can be seconds later.)
            best = None
            for f, b in hist:
                key = abs(b[3] - y_ref)
                if best is None or key < best[0]:
                    best = (key, f, b)
            if best is not None:
                pf, pb = best[1], best[2]
            else:
                pf, pb = at_spot, t.box_at(at_spot, tol=8)
            if pb is None:
                continue
            bw = max(pb[2] - pb[0], 1.0)
            gap = self._gap_to_zone(pb, self._zone_side(pb))
            rows.append({"tid": t.tid, "cls": want_cls if is_culprit else t.cls, "t_pass": self._T(pf),
                         "lane": float(min(gap, 1e6)) / bw, "jpeg": t.best_crop, "culprit": is_culprit})
        tc = self._T(at_spot)
        rows.sort(key=lambda r: (not r["culprit"], abs(r["t_pass"] - tc)))
        return rows[:limit]

    def _apply_platecam(self, ev, res):
        if not res:
            return
        if ev.get("plate_status") == "pending":
            ev["plate_status"] = "unreadable"      # set again below when a number was read
        keep = ("status", "error", "frames", "vehicles_seen", "match_score", "appearance", "time_gap",
                "type_ok", "ambiguous", "vehicle_class", "votes", "seen_from", "seen_to", "frame_time", "secs",
                "method", "overview_vehicles", "learned_delay", "delay_samples")
        ev["platecam"] = {k: res.get(k) for k in keep if res.get(k) is not None}
        for key, fn in (res.get("files") or {}).items():
            ev["files"]["pc_" + key] = fn
        norm = res.get("plate_norm")
        if norm:
            if ev.get("plate_status") == "ok" and ev.get("plate_norm") != norm:
                ev["plate_overview"] = ev.get("plate_text")       # keep the overview read for the record
            ev.update(plate_text=display_plate(norm), plate_norm=norm, plate_conf=res.get("plate_conf"),
                      plate_status="ok", plate_from="plate_cam", plate_votes=res.get("votes"))
            ev["files"]["plate"] = res["files"]["plate"]
            self.log(f"[event {ev['id']}] plate camera: {ev['plate_text']} "
                     f"({res.get('votes')} frames agree, match {res.get('match_score')})")
        elif ev.get("plate_status") != "ok" and (res.get("files") or {}).get("plate"):
            ev["files"]["plate"] = res["files"]["plate"]
            ev["plate_from"] = "plate_cam"
            ev["plate_status"] = "unreadable"
            self.log(f"[event {ev['id']}] plate camera: photo saved, number not confident")
        else:
            self.log(f"[event {ev['id']}] plate camera: {res.get('status')}"
                     f"{' - ' + res['error'] if res.get('error') else ''}")

    def _flush_gifs(self, idx, final=False):
        still = []
        for g in self.pending_gifs:
            if not (idx >= g["end"] or final):
                still.append(g)
                continue
            if "pc" not in g:
                g["pc"] = self._platecam_job(g)
                g["pc_deadline"] = time.time() + 90
            fut = g["pc"]
            if fut is not None and not fut.done() and not final and time.time() < g["pc_deadline"]:
                still.append(g)                      # live: plate camera still waiting for frames
                continue
            tid = g.get("tid")
            if tid is not None and tid in self.tracks:
                self._maybe_plate(g["ev"], self.tracks[tid], force=True)
            self._scan_plate_clip(g)
            if fut is not None:
                try:
                    res = fut.result(timeout=90 if final else 0)
                except Exception as e:  # noqa: BLE001 - timeout or worker error
                    res = {"status": "error", "error": f"plate camera did not answer: {e}"}
                self._apply_platecam(g["ev"], res)
            self._save_json()
            self._write_gif(g)
        self.pending_gifs = still

    def _write_gif(self, g):
        c = self.cfg
        ev = g["ev"]
        step = max(1, int(round(self.fps / c.gif_fps)))
        t = self.tracks.get(g["tid"]) if g["tid"] is not None else None
        frames = []
        for f, buf in list(self.buffer):
            if f < g["start"] or f > g["end"] or (f - g["start"]) % step:
                continue
            img = _unjpeg(buf)
            s = self.ss
            if t is not None:
                b = t.box_at(f)
                if b is not None:
                    cv2.rectangle(img, (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)), COLORS["culprit"], 3)
                    self._label(img, f"{ev['vehicle_class']} #{t.tid}", (int(b[0] * s), int(b[1] * s)), COLORS["culprit"])
            if f >= g["first"]:
                wb = g["wbox"]
                cv2.rectangle(img, (int(wb[0] * s) - 4, int(wb[1] * s) - 4),
                              (int(wb[2] * s) + 4, int(wb[3] * s) + 4), COLORS["waste"], 2)
                self._label(img, "WASTE", (int(wb[0] * s) - 4, int(wb[1] * s) - 4), (0, 140, 200))
            self._label(img, f"t={self._tstr(f)}", (8, 26), (40, 40, 40))
            if ev.get("plate_text"):
                self._label(img, ev["plate_text"], (8, 52), (20, 20, 20))
            gw = c.gif_width
            gh = int(img.shape[0] * gw / img.shape[1])
            img = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if not frames:
            return
        p = os.path.join(self.out_dir, ev["dir"], "action.gif")
        pal = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
        q[0].save(p, save_all=True, append_images=q[1:], duration=int(1000 / c.gif_fps), loop=0, optimize=True)
        ev["files"]["gif"] = "action.gif"
        ev["gif_ready"] = True
        ev["gif_range"] = [self._tstr(g["start"]), self._tstr(g["end"])]
        self._save_json()
        self.on_event(ev)

    def _save_json(self):
        with self._json_lock:
            tmp = os.path.join(self.out_dir, "events.json.tmp")
            with open(tmp, "w") as f:
                json.dump(self.events, f, indent=2)
            os.replace(tmp, os.path.join(self.out_dir, "events.json"))

    # ------------------------------------------------------------------ preview
    def _near_zone(self, box, margin=0.6):
        """True when a box is in or beside the watched zone (or there is no zone)."""
        if self.zone_full is None:
            return True
        bw = box[2] - box[0]
        for px in (box[0] - margin * bw, (box[0] + box[2]) / 2, box[2] + margin * bw):
            if cv2.pointPolygonTest(self.zone_full.astype(np.float32), (float(px), float(box[3])), True) > -0.3 * bw:
                return True
        return False

    def _draw_preview(self, frame, dets):
        """Live view: the zone, the people and vehicles near it, the culprits and the waste.
        Far-away traffic and the background model's candidate blobs are not drawn."""
        img = cv2.resize(frame, (int(self.W * self.ss), int(self.H * self.ss)))
        s = self.ss
        if self.zone_full is not None:
            cv2.polylines(img, [(self.zone_full * s).astype(np.int32)], True, COLORS["zone"], 2)
        # culprits and waste stay on screen for a few seconds after a throw, then go
        recent = int(5 * self.fps)
        idx = getattr(self, "_idx", 0)
        culprits = {ev.get("track_id") for ev in self.events if idx - ev["detected_frame"] <= recent}
        for d in dets:
            b = d.get("ebox", d["box"])
            is_culprit = d["tid"] in culprits
            if not is_culprit and (d["cls"] not in VEHICLES or self.zone_full is None or not self._near_zone(b)):
                continue                       # no people, no traffic away from the litter zone
            col = COLORS["culprit"] if is_culprit else COLORS["vehicle"]
            cv2.rectangle(img, (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)), col, 3 if is_culprit else 1)
            if is_culprit:
                self._label(img, f"CULPRIT {CLASS_NAMES[d['cls']]} #{d['tid']}", (int(b[0] * s), int(b[1] * s)), col)
        k = s / self.ps
        for b, _fr, _col in [x for x in self.litter if idx - x[1] <= recent]:
            cv2.rectangle(img, (int(b[0] * k) - 4, int(b[1] * k) - 4), (int(b[2] * k) + 4, int(b[3] * k) + 4), (0, 0, 255), 2)
            self._label(img, "LITTER", (int(b[0] * k) - 4, int(b[1] * k) - 4), (0, 0, 200))
        return img

    # ------------------------------------------------------------------ main loop
    def _open_platecam(self):
        c = self.cfg
        if not c.plate_source:
            return
        self.pc_src = open_plate_source(c.plate_source, c.plate_offset, log=self.log)
        if self.pc_src.live != self.live:
            self.pc_src.close()
            self.pc_src = None
            raise ValueError("Use two recorded clips or two live cameras for the overview and plate camera, not a mix.")
        self.platecam = PlateCamMatcher(c.model, c.conf, c.imgsz, c.device, log=self.log)
        # learned travel time between the cameras, shared by all jobs of this install
        calib_path = os.path.join(os.path.dirname(os.path.abspath(self.out_dir)), "platecam_calib.json")
        self._calib = Calibration(calib_path, key="live" if self.pc_src.live else "clip", log=self.log)
        dl = self._calib.delay()
        self.log(f"plate camera: travel time {'learned ' + format(dl[0], '+.2f') + 's' if dl else 'not learned yet'}"
                 f" ({self._calib.samples()} samples)")
        if self.pc_src.live:
            self._pc_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="platecam")
            self.log(f"plate camera: live, buffering {self.pc_src.buffer_sec:.0f}s")
        else:
            self.log(f"plate camera: clip {os.path.basename(c.plate_source)}, "
                     f"{self.pc_src.width}x{self.pc_src.height} @ {self.pc_src.fps:.0f} fps, "
                     f"starts {c.plate_offset:+.2f}s after the overview clip")

    def _close_platecam(self):
        if self._pc_pool is not None:
            self._pc_pool.shutdown(wait=False)
        if self.pc_src is not None:
            self.pc_src.close()

    def run(self, video_path, progress=None):
        self.live = is_stream(video_path)
        reader = cap = None
        if self.live:
            reader = LiveReader(video_path, target_fps=self.cfg.live_fps, log=self.log)
            first = reader.read(timeout=20)
            if first is None:
                reader.close()
                raise RuntimeError(f"No frames from camera: {video_path}")
            fps, total = reader.fps, 0
            h, w = first[1].shape[:2]
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            if fps <= 1 or fps > 240:
                fps = 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self._setup(w, h, fps)
        if not self.live:
            try:
                self.clock0 = osd_start(video_path)
            except Exception as e:  # noqa: BLE001 - clock times are a convenience
                self.log(f"could not read the camera clock: {e}")
            if self.clock0 is not None:
                self.log(f"overview clip starts at {self._clock(0.0)} (camera clock)")
        self.stamps = collections.deque(maxlen=int(fps * 180))
        try:
            self._open_platecam()
        except Exception:
            if reader:
                reader.close()
            if cap:
                cap.release()
            raise
        idx, t0 = 0, time.time()
        pending_first = first if self.live else None
        while not self.stop_flag:
            if self.live:
                item, pending_first = (pending_first, None) if pending_first else (reader.read(timeout=5), None)
                if item is None:
                    self._flush_gifs(idx)          # keep finishing evidence while the stream is quiet
                    continue
                stamp, frame = item
                self.stamps.append((idx, stamp))
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            self._frame = frame
            snap = cv2.resize(frame, (int(w * self.ss), int(h * self.ss)), interpolation=cv2.INTER_AREA) \
                if self.ss < 1 else frame
            self.buffer.append((idx, _jpeg(snap, 82)))
            dets = self._detect(frame, idx)
            self._update_tracks(frame, dets, idx)
            small = cv2.resize(frame, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
            blobs = self._waste_step(small, dets, idx)
            if blobs is not None and hasattr(self, "_objm"):
                for cd in self._update_candidates(blobs, idx):
                    self._make_event(cd, idx, hidden=cd.lost)
                for cd in self._watch_rider_throws(dets, idx):
                    self._make_event(cd, idx, hidden=bool(cd.lost))
            # refresh culprit snapshots while the vehicle is still visible (better views)
            for ev in self.events:
                tid = ev.get("track_id")
                if tid is not None and tid in self.tracks:
                    tr = self.tracks[tid]
                    if tr.best_frame > ev.get("snapshot_frame", -1) and tr.last_seen == idx:
                        self._write_snapshot(ev, tr)
                        self._maybe_plate(ev, tr)
                        self._save_json()
                        self.on_event(ev)
            self._flush_gifs(idx)
            if idx % 3 == 0:
                self._idx = idx
                self.preview = _jpeg(self._draw_preview(frame, dets), 75)
            idx += 1
            if progress:
                el = time.time() - t0
                progress(idx, total, idx / max(el, 1e-6))
        if cap:
            cap.release()
        if reader:
            reader.close()
        self._flush_gifs(idx, final=True)
        self._close_platecam()
        self._save_json()
        return self.events


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Detect waste throwing in a video (CLI mode)")
    ap.add_argument("video")
    ap.add_argument("--out", default="outputs/cli")
    ap.add_argument("--zone", help='JSON polygon, normalised, e.g. "[[0.5,0.4],[1,0.4],[1,1],[0.5,1]]"')
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fragment", type=float, default=1.5,
                    help="seconds to merge scraps of one throw in the same spot (0 = every dump is separate)")
    ap.add_argument("--plate", help="plate camera: clip recorded at the same time, or rtsp:// URL")
    ap.add_argument("--plate-offset", type=float,
                    help="seconds the plate clip started after the overview clip (default: matched from the traffic)")
    ap.add_argument("--plate-adjust", type=float, default=0.0, help="extra sync correction in seconds")
    a = ap.parse_args()
    off = a.plate_offset
    if a.plate and off is None and not is_stream(a.plate):
        off, method, score = resolve_offset(a.video, a.plate)
        print(f"plate camera offset: {off:+.2f}s ({method}, match {score})")
    cfg = Config(model=a.model, detect_stride=a.stride, fragment_sec=a.fragment,
                 zone=json.loads(a.zone) if a.zone else None,
                 plate_source=a.plate, plate_offset=(off or 0.0) + a.plate_adjust)
    det = WasteDetector(cfg, a.out)
    evs = det.run(a.video, progress=lambda i, n, f: print(f"\r{i}/{n} frames  {f:.1f} fps", end="") if i % 24 == 0 else None)
    print(f"\n{len(evs)} event(s). Evidence in {a.out}")
