"""Second camera: a narrow, zoomed view used only to read number plates.

The overview camera (detector.py) sees the throw and picks the culprit. This
module then looks at the plate camera over the time the culprit was passing,
finds the same vehicle there, and reads its plate on several frames.

Both cameras share one timeline, in seconds:
  * recorded clips - seconds since the overview clip started. The plate clip is
    shifted by `offset`, which the dashboard works out from the NVR file names
    (VIGI_NVR..._20261024_15.46.28.797.mp4) and you can fine-tune.
  * live RTSP      - wall-clock time (time.time()) stamped as each frame arrives.

Matching the culprit across cameras uses three cues:
  1. vehicle type   - two-wheeler vs car/bus/truck
  2. timing         - the vehicle passes this zoomed view, but the throw can be
                      several seconds earlier or later than that pass
  3. appearance     - colour histogram of the plate-view vehicle against the
                      culprit snapshot from the overview (rider clothes count)
"""
from __future__ import annotations

import collections
import datetime as _dt
import os
import re
import threading
import time

import cv2
import numpy as np

from .plates import PlateReader, edit_distance

PERSON = 0
VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TWO_WHEELERS = {1, 3}

_NVR_TS = re.compile(r"(\d{8})[_-](\d{2})[.\-:](\d{2})[.\-:](\d{2})(?:[.\-:](\d{1,3}))?")


def is_stream(src) -> bool:
    return isinstance(src, str) and src.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))


def parse_nvr_start(name):
    """Start time from an NVR export name such as ..._20261024_15.46.28.797.mp4. None if absent."""
    m = None
    for m in _NVR_TS.finditer(os.path.basename(name or "")):
        pass  # the last match is the clip start; channel names can contain digits too
    if not m:
        return None
    day, hh, mm, ss, ms = m.groups()
    try:
        t = _dt.datetime.strptime(f"{day}{hh}{mm}{ss}", "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return t + _dt.timedelta(milliseconds=int((ms or "0").ljust(3, "0")))


def auto_offset(overview_name, plate_name):
    """Seconds the plate clip started after the overview clip, from the file names. None if unknown."""
    a, b = parse_nvr_start(overview_name), parse_nvr_start(plate_name)
    if a is None or b is None:
        return None
    return round((b - a).total_seconds(), 3)


def _activity(path, rate=10.0, width=160, max_sec=None):
    """Traffic activity per time step: how much the picture changes, sampled `rate` times a second."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fps = fps if 1 < fps < 240 else 25
    prev, out, i, next_t = None, [], 0, 0.0
    limit = int(max_sec * fps) if max_sec else None
    while True:
        if limit is not None and i >= limit:
            break
        if not cap.grab():
            break
        if i / fps + 1e-6 >= next_t:          # sample by time, so clips with different fps line up
            next_t += 1.0 / rate
            ok, f = cap.retrieve()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(f, (width, max(1, int(width * f.shape[0] / f.shape[1])))), cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)
            if prev is not None:
                out.append(float(np.abs(g - prev).mean()))
            prev = g
        i += 1
    cap.release()
    return np.array(out, np.float32), 1.0 / rate


def offset_by_motion(overview_path, plate_path, around=None, search=None, max_sec=900):
    """Seconds the plate clip started after the overview clip, found by matching traffic.

    Both cameras watch the same road, so cars passing make the same pattern of activity
    in both, shifted by the start-time difference. Returns (offset, score); score is the
    correlation (above ~0.35 is a clear match). `around`/`search` limit the search window.
    """
    a, dt = _activity(overview_path, max_sec=max_sec)
    b, dtb = _activity(plate_path, max_sec=max_sec)
    if len(a) < 20 or len(b) < 20 or abs(dt - dtb) > 1e-3:
        return None, 0.0
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    min_overlap = max(int(5.0 / dt), int(0.4 * min(len(a), len(b))))
    best = (None, -1.0)
    scores = {}
    for lag in range(-(len(b) - min_overlap), len(a) - min_overlap + 1):
        off = lag * dt          # plate sample j matches overview sample j + lag -> plate started lag*dt later
        if around is not None and search is not None and abs(off - around) > search:
            continue
        i0, j0 = max(0, lag), max(0, -lag)
        n = min(len(a) - i0, len(b) - j0)
        if n < min_overlap:
            continue
        x, y = a[i0:i0 + n], b[j0:j0 + n]
        sx, sy = x.std(), y.std()
        if sx < 1e-6 or sy < 1e-6:
            continue
        s = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
        scores[lag] = s
        if s > best[1]:
            best = (round(off, 2), s)
    return best[0], round(best[1], 3)


_OSD_OCR = None


def _osd_seconds(frame):
    """Clock burned into the picture (VIGI OSD, top left: 'MM-DD-YYYY Ddd HH:MM:SS'), as
    seconds since midnight, or None. Only the six time digits are cut out and read;
    the font is fixed-width, so their place is known."""
    global _OSD_OCR
    if _OSD_OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OSD_OCR = RapidOCR()
    H, W = frame.shape[:2]
    sx, sy = W / 2560, H / 1440
    x0, pitch, y0, y1 = 66, 31.65, 60, 134
    parts = []
    for i in (15, 16, 18, 19, 21, 22):
        parts.append(frame[int(y0 * sy):int(y1 * sy), int((x0 + i * pitch) * sx):int((x0 + (i + 1) * pitch) * sx)])
        if i in (16, 19):
            parts.append(np.zeros((parts[-1].shape[0], max(1, int(12 * sx)), 3), np.uint8))
    img = np.hstack(parts)
    img = cv2.resize(img, None, fx=48 / img.shape[0], fy=48 / img.shape[0])
    res, _ = _OSD_OCR(img, use_det=False, use_cls=False, use_rec=True)
    d = re.sub(r"\D", "", res[0][0]) if res else ""
    if len(d) != 6:
        return None
    hh, mm, ss = int(d[:2]), int(d[2:4]), int(d[4:])
    if hh > 23 or mm > 59 or ss > 59:
        return None
    return hh * 3600 + mm * 60 + ss


def osd_start(path):
    """Clock time of the first frame of a clip, to a frame, from the burned-in OSD.

    The OSD shows whole seconds, so the clip start is found from the frame where the
    seconds digit first changes. Returns seconds since midnight, or None.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fps = fps if 1 < fps < 240 else 25
    try:
        ok, f0 = cap.read()
        s0 = _osd_seconds(f0) if ok else None
        if s0 is None:
            return None
        prev_i, i = 0, 0
        while i < int(fps * 1.2):
            i = min(prev_i + 2, int(fps * 1.2))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                return None
            s = _osd_seconds(f)
            if s is None:
                prev_i = i
                continue
            if s != s0:
                # the tick is between prev_i and i; check the middle frame
                if i - prev_i == 2:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, prev_i + 1)
                    ok, fm = cap.read()
                    if ok and _osd_seconds(fm) == s:
                        i = prev_i + 1
                if (s - s0) % 86400 != 1:
                    return None
                return s - i / fps
            prev_i = i
        return None
    finally:
        cap.release()


def resolve_offset(overview_path, plate_path, overview_name=None, plate_name=None, log=print):
    """Pick the clip offset: traffic matching first, NVR file names as a fallback.

    NVR export names are not always the recording time (an export made later can carry
    the export time), so the names are only trusted when the pictures cannot decide.
    Returns (offset, method, score).
    """
    by_name = auto_offset(overview_name or overview_path, plate_name or plate_path)
    off, score = offset_by_motion(overview_path, plate_path)
    if off is not None and score >= 0.35:
        if by_name is not None and abs(by_name - off) > 2:
            log(f"sync: file names say {by_name:+.2f}s but the traffic says {off:+.2f}s - using traffic")
        return off, "traffic", score
    if by_name is not None:
        return by_name, "file names", score
    if off is not None and score >= 0.2:
        return off, "traffic (weak)", score
    return 0.0, "none", score


def _jpeg(img, q=90):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def _unjpeg(buf):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _fmt_clock(t, live):
    if live:
        return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 100):02d}"
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:05.2f}"


# --------------------------------------------------------------------------- sources
class FileSource:
    """Plate camera from a recorded clip. `offset` = seconds it started after the overview clip."""

    live = False

    def __init__(self, path, offset=0.0):
        self.path = path
        self.offset = float(offset or 0.0)
        self._lock = threading.Lock()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open plate camera video: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25
        self.fps = fps if 1 < fps < 240 else 25
        self.total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self, t0, t1, step_sec=0.12):
        """[(overview_time, frame)] for t0..t1 on the overview timeline."""
        a = max(0.0, t0 - self.offset)
        b = t1 - self.offset
        if b <= 0:
            return []
        f0 = int(a * self.fps)
        f1 = int(b * self.fps)
        if self.total:
            f1 = min(f1, self.total - 1)
        if f1 < f0:
            return []
        step = max(1, int(round(step_sec * self.fps)))
        out = []
        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            for f in range(f0, f1 + 1):
                ok, frame = self._cap.read()
                if not ok:
                    break
                if (f - f0) % step == 0:
                    out.append((f / self.fps + self.offset, frame))
        return out

    def wait_until(self, t, timeout=0):
        return True

    def close(self):
        with self._lock:
            self._cap.release()


class StreamSource:
    """Plate camera from RTSP. Keeps the last `buffer_sec` seconds in memory, stamped with time.time()."""

    live = True

    def __init__(self, url, buffer_sec=25.0, keep_fps=8.0, log=print):
        self.url = url
        self.buffer_sec = buffer_sec
        self.keep_fps = keep_fps
        self.log = log
        self.fps = 25.0
        self.width = self.height = 0
        self._buf = collections.deque()
        self._lock = threading.Lock()
        self._stop = False
        self.newest = 0.0
        self.error = None
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - not every backend has it
            pass
        return cap

    def _run(self):
        cap, last_keep, fails = self._open(), 0.0, 0
        while not self._stop:
            ok, frame = cap.read()
            now = time.time()
            if not ok:
                fails += 1
                if fails in (1, 30):
                    self.log("plate camera: no frames, reconnecting…")
                time.sleep(1.0)
                cap.release()
                cap = self._open()
                continue
            fails = 0
            self.height, self.width = frame.shape[:2]
            if now - last_keep < 1.0 / self.keep_fps:
                continue
            last_keep = now
            with self._lock:
                self._buf.append((now, _jpeg(frame, 90)))
                self.newest = now
                while self._buf and now - self._buf[0][0] > self.buffer_sec:
                    self._buf.popleft()
        cap.release()

    def wait_until(self, t, timeout=15.0):
        end = time.time() + timeout
        while self.newest < t and time.time() < end and not self._stop:
            time.sleep(0.2)
        return self.newest >= t

    def frames(self, t0, t1, step_sec=0.12):
        with self._lock:
            items = [(t, b) for t, b in self._buf if t0 <= t <= t1]
        out, last = [], -1e9
        for t, b in items:
            if t - last >= step_sec * 0.9:
                out.append((t, _unjpeg(b)))
                last = t
        return out

    def close(self):
        self._stop = True


class LiveReader:
    """Overview camera from RTSP. Reads on its own thread so each frame is stamped when it arrives."""

    def __init__(self, url, max_queue_sec=10.0, target_fps=0.0, log=print):
        self.url = url
        self.log = log
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera stream: {url}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        fps = fps if 1 < fps < 120 else 25
        # Process a steady, known rate instead of dropping frames at random when the PC is slow.
        self.fps = min(fps, target_fps) if target_fps and target_fps > 0 else fps
        self._min_gap = 0.0 if self.fps >= fps else 1.0 / self.fps
        self._last = 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.q = collections.deque(maxlen=int(max_queue_sec * self.fps))
        self.dropped = 0
        self._stop = False
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        fails = 0
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails in (1, 30):
                    self.log("overview camera: no frames, reconnecting…")
                time.sleep(1.0)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue
            fails = 0
            now = time.time()
            if self._min_gap and now - self._last < self._min_gap * 0.95:
                continue
            self._last = now
            with self._cv:
                if len(self.q) == self.q.maxlen:
                    self.dropped += 1
                    if self.dropped in (1, 100) or self.dropped % 1000 == 0:
                        self.log(f"overview camera: processing is slower than the camera, "
                                 f"{self.dropped} frames dropped (lower 'Live processing fps', or use a GPU)")
                self.q.append((now, frame))
                self._cv.notify()

    def read(self, timeout=5.0):
        with self._cv:
            if not self.q:
                self._cv.wait(timeout)
            if not self.q:
                return None
            return self.q.popleft()

    def close(self):
        self._stop = True


def open_plate_source(src, offset=0.0, log=print):
    if not src:
        return None
    if is_stream(src):
        return StreamSource(src, buffer_sec=45.0, log=log)
    return FileSource(src, offset)


# --------------------------------------------------------------------------- matcher
def _hist(bgr):
    """Colour signature of a vehicle. Uses the centre of the box (less road) and keeps
    brightness, so a grey car and a white car do not look alike."""
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    core = bgr[int(h * 0.12):int(h * 0.88), int(w * 0.12):int(w * 0.88)]
    if core.size == 0:
        core = bgr
    small = cv2.resize(core, (96, max(1, int(96 * core.shape[0] / max(core.shape[1], 1)))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def _similar(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    # Bhattacharyya distance: 0 = identical. Turn it into 0..1 similarity.
    return float(max(0.0, 1.0 - cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)))


def _group_of(cls):
    return "2w" if cls in TWO_WHEELERS else "4w"


class _Group:
    """One vehicle followed through the plate-camera frames in the window."""

    def __init__(self, gid):
        self.gid = gid
        self.obs = []                       # (t, frame_no, box, cls, conf)
        self.cls_votes = collections.Counter()

    @property
    def last_box(self):
        return self.obs[-1][2]

    @property
    def last_t(self):
        return self.obs[-1][0]

    @property
    def cls(self):
        return self.cls_votes.most_common(1)[0][0]


class PlateCamMatcher:
    max_reads = 40                     # OCR attempts per event (largest, sharpest views first; OCR is cheap)

    def __init__(self, model="yolo11n.pt", conf=0.30, imgsz=640, device="", log=print):
        from ultralytics import YOLO

        self.model = YOLO(model)       # own instance: never disturbs the overview tracker
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or None
        self.log = log
        self.plates = PlateReader()
        self.plates.start()

    # ---------------------------------------------------------------- detection
    def _detect(self, frame):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False,
                               classes=[PERSON] + list(VEHICLES), device=self.device)[0]
        if r.boxes is None or not len(r.boxes):
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        persons = [b.tolist() for b, k in zip(xyxy, cls) if k == PERSON]
        out = []
        for b, k, c in zip(xyxy, cls, conf):
            if k not in VEHICLES:
                continue
            box = b.tolist()
            if k in TWO_WHEELERS:              # add the rider, as the overview does
                w = box[2] - box[0]
                for p in persons:
                    pcx = (p[0] + p[2]) / 2
                    if box[0] - 0.1 * w <= pcx <= box[2] + 0.1 * w and p[3] > box[1] \
                            and p[1] < box[1] + 0.3 * (box[3] - box[1]):
                        box = _union(box, p)
            out.append((box, int(k), float(c)))
        # one box per vehicle: YOLO often reports a car also as a truck, or a bike twice
        out.sort(key=lambda d: -d[2])
        kept = []
        for d in out:
            if all(_group_of(d[1]) != _group_of(o[1]) or _iou(d[0], o[0]) < 0.6 for o in kept):
                kept.append(d)
        return kept

    @staticmethod
    def _predict(g, t):
        """Where group g should be at time t, assuming it keeps its last speed."""
        lb = g.last_box
        if len(g.obs) < 2:
            return lb
        (ta, _, pa, _, _), (tb, _, pb, _, _) = g.obs[-2], g.obs[-1]
        if tb - ta <= 0:
            return lb
        k = (t - tb) / (tb - ta)
        return [pb[i] + (pb[i] - pa[i]) * k for i in range(4)]

    def _link(self, samples, frame_w):
        """Greedy IoU/centre linking across the sampled frames -> one group per vehicle."""
        groups, nxt = [], 1
        for fno, (t, _frame, dets) in enumerate(samples):
            used = set()
            for box, k, c in sorted(dets, key=lambda d: -(d[0][2] - d[0][0]) * (d[0][3] - d[0][1])):
                best, bs = None, 0.0
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                area = (box[2] - box[0]) * (box[3] - box[1])
                for g in groups:
                    if g.gid in used or t - g.last_t > 0.5 or _group_of(g.cls) != _group_of(k):
                        continue
                    lb = g.last_box
                    ratio = area / max((lb[2] - lb[0]) * (lb[3] - lb[1]), 1.0)
                    if not 0.45 <= ratio <= 2.2:      # a sudden size jump is a different vehicle
                        continue
                    pb = self._predict(g, t)
                    s = _iou(box, pb)
                    if s < 0.1:
                        lx, ly = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                        d = np.hypot(cx - lx, cy - ly)
                        lim = max(0.06 * frame_w, 0.6 * (pb[2] - pb[0]))
                        s = 0.1 * max(0.0, 1 - d / lim)
                    if s > bs:
                        best, bs = g, s
                if best is None:
                    best = _Group(nxt)
                    nxt += 1
                    groups.append(best)
                used.add(best.gid)
                best.obs.append((t, fno, box, k, c))
                best.cls_votes[k] += 1
        return groups

    def _dense_views(self, source, g, samples, max_frames=90):
        """(t, frame, box) for every frame the vehicle is in view, box interpolated between detections."""
        obs = g.obs
        t0, t1 = obs[0][0], obs[-1][0]
        frames = source.frames(t0, t1 + 1e-3, step_sec=0.0)
        if not frames:                                     # fall back to the sampled frames
            return [(o[0], samples[o[1]][1], o[2]) for o in obs]
        if len(frames) > max_frames:
            pick = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
            frames = [frames[i] for i in pick]
        times = [o[0] for o in obs]
        out = []
        for t, frame in frames:
            j = int(np.searchsorted(times, t))
            if j <= 0:
                box = obs[0][2]
            elif j >= len(obs):
                box = obs[-1][2]
            else:
                (ta, _, ba, _, _), (tb, _, bb, _, _) = obs[j - 1], obs[j]
                a = (t - ta) / max(tb - ta, 1e-6)
                box = [ba[i] + (bb[i] - ba[i]) * a for i in range(4)]
            out.append((t, frame, box))
        return out

    @staticmethod
    def _crop(frame, box, pad=0.06):
        h, w = frame.shape[:2]
        bw, bh = box[2] - box[0], box[3] - box[1]
        x1, y1 = int(max(0, box[0] - pad * bw)), int(max(0, box[1] - pad * bh))
        x2, y2 = int(min(w, box[2] + pad * bw)), int(min(h, box[3] + pad * bh))
        return frame[y1:y2, x1:x2]

    # ---------------------------------------------------------------- main entry
    def match(self, source, culprit_jpeg, culprit_cls, t_from, t_to, t_expect, out_dir,
              step_sec=0.12, max_frames=60):
        """Find the culprit in the plate camera and read its plate.

        t_from..t_to   window to search (overview timeline)
        t_expect       (start, end) when the culprit is most likely in the plate view
        Returns a dict (see bottom) or None when the plate camera had no frames.
        """
        t_start = time.time()
        if source.live:
            source.wait_until(t_to, timeout=max(5.0, t_to - time.time() + 10.0))
        frames = source.frames(t_from, t_to, step_sec)
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
            frames = [frames[i] for i in idx]
        if not frames:
            return {"status": "no_frames"}
        samples = [(t, f, self._detect(f)) for t, f in frames]
        fw = frames[0][1].shape[1]
        groups = [g for g in self._link(samples, fw) if len(g.obs) >= 2]
        if not groups:
            return {"status": "no_vehicle", "frames": len(frames)}

        ref = _hist(_unjpeg(culprit_jpeg)) if culprit_jpeg else None
        want = _group_of(culprit_cls) if culprit_cls in VEHICLES else None
        scored = []
        for g in groups:
            # appearance on the 3 largest views
            views = sorted(g.obs, key=lambda o: -(o[2][2] - o[2][0]) * (o[2][3] - o[2][1]))[:3]
            sims = [_similar(ref, _hist(self._crop(samples[o[1]][1], o[2]))) for o in views]
            app = float(np.mean(sims)) if sims else 0.0
            t0g, t1g = g.obs[0][0], g.obs[-1][0]
            gap = max(0.0, t_expect[0] - t1g, t0g - t_expect[1])     # 0 when it overlaps
            type_ok = want is None or _group_of(g.cls) == want
            score = (1.0 if type_ok else -0.6) + 1.2 * app - 0.08 * gap
            scored.append((score, g, app, gap, type_ok))
        scored.sort(key=lambda s: -s[0])
        best_score, g, app, gap, type_ok = scored[0]
        same_type = [s for s in scored if s[4]]
        ambiguous = len(same_type) > 1 and same_type[0][1] is g and same_type[0][0] - same_type[1][0] < 0.15

        # Read the plate on every frame where that vehicle is in view, not only the
        # sampled ones: a plate is sharp on just a few frames while it moves.
        ranked = []
        for t, frame, box in self._dense_views(source, g, samples):
            crop = self._crop(frame, box, pad=0.10)
            if crop.shape[0] < 30 or crop.shape[1] < 30:
                continue
            lower = cv2.cvtColor(crop[int(crop.shape[0] * 0.4):], cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(lower, cv2.CV_32F).var())
            area = crop.shape[0] * crop.shape[1]
            ranked.append((np.sqrt(area) * np.log1p(sharp), t, frame, box, crop))
        ranked.sort(key=lambda r: -r[0])
        votes = collections.defaultdict(list)       # norm -> [(conf, plate crop, t, frame, box)]
        frame_raws = []                             # per frame: all texts read, for near-match support
        photo = None
        tried = 0
        for _s, t, frame, box, crop in ranked[:self.max_reads]:
            tried += 1
            hit = self.plates.read(crop)
            if not hit:
                continue
            frame_raws.append(set(hit.get("raws") or []))
            if hit.get("norm"):
                votes[hit["norm"]].append((hit["conf"], hit["crop"], t, frame, box))
            elif photo is None and hit.get("crop") is not None:
                photo = (hit["crop"], t, frame, box)

        result = {
            "status": "matched", "frames": len(frames), "vehicles_seen": len(groups),
            "match_score": round(best_score, 3), "appearance": round(app, 3),
            "time_gap": round(gap, 2), "type_ok": bool(type_ok), "ambiguous": bool(ambiguous),
            "vehicle_class": VEHICLES.get(g.cls, "vehicle"), "reads_tried": tried,
            "plate_norm": None, "plate_conf": None, "votes": None,
            "seen_from": _fmt_clock(g.obs[0][0], source.live), "seen_to": _fmt_clock(g.obs[-1][0], source.live),
            "files": {}, "secs": None,
            "alternatives": [{"class": VEHICLES.get(s[1].cls), "score": round(s[0], 2), "appearance": round(s[2], 2),
                              "seen": f"{_fmt_clock(s[1].obs[0][0], source.live)}-{_fmt_clock(s[1].obs[-1][0], source.live)}"}
                             for s in scored[1:4]],
        }
        os.makedirs(out_dir, exist_ok=True)
        best_view = ranked[0] if ranked else None
        if votes:
            # frames that read the same plate, or the same plate with one character off
            # (a dropped or blurred digit), support it
            def support(n):
                return sum(1 for rs in frame_raws if any(edit_distance(r, n) <= 1 for r in rs))
            norm, hits = max(votes.items(), key=lambda kv: (len(kv[1]), support(kv[0]),
                                                             np.mean([h[0] for h in kv[1]])))
            confs = [h[0] for h in hits]
            near = support(norm)
            rivals = [n for n in votes if n != norm and len(votes[n]) >= len(hits)]
            # exact reads on two frames; or one sure read backed by near-misses and no rival
            if not rivals and (len(hits) >= 2 or (max(confs) >= 0.9 and (near >= 2 or len(votes) == 1))):
                top = max(hits, key=lambda h: h[0])
                result.update(plate_norm=norm, plate_conf=round(float(np.mean(confs)), 3),
                              votes=f"{len(hits)} exact + {max(0, near - len(hits))} near of {tried}")
                cv2.imwrite(os.path.join(out_dir, "plate_cam_plate.jpg"), top[1])
                result["files"]["plate"] = "plate_cam_plate.jpg"
                best_view = (0, top[2], top[3], top[4], self._crop(top[3], top[4], pad=0.10))
            else:
                result["votes"] = f"{len(hits)}/{tried} (not enough agreement)"
                top = max(hits, key=lambda h: h[0])
                cv2.imwrite(os.path.join(out_dir, "plate_cam_plate.jpg"), top[1])
                result["files"]["plate"] = "plate_cam_plate.jpg"
        elif photo is not None:
            cv2.imwrite(os.path.join(out_dir, "plate_cam_plate.jpg"), photo[0])
            result["files"]["plate"] = "plate_cam_plate.jpg"

        if best_view is not None:
            _s, t, frame, box, crop = best_view
            cv2.imwrite(os.path.join(out_dir, "plate_cam_vehicle.jpg"), crop)
            result["files"]["vehicle"] = "plate_cam_vehicle.jpg"
            full = frame.copy()
            cv2.rectangle(full, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (40, 40, 255), 4)
            label = result["plate_norm"] or "culprit"
            cv2.putText(full, label, (int(box[0]), max(30, int(box[1]) - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (40, 40, 255), 3, cv2.LINE_AA)
            sc = min(1.0, 1280 / full.shape[1])
            if sc < 1:
                full = cv2.resize(full, (int(full.shape[1] * sc), int(full.shape[0] * sc)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, "plate_cam.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 88])
            result["files"]["scene"] = "plate_cam.jpg"
            result["frame_time"] = _fmt_clock(t, source.live)
        result["secs"] = round(time.time() - t_start, 1)
        return result
