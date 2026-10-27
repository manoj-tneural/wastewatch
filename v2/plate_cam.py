"""Second camera: a narrow, zoomed view used only to read number plates.

The overview camera (detector.py) sees the throw and picks the culprit. PlateLogger
reads this camera all the time: every vehicle that passes is tracked and its plate crops
are kept in memory. When an event is confirmed, the culprit is picked from the vehicles
that passed while it was in the overview, and its plate gets a multi-frame read.

Both cameras share one timeline, in seconds:
  * recorded clips - seconds since the overview clip started. The plate clip is
    shifted by `offset`, read from the clock burned into each picture and fine-tuned
    by matching the traffic in both clips (resolve_offset).
  * live RTSP      - wall-clock time (time.time()) stamped as each frame arrives.

Matching the culprit across cameras uses three cues:
  1. vehicle type   - two-wheeler vs car/bus/truck
  2. timing         - the plate camera sees part of the road the overview also sees,
                      so the culprit passes it while it is in the overview
  3. appearance     - colour histogram of the plate-view vehicle against the
                      culprit snapshot from the overview (rider clothes count)
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import re
import threading
import time

import cv2
import numpy as np

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
    """Pick the clip offset: the burned-in camera clocks first, traffic matching to fine-tune,
    NVR file names only as a last resort.

    NVR export names are not always the recording time (an export made later can carry
    the export time), so the names are only trusted when the pictures cannot decide.
    Returns (offset, method, score).
    """
    by_name = auto_offset(overview_name or overview_path, plate_name or plate_path)
    try:
        a, b = osd_start(overview_path), osd_start(plate_path)
    except Exception as e:  # noqa: BLE001 - OCR problems fall back to the traffic match
        log(f"sync: could not read the camera clocks ({e})")
        a = b = None
    if a is not None and b is not None and abs(b - a) < 3600:
        off = round(b - a, 2)
        fine, score = offset_by_motion(overview_path, plate_path, around=off, search=1.5, max_sec=120)
        if fine is not None and score >= 0.35:
            return fine, "camera clocks + traffic", score
        return off, "camera clocks", score
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

    def iter_frames(self, rate=10.0):
        """Every frame of the clip in order, thinned to about `rate` per second:
        yields (overview_time, frame). Uses its own reader so frames() stays free."""
        cap = cv2.VideoCapture(self.path)
        step = max(1, int(round(self.fps / rate))) if rate else 1
        i = 0
        try:
            while True:
                if i % step:
                    if not cap.grab():
                        break
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield i / self.fps + self.offset, frame
                i += 1
        finally:
            cap.release()

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
        # dense enough that a fast vehicle in the zoomed view stays one track
        return StreamSource(src, buffer_sec=30.0, keep_fps=15.0, log=log)
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


class Calibration:
    """What the system has learned about this camera pair from confident matches.

    `delay` is how long after passing the dumping spot a vehicle is at its closest in
    the plate camera. Kept per mode ("clip" / "live"): clip sync already absorbs part
    of the travel time, live arrival stamps do not.
    """

    MIN_SAMPLES = 5
    KEEP = 60

    def __init__(self, path, key="live", log=print):
        self.path, self.key, self.log = path, key, log
        self._lock = threading.Lock()

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def delay(self):
        """(median delay, tolerance) once there are enough samples, else None."""
        with self._lock:
            d = self._read().get(self.key, {}).get("delays", [])
        if len(d) < self.MIN_SAMPLES:
            return None
        arr = np.array(d[-self.KEEP:], np.float32)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) * 1.4826
        return med, max(0.8, 2.5 * mad)

    def samples(self):
        with self._lock:
            return len(self._read().get(self.key, {}).get("delays", []))

    def add(self, delay):
        with self._lock:
            data = self._read()
            ent = data.setdefault(self.key, {"delays": []})
            ent["delays"] = (ent.get("delays", []) + [round(float(delay), 3)])[-self.KEEP:]
            tmp = self.path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=1)
                os.replace(tmp, self.path)
            except OSError as e:
                self.log(f"calibration not saved: {e}")


def _pct_ranks(values):
    """0 for the smallest value, 1 for the largest; ties share a rank."""
    n = len(values)
    if n < 2 or max(values) - min(values) < 1e-6:
        return [0.5] * n
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return [float(o) / (n - 1) for o in order]


class _Pass:
    """One vehicle passing the plate camera: its track, best photo and plate crops."""

    keep_plates = 12

    def __init__(self, tid, t):
        self.tid = tid
        self.t0 = self.t1 = t
        self.cls_votes = collections.Counter()
        self.box = None
        self.photo = None              # (score, vehicle crop jpeg, scene jpeg, box, t)
        # plate candidates grouped by where they sit on the vehicle (rear middle, rear
        # left, side...): each group is {"pos", "n", "plates": [(score, crop)], "photo"}
        self.groups = []
        self.chosen = None             # index of the group the plate was read from
        self.views = []                # (score, vehicle crop jpeg) for the second-opinion reader
        self.seen = 0
        self.quick = None              # quick read, done when the vehicle leaves
        self.full = None               # thorough read, done only for a culprit
        self.hist = None

    @property
    def cls(self):
        return self.cls_votes.most_common(1)[0][0] if self.cls_votes else 2

    @property
    def plates(self):
        return [q for g in self.groups for q in g["plates"]]

    def add_plate(self, score, crop, pos, photo=None):
        """Add a plate crop to the group at this position on the vehicle (or a new group).
        photo() makes the evidence photo; it is called only when this is the group's best."""
        best = None
        for g in self.groups:
            d = max(abs(g["pos"][0] - pos[0]), abs(g["pos"][1] - pos[1]))
            if d < 0.18 and (best is None or d < best[0]):
                best = (d, g)
        if best is None:
            if len(self.groups) >= 4:
                return
            g = {"pos": pos, "n": 0, "plates": [], "photo": None}
            self.groups.append(g)
        else:
            g = best[1]
        g["n"] += 1
        g["pos"] = tuple(a + (b - a) / g["n"] for a, b in zip(g["pos"], pos))
        g["plates"].append((score, crop))
        if len(g["plates"]) > self.keep_plates:
            g["plates"].sort(key=lambda q: -q[0])
            del g["plates"][self.keep_plates:]
        if photo is not None and (g["photo"] is None or score > g["photo"][0]):
            g["photo"] = photo()

    def best_plates(self, group=None):
        src = self.groups[group]["plates"] if group is not None else self.plates
        return [c for _, c in sorted(src, key=lambda q: -q[0])]

    @property
    def plate_photo(self):
        """Evidence photo from the frame where the plate that was read is clearest."""
        if self.chosen is not None:
            return self.groups[self.chosen]["photo"]
        g = max(self.groups, key=lambda g: g["n"], default=None)
        return g["photo"] if g else None

    def read_plate(self, plates_mod):
        """Thorough read of this vehicle's plate. With plate candidates in more than one
        place on the vehicle, the two most seen are read and the clearer result is kept."""
        if self.full is not None or not self.groups:
            return self.full
        order = sorted(range(len(self.groups)), key=lambda i: -self.groups[i]["n"])[:2]
        results = []
        for i in order:
            if i != order[0] and self.groups[i]["n"] < 2:
                continue                       # one sighting elsewhere is usually a sticker
            r = plates_mod.read_plate_views(self.best_plates(i), thorough=True)
            if r is not None:
                results.append((i, r))
        if not results:
            return None

        def rank(item):
            i, r = item
            return (r["status"] == "ok", r.get("norm") is not None, r.get("conf") or 0, self.groups[i]["n"])
        self.chosen, self.full = max(results, key=rank)
        return self.full

    def add_view(self, score, jpeg, keep=24):
        """Views spread over the whole pass: when full, every other one is dropped.
        The plate is often whole only in the middle of a pass, not in the biggest view."""
        self.view_n = getattr(self, "view_n", 0) + 1
        self.view_step = getattr(self, "view_step", 1)
        if self.view_n % self.view_step:
            return
        self.views.append((score, jpeg))
        if len(self.views) > keep:
            self.views = self.views[::2]
            self.view_step *= 2


def _mask_osd(frame):
    """Grey out the camera's own text (clock top left, 'Camera 01' bottom right) so it is
    never taken for a number plate."""
    H, W = frame.shape[:2]
    out = frame.copy()
    out[:int(0.10 * H), :int(0.36 * W)] = 114
    out[int(0.86 * H):int(0.95 * H), int(0.70 * W):int(0.86 * W)] = 114
    return out


_LEGACY = None
_LEGACY_LOCK = threading.Lock()


def legacy_read(vehicle_crops, log=print):
    """The original reader and its frame vote: a number is accepted when two frames read it
    exactly, or one sure read is backed by near-identical reads and nothing rivals it.
    Returns (norm, conf, tally, plate crop) or None."""
    global _LEGACY
    from .plates_legacy import PlateReader as LegacyReader, edit_distance

    with _LEGACY_LOCK:
        if _LEGACY is None:
            _LEGACY = LegacyReader()
    votes = collections.defaultdict(list)
    frame_raws = []
    for crop in vehicle_crops:
        try:
            hit = _LEGACY.read(crop)
        except Exception as e:  # noqa: BLE001 - a second opinion must not break the event
            log(f"plate camera: second reader failed: {e}")
            return None
        if not hit:
            continue
        frame_raws.append(set(hit.get("raws") or []))
        if hit.get("norm"):
            votes[hit["norm"]].append((hit["conf"], hit["crop"]))
    if not votes:
        return None

    def support(n):
        return sum(1 for rs in frame_raws if any(edit_distance(r, n) <= 1 for r in rs))

    norm, hits = max(votes.items(), key=lambda kv: (len(kv[1]), support(kv[0]), np.mean([h[0] for h in kv[1]])))
    confs = [h[0] for h in hits]
    near = support(norm)
    rivals = [n for n in votes if n != norm and len(votes[n]) >= len(hits)]
    if rivals or not (len(hits) >= 2 or (max(confs) >= 0.9 and (near >= 2 or len(votes) == 1))):
        return None
    top = max(hits, key=lambda h: h[0])
    return norm, round(float(np.mean(confs)), 3),         f"{len(hits)} exact + {max(0, near - len(hits))} near of {len(vehicle_crops)}", top[1]


def _crop(frame, box, pad=0.06):
    h, w = frame.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1, y1 = int(max(0, box[0] - pad * bw)), int(max(0, box[1] - pad * bh))
    x2, y2 = int(min(w, box[2] + pad * bw)), int(min(h, box[3] + pad * bh))
    return frame[y1:y2, x1:x2]


def _align_passes(context, passes, fw, verge_side, delay):
    """Pair overview vehicles with plate-camera passes, keeping their order (Manoj's
    order + lane matching, on v2's vehicle passes).

    On one stretch of road vehicles pass both cameras in the same order, so the overview
    sequence is lined up against the plate-camera sequence like two lists (a vehicle one
    camera missed is skipped). Each pairing costs: type (two-wheeler vs car/bus/truck),
    colour, lane (near-verge pairs with near-verge) and timing (learned travel time, or
    estimated from this window). Returns (pairs, cost, info, total, solve).
    """
    n, m = len(context), len(passes)
    pl_t = [p.photo[4] for p in passes]                     # time of the biggest view = closest
    ov_order = sorted(range(n), key=lambda i: context[i]["t_pass"])
    pl_order = sorted(range(m), key=lambda j: pl_t[j])
    lane_raw = [min(10.0, float(c.get("lane", 0.0))) for c in context]
    lane_ov = _pct_ranks(lane_raw)

    def lane_pl(box):
        bw = max(box[2] - box[0], 1.0)
        return (box[0] / bw) if verge_side < 0 else ((fw - box[2]) / bw)
    lane_pv = _pct_ranks([lane_pl(p.photo[3]) for p in passes])
    hist_ov = [_hist(_unjpeg(c["jpeg"])) if c.get("jpeg") else None for c in context]
    for p in passes:
        if p.hist is None:
            p.hist = _hist(_unjpeg(p.photo[1]))
    cost = np.zeros((n, m), np.float32)
    info = {}
    for i in range(n):
        for j in range(m):
            c = context[i]
            type_ok = _group_of(c["cls"]) == _group_of(passes[j].cls)
            app = _similar(hist_ov[i], passes[j].hist)
            dt = pl_t[j] - c["t_pass"]
            tterm = min(3.0, abs(dt - delay[0]) / delay[1]) if delay is not None else 0.05 * max(0.0, abs(dt) - 3.0)
            lterm = 0.5 * abs(lane_ov[i] - lane_pv[j]) if n > 1 and m > 1 else 0.0
            cost[i, j] = (0.0 if type_ok else 2.5) + 1.2 * (1.0 - app) + tterm + lterm
            info[(i, j)] = {"app": app, "type_ok": type_ok, "dt": dt, "lane": lterm, "time": tterm}
    skip_ov_cost = [1.5 if lane_raw[i] <= 2.5 else 0.45 for i in range(n)]
    skip_pl = 0.6
    INF = 1e9

    def solve(forbid=None):
        D = np.full((n + 1, m + 1), INF, np.float64)
        back = np.zeros((n + 1, m + 1), np.int8)
        D[0, 0] = 0.0
        for a in range(n + 1):
            for b in range(m + 1):
                if a == 0 and b == 0:
                    continue
                best, how = INF, 0
                if a > 0 and b > 0:
                    i, j = ov_order[a - 1], pl_order[b - 1]
                    if (i, j) != forbid:
                        v = D[a - 1, b - 1] + cost[i, j]
                        if v < best:
                            best, how = v, 1
                if a > 0 and D[a - 1, b] + skip_ov_cost[ov_order[a - 1]] < best:
                    best, how = D[a - 1, b] + skip_ov_cost[ov_order[a - 1]], 2
                if b > 0 and D[a, b - 1] + skip_pl < best:
                    best, how = D[a, b - 1] + skip_pl, 3
                D[a, b], back[a, b] = best, how
        pairs, a, b = {}, n, m
        while a > 0 or b > 0:
            how = back[a, b]
            if how == 1:
                pairs[ov_order[a - 1]] = pl_order[b - 1]
                a, b = a - 1, b - 1
            elif how == 2:
                a -= 1
            else:
                b -= 1
        return pairs, float(D[n, m])

    if delay is None and n >= 2:
        grid = np.arange(-6.0, 6.01, 0.1)
        votes = np.zeros_like(grid)
        for i in range(n):
            best = np.zeros_like(grid)
            for j in range(m):
                inf = info[(i, j)]
                if inf["type_ok"]:
                    best = np.maximum(best, inf["app"] * np.exp(-((inf["dt"] - grid) / 0.6) ** 2))
            votes += best
        k = int(np.argmax(votes))
        if votes[k] >= 0.8 and votes[k] >= 1.25 * float(np.median(votes) + 1e-6):
            med, tol = float(grid[k]), 0.8
            for (i, j), inf in list(info.items()):
                old = inf["time"]
                inf["time"] = min(3.0, abs(inf["dt"] - med) / tol)
                cost[i, j] += inf["time"] - old
            info["_window_delay"] = (round(med, 2), tol)
    pairs, total = solve()
    return pairs, cost, info, total, solve


class PlateLogger:
    """Reads every vehicle that passes the plate camera, continuously, in its own thread.

    Each pass keeps the vehicle's best photo and its sharpest plate crops, and gets a
    quick read as it leaves the view. Everything is held in memory only: when live,
    passes older than `retention_sec` are dropped, and nothing is written to disk unless
    the pass is matched to a littering event. When an event asks, the culprit is picked
    from the passes around that time and its plate gets a thorough multi-frame read.
    """

    def __init__(self, source, model="yolo11n.pt", conf=0.30, imgsz=960, device="", rate=0.0,
                 retention_sec=90.0, log=print):
        from ultralytics import YOLO

        self.source = source
        self.model = YOLO(model)       # own instance: never disturbs the overview tracker
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or None
        self.rate = rate
        self.retention = retention_sec
        self.log = log
        self.active: dict[int, _Pass] = {}
        self.done: collections.deque = collections.deque()
        self.lock = threading.Lock()
        self.processed_t = -1e18
        self.finished = False
        self.frames = 0
        self.error = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="plate-logger")

    def start(self):
        from . import plates
        plates._Engine.get()           # load the plate models before the first vehicle arrives
        self._thread.start()
        return self

    def stop(self):
        self._stop = True

    # ---------------------------------------------------------------- reading the camera
    def _frames(self):
        if not self.source.live:
            yield from self.source.iter_frames(self.rate)
            return
        last = time.time() - 1.0
        while not self._stop:
            got = self.source.frames(last + 1e-3, time.time(), step_sec=1.0 / self.rate if self.rate else 0.0)
            for t, f in got:
                last = t
                yield t, f
            if not got:
                time.sleep(0.05)

    def _run(self):
        try:
            for t, frame in self._frames():
                if self._stop:
                    break
                self._process(t, frame)
        except Exception as e:  # noqa: BLE001 - the overview keeps running without plates
            import traceback
            traceback.print_exc()
            self.error = str(e)
            self.log(f"plate camera stopped: {e}")
        finally:
            with self.lock:
                for p in list(self.active.values()):
                    self._close_pass(p)
                self.active.clear()
                self.finished = True

    def _detect(self, frame):
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.conf, imgsz=self.imgsz,
                             classes=[PERSON] + list(VEHICLES), verbose=False, device=self.device)[0]
        if r.boxes is None or not len(r.boxes) or r.boxes.id is None:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        persons = [b.tolist() for b, k in zip(xyxy, cls) if k == PERSON]
        out = []
        for b, k, c, tid in zip(xyxy, cls, conf, ids):
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
            out.append((int(tid), box, int(k), float(c)))
        return out

    def _process(self, t, frame):
        from . import plates

        H, W = frame.shape[:2]
        dets = self._detect(frame)
        clean = _mask_osd(frame) if dets else frame
        with self.lock:
            for tid, box, k, c in dets:
                p = self.active.get(tid)
                if p is None:
                    p = self.active[tid] = _Pass(tid, t)
                p.t1 = t
                p.box = box
                p.seen += 1
                p.cls_votes[k] += 1
        for tid, box, k, c in dets:
            p = self.active[tid]
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < 0.05 * W or bh < 0.05 * H:
                continue
            border = box[0] < 4 or box[1] < 4 or box[2] > W - 4 or box[3] > H - 4
            veh = _crop(clean, box, pad=0.06)
            score = bw * bh * c * (0.35 if border else 1.0)
            wide = _crop(clean, box, pad=0.10)
            low = cv2.cvtColor(wide[int(wide.shape[0] * 0.4):], cv2.COLOR_BGR2GRAY)
            # the plate search looks a little beyond the box: boxes often stop at the bumper
            p.add_view(np.sqrt(wide.shape[0] * wide.shape[1]) * np.log1p(float(cv2.Laplacian(low, cv2.CV_32F).var())),
                       _jpeg(wide, 90))
            if p.photo is None or score > p.photo[0]:
                veh = _crop(frame, box, pad=0.06)
                sc = min(1.0, 1280 / W)
                scene = cv2.resize(frame, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
                p.photo = (score, _jpeg(veh, 92), _jpeg(scene, 85), [v * sc for v in box], t)
            # plate search on alternate views (it is the slow step); once enough sharp
            # crops are kept, every fourth
            if p.seen % (4 if len(p.plates) >= _Pass.keep_plates else 2):
                continue
            px1, py1 = int(max(0, box[0] - 0.15 * bw)), int(max(0, box[1] - 0.05 * bh))
            px2, py2 = int(min(W, box[2] + 0.15 * bw)), int(min(H, box[3] + 0.15 * bh))
            search = clean[py1:py2, px1:px2]
            inner = (box[0] - px1, box[1] - py1, box[2] - px1, box[3] - py1)
            # up to two plates per view: some vehicles carry one at the rear and one at the side
            for crop, pb, pc, pos in plates.find_plates(search, min_w=28, inner=inner, with_pos=True)[:2]:
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
                pscore = np.sqrt(sharp) * min(crop.shape[1], 220) * pc

                def photo(pscore=pscore, pb=pb, box=box):
                    # evidence photo: the frame where this plate is clearest
                    sc = min(1.0, 1280 / W)
                    scene = cv2.resize(frame, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
                    pbox = [(pb[0] + px1) * sc, (pb[1] + py1) * sc, (pb[2] + px1) * sc, (pb[3] + py1) * sc]
                    return (pscore, _jpeg(_crop(frame, box, pad=0.06), 92), _jpeg(scene, 88),
                            [v * sc for v in box], t, pbox)
                p.add_plate(pscore, crop, pos, photo)
        seen_now = {d[0] for d in dets}
        with self.lock:
            for tid in [tid for tid, p in self.active.items() if tid not in seen_now and t - p.t1 > 0.8]:
                self._close_pass(self.active.pop(tid))
            self.processed_t = t
            self.frames += 1
            if self.source.live:
                while self.done and t - self.done[0].t1 > self.retention:
                    self.done.popleft()

    def _close_pass(self, p):
        from . import plates

        if p.seen < 2:
            return
        # crops are kept; the plate is read only if this vehicle is matched to a throw
        self.done.append(p)

    # ---------------------------------------------------------------- answering an event
    def _wait(self, t_to, live):
        end = time.time() + (max(10.0, t_to - time.time() + 15.0) if live else 900.0)
        while not self.finished and self.processed_t < t_to and time.time() < end:
            time.sleep(0.05)

    def match(self, culprit_jpeg, culprit_cls, t_from, t_to, t_expect, out_dir, label="event",
              context=None, verge_side=-1, calib=None):
        """Find the culprit among the vehicles that passed the plate camera and read its plate.

        t_from..t_to   window to search (overview timeline)
        t_expect       (start, end) when the culprit was in the overview view
        label          names the event in the log
        context        overview vehicles in the window ({"cls", "t_pass", "lane", "jpeg", "culprit"})
                       for order + lane matching; None = best single match only
        verge_side     -1 when the verge is on the left of the plate view, +1 on the right
        calib          Calibration: learned travel time between the cameras
        Returns a dict (see below) for detector._apply_platecam. result["trace"] records
        every step (window, candidates and their scores, both readers, the decision) for
        the dashboard's matching log.
        """
        from . import plates
        from .plates import display_plate

        t_start = time.time()
        live = self.source.live
        self._wait(t_to + 1.0, live)
        with self.lock:
            passes = [p for p in list(self.done) + list(self.active.values())
                      if p.t1 >= t_from and p.t0 <= t_to and p.photo is not None]
        clk = lambda t: _fmt_clock(t, live)
        window = f"{clk(t_from)} - {clk(t_to)}"
        expect = f"{clk(t_expect[0])} - {clk(t_expect[1])}"
        want_name = VEHICLES.get(culprit_cls, "vehicle")
        if not passes:
            status = "no_vehicle" if self.frames else "no_frames"
            self.log(f"[match] {label}: culprit {want_name} in overview {expect}; plate camera searched {window}: "
                     f"{'no vehicle passed' if self.frames else 'no plate camera footage for that time'}")
            return {"status": status, "frames": self.frames,
                    "trace": {"window": window, "expected": expect, "culprit_class": want_name, "candidates": [],
                              "decision": "no vehicle in the plate camera at that time"}}
        ref = _hist(_unjpeg(culprit_jpeg)) if culprit_jpeg else None
        want = _group_of(culprit_cls) if culprit_cls in VEHICLES else None
        scored = []
        for p in passes:
            if p.hist is None:
                p.hist = _hist(_unjpeg(p.photo[1]))
            app = _similar(ref, p.hist)
            gap = max(0.0, t_expect[0] - p.t1, p.t0 - t_expect[1])     # 0 when it overlaps
            type_ok = want is None or _group_of(p.cls) == want
            score = (1.0 if type_ok else -0.6) + 1.2 * app - 0.08 * gap
            scored.append((score, p, app, gap, type_ok))
        scored.sort(key=lambda s: -s[0])
        best_score, p, app, gap, type_ok = scored[0]
        same_type = [s for s in scored if s[4]]
        ambiguous = len(same_type) > 1 and same_type[0][1] is p and same_type[0][0] - same_type[1][0] < 0.15
        method = "best single match"
        delay = calib.delay() if calib is not None else None
        # order + lane: line up every overview vehicle with the plate-camera vehicles
        ci = next((i for i, c in enumerate(context or []) if c.get("culprit")), None)
        if ci is not None:
            try:
                fw = (self.source.width or 1280) * min(1.0, 1280 / max(self.source.width or 1280, 1))
                pairs, cost, info, total, solve = _align_passes(context, passes, fw, verge_side, delay)
            except Exception as e:  # noqa: BLE001 - fall back to the single-vehicle score
                self.log(f"plate camera: order matching failed ({e}), using best single match")
                pairs = {}
            if ci in pairs:
                j = pairs[ci]
                inf = info[(ci, j)]
                _p2, total2 = solve(forbid=(ci, j))
                order_ambiguous = (total2 - total) < 0.35
                wd = info.get("_window_delay")
                order_name = "order + lane" + (" + learned timing" if delay else (" + timing from this window" if wd else ""))
                # the travel time is learned from every clear pairing, whichever method is used
                if calib is not None:
                    learned = []
                    if not order_ambiguous and inf["type_ok"] and inf["app"] >= 0.45:
                        learned.append(inf["dt"])
                    for i2, j2 in pairs.items():
                        if i2 == ci or len(learned) >= 4 or (i2, j2) not in info:
                            continue
                        f2 = info[(i2, j2)]
                        if f2["type_ok"] and f2["app"] >= 0.6 and cost[i2, j2] < 0.7:
                            learned.append(f2["dt"])
                    for dt in learned:
                        calib.add(dt)
                # Single match first; order + lane decides only when the single match is
                # ambiguous. Checked by eye on the 20 sample-footage events where the two
                # differed: single 5 right, order 6, this rule 8. When the two methods pick
                # different vehicles the match stays flagged for a person to check.
                if ambiguous:
                    disagree = passes[j] is not p
                    p = passes[j]
                    app, type_ok = inf["app"], inf["type_ok"]
                    gap = max(0.0, t_expect[0] - p.t1, p.t0 - t_expect[1])
                    best_score = round(3.2 - float(cost[ci, j]), 3)
                    ambiguous = order_ambiguous or disagree
                    method = order_name + (" (single match was ambiguous; the two disagree)" if disagree
                                           else " (confirms the single match)")
                    # keep the picked vehicle first, so the trace shows it as the choice
                    scored.sort(key=lambda s: s[1] is not p)
                    scored[0] = (best_score, p, app, gap, type_ok)
                else:
                    method = "best single match" + (" (order + lane agrees)" if passes[j] is p else
                                                    " (order + lane differs; single match was clear)")
            elif len(context) > 1:
                method = "best single match (culprit not lined up)"

        if p.full is None and p.groups:
            p.read_plate(plates)
        if getattr(p, "legacy", "unset") == "unset":
            # second opinion only when the main reader is not sure
            sure = p.full is not None and p.full.get("status") == "ok"
            p.legacy = None if sure or not p.views else                 legacy_read([_unjpeg(j) for _, j in sorted(p.views, key=lambda v: -v[0])[:12]], log=self.log)
        r = p.full or p.quick
        cands = [{"rank": k + 1, "class": VEHICLES.get(s[1].cls, "vehicle"), "seen": f"{clk(s[1].t0)} - {clk(s[1].t1)}",
                  "type_ok": bool(s[4]), "appearance": round(s[2], 2), "time_gap": round(s[3], 2),
                  "score": round(s[0], 2), "plate": (s[1].full or s[1].quick or {}).get("display"),
                  "chosen": s[1] is p} for k, s in enumerate(scored[:8])]
        result = {
            "status": "matched", "frames": self.frames, "vehicles_seen": len(passes),
            "match_score": round(best_score, 3), "appearance": round(app, 3),
            "time_gap": round(gap, 2), "type_ok": bool(type_ok), "ambiguous": bool(ambiguous),
            "vehicle_class": VEHICLES.get(p.cls, "vehicle"), "reads_tried": len(p.plates),
            "method": method, "overview_vehicles": len(context or []),
            "learned_delay": round(delay[0], 2) if delay else None,
            "delay_samples": calib.samples() if calib is not None else 0,
            "plate_norm": None, "plate_conf": None, "votes": None, "plate_guess": None, "plate_agree": None,
            "plate_type": (p.full or p.quick or {}).get("plate_type"),
            "seen_from": _fmt_clock(p.t0, live), "seen_to": _fmt_clock(p.t1, live),
            "files": {}, "secs": None,
            "alternatives": [{"class": VEHICLES.get(s[1].cls), "score": round(s[0], 2), "appearance": round(s[2], 2),
                              "plate": (s[1].quick or {}).get("display"),
                              "seen": f"{_fmt_clock(s[1].t0, live)}-{_fmt_clock(s[1].t1, live)}"}
                             for s in scored[1:4]],
        }
        os.makedirs(out_dir, exist_ok=True)
        if r is not None and r.get("norm"):
            img = r.get("fused") if r.get("fused") is not None else r.get("crop")
            if img is not None:
                cv2.imwrite(os.path.join(out_dir, "plate_cam_plate.jpg"), img)
                result["files"]["plate"] = "plate_cam_plate.jpg"
            if r.get("norm"):
                tally = f"{r.get('support', 0)} of {r.get('reads', 0)} reads agree, {r.get('used', 0)} frames fused"
                if r["status"] == "ok":
                    result.update(plate_norm=r["norm"], plate_conf=r["conf"], votes=tally)
                else:
                    result.update(plate_guess=r["norm"], plate_agree=r.get("agree"), votes=tally)
        leg = p.legacy
        if leg:
            lnorm, lconf, ltally, lcrop = leg
            if result["plate_norm"] and result["plate_norm"] != lnorm:
                # the two readers disagree: do not assert either number
                result.update(plate_guess=result["plate_norm"], plate_norm=None, plate_conf=None,
                              votes=f"{result['votes']}; second reader says {display_plate(lnorm)}")
            elif result["plate_norm"] == lnorm:
                result["votes"] = f"{result['votes']}; second reader agrees ({ltally})"
            else:
                result.update(plate_norm=lnorm, plate_conf=lconf, plate_guess=None, votes=f"second reader: {ltally}")
                cv2.imwrite(os.path.join(out_dir, "plate_cam_plate.jpg"), lcrop)
                result["files"]["plate"] = "plate_cam_plate.jpg"
        # which characters are clear, and the likely values of the unclear ones
        shown = result["plate_norm"] or result["plate_guess"]
        if r is not None and r.get("chars") and shown == r.get("norm"):
            result["plate_chars"] = r["chars"]
            result["plate_unclear"] = plates.unclear_note(r["chars"])
        # ---- trace: how the plate was found
        if result["plate_norm"]:
            decision = f"plate {display_plate(result['plate_norm'])} confirmed"
        elif result["plate_guess"]:
            decision = f"best guess {display_plate(result['plate_guess'])} - not confirmed, check the plate photo"
        else:
            decision = "plate not readable"
        if ambiguous:
            decision += "; another vehicle of the same type scored almost as high - check the photo"
        new_read = None
        if r is not None:
            new_read = {"plate": display_plate(r["norm"]) if r.get("norm") else None, "status": r.get("status"),
                        "agreement": r.get("agree"), "reads_agree": r.get("support"), "reads": r.get("reads"),
                        "frames_fused": r.get("used"), "plate_type": r.get("plate_type"),
                        "unclear": plates.unclear_note(r.get("chars")) if r.get("chars") else None}
        second = {"plate": display_plate(leg[0]), "votes": leg[2]} if leg else None
        result["trace"] = {"window": window, "expected": expect, "culprit_class": want_name, "method": method,
                           "candidates": [c for c in cands if c["chosen"]], "vehicles_considered": len(passes),
                           "plate_crops": len(p.plates), "new_reader": new_read,
                           "second_reader": second, "decision": decision}
        for c in cands:
            if c["chosen"]:
                self.log(f"[match] {label}: picked the {c['class']} seen in the plate camera {c['seen']} "
                         f"(type {'same' if c['type_ok'] else 'different'}, colour match {c['appearance']:.2f}, "
                         f"time gap {c['time_gap']:.1f}s, out of {len(passes)} vehicles)")
        if new_read:
            self.log(f"[plate] {label}: reader: {new_read['plate'] or 'no number'} ({new_read['status']}, "
                     f"{new_read['reads_agree'] or 0} of {new_read['reads'] or 0} reads agree, "
                     f"{new_read['frames_fused']} frames fused)"
                     + (f"; unclear {new_read['unclear']}" if new_read.get("unclear") else ""))
        self.log(f"[plate] {label}: second reader: "
                 f"{second['plate'] + ' (' + second['votes'] + ')' if second else 'no confident read'}")
        self.log(f"[plate] {label}: {decision}")
        # evidence photos from the frame where the plate is clearest (else the biggest view)
        pp = p.plate_photo
        _s, veh_jpg, scene_jpg, box, t = pp[:5] if pp else p.photo
        with open(os.path.join(out_dir, "plate_cam_vehicle.jpg"), "wb") as f:
            f.write(veh_jpg)
        result["files"]["vehicle"] = "plate_cam_vehicle.jpg"
        full = _unjpeg(scene_jpg)
        cv2.rectangle(full, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (40, 40, 255), 3)
        if pp:
            b = pp[5]
            cv2.rectangle(full, (int(b[0]) - 3, int(b[1]) - 3), (int(b[2]) + 3, int(b[3]) + 3), (0, 215, 255), 2)
        tag = result["plate_norm"] or result["plate_guess"] or "culprit"
        tag = display_plate(tag) + ("" if result["plate_norm"] or tag == "culprit" else " ?")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        ty = max(th + 10, int(box[1]) - 10)
        cv2.rectangle(full, (int(box[0]), ty - th - 8), (int(box[0]) + tw + 10, ty + 6), (40, 40, 255), -1)
        cv2.putText(full, tag, (int(box[0]) + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, "plate_cam.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 88])
        result["files"]["scene"] = "plate_cam.jpg"
        result["frame_time"] = _fmt_clock(t, live)
        result["frame_t"] = t
        result["secs"] = round(time.time() - t_start, 1)
        return result
