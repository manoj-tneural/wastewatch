"""Read a registration number from one or more views of the same vehicle.

Pipeline:
  1. A trained plate detector (YOLOv9, open-image-models) finds the plate on the
     vehicle crop.
  2. Crops of the same plate from several frames are aligned and median-stacked.
     A plate is sharp on only a few frames; stacking removes noise and blur that
     no single frame is free of.
  3. Two-line plates (usual on two-wheelers) are split and each line is read.
  4. Three OCR models read several slightly different versions of each image.
  5. The reads are voted character by character inside an Indian plate pattern,
     so one blurred digit on one read does not decide the number.

The result carries a per-character agreement score. A plate is "ok" only when
every character is backed by a clear majority; otherwise it is "uncertain" and
shows the best guess for a human to check.
"""
from __future__ import annotations

import collections
import re
import threading

import cv2
import numpy as np

DETECTOR_MODEL = "yolo-v9-s-608-license-plate-end2end"
OCR_MODELS = ("cct-s-v2-global-model", "cct-xs-v2-global-model", "global-plates-mobile-vit-v2-model")

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK",
    "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB",
}
_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A", "7": "T"}
_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0", "D": "0", "T": "7",
          "A": "4", "L": "1", "J": "1", "U": "0"}
# letters an OCR model commonly mixes up on plates
_CONFUSABLE = {"E": "LF", "L": "EI", "F": "EP", "K": "XR", "X": "K", "H": "NM", "N": "HM", "M": "NH",
               "O": "DQ", "D": "O", "Q": "O", "V": "UYK", "U": "V", "P": "RF", "R": "PB", "B": "R",
               "C": "G", "G": "C", "T": "I", "I": "TL", "J": "I", "W": "VK", "Y": "V"}
# (name, slot kinds). L = letter, D = digit. Standard series first, BH series last.
_TEMPLATES = []
for _d in (2, 1):
    for _l in (2, 1, 3):
        _TEMPLATES.append((f"std{_d}{_l}", "LL" + "D" * _d + "L" * _l + "DDDD"))
_TEMPLATES += [("bh2", "DDLLDDDDLL"), ("bh1", "DDLLDDDDL")]
_STD = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{4})$")
_BH = re.compile(r"^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$")


# --------------------------------------------------------------------------- text rules
def _fix_state(text):
    """A state code is a closed set. If the first two letters are not one, and exactly one
    look-alike swap makes them one (KE -> KL), use it."""
    if len(text) < 2 or text[:2] in STATE_CODES or not text[:2].isalpha():
        return text
    options = set()
    for i in (0, 1):
        for alt in _CONFUSABLE.get(text[i], ""):
            code = text[:i] + alt + text[i + 1:2]
            if code in STATE_CODES:
                options.add(code)
    return options.pop() + text[2:] if len(options) == 1 else text


def valid_plate(norm):
    m = _STD.match(norm or "")
    if m:
        return m.group(1) in STATE_CODES and int(m.group(2)) > 0     # there is no district 0 or 00
    return bool(_BH.match(norm or ""))


def _fit(text, kinds):
    """Force `text` into the slot kinds. Returns (plate, fixes) or None."""
    if len(text) != len(kinds):
        return None
    out, fixes = [], 0
    for ch, k in zip(text, kinds):
        table = _LETTER if k == "L" else _DIGIT
        fixed = table.get(ch, ch)
        if (k == "L" and not fixed.isalpha()) or (k == "D" and not fixed.isdigit()):
            return None
        fixes += fixed != ch
        out.append(fixed)
    plate = _fix_state("".join(out)) if kinds.startswith("LL") else "".join(out)
    if not valid_plate(plate):
        return None
    return plate, fixes


def format_indian(raw):
    """Return (canonical plate, substitutions) or None.

    The template that rewrites the fewest characters wins, so a string is not
    forced into a plate by swapping half of it. The state code must be a real one.
    """
    text = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    best = None
    for _name, kinds in _TEMPLATES:
        hit = _fit(text, kinds)
        if hit and (best is None or hit[1] < best[1]):
            best = hit
    if best is None or best[1] > 1:
        return None
    return best


def display_plate(norm):
    m = _STD.match(norm or "")
    if m:
        return " ".join(g for g in m.groups() if g)
    m = _BH.match(norm or "")
    if m:
        return " ".join(m.groups())
    return norm or ""


def edit_distance(a, b):
    """Levenshtein distance, small strings only."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- image helpers
def _sharpness(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def _clipped(bgr):
    """Fraction of blown-out pixels. Headlight glare turns a plate into a white blob."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float((g >= 250).mean())


def is_two_line(crop):
    h, w = crop.shape[:2]
    return w / max(h, 1) < 2.4


def _line_split(crop):
    """Cut a two-line plate between the rows of characters."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h = g.shape[0]
    k = max(5, int(h * 0.12)) | 1
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    prof = cv2.GaussianBlur(bh.mean(axis=1).reshape(-1, 1).astype(np.float32), (1, 9), 0).ravel()
    lo, hi = int(0.35 * h), int(0.65 * h)
    cut = lo + int(np.argmin(prof[lo:hi])) if hi > lo else h // 2
    m = max(2, int(0.03 * h))
    return crop[:cut + m], crop[max(0, cut - m):]


def fuse_crops(crops, size=None):
    """Align plate crops of one vehicle to the sharpest one and median-stack them.

    Returns (fused image, number of crops used). Crops that fail to align are left out.
    """
    crops = [c for c in crops if c is not None and c.size and c.shape[0] >= 10 and c.shape[1] >= 16]
    if not crops:
        return None, 0
    if size is None:
        ar = float(np.median([c.shape[1] / c.shape[0] for c in crops]))
        size = (360, max(60, int(360 / ar))) if ar >= 2.4 else (240, 180)
    W, H = size
    rs = [cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC) for c in crops]
    ref_i = int(np.argmax([_sharpness(r) for r in rs]))
    ref = cv2.cvtColor(rs[ref_i], cv2.COLOR_BGR2GRAY).astype(np.float32)
    aligned = [rs[ref_i].astype(np.float32)]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    for i, r in enumerate(rs):
        if i == ref_i:
            continue
        g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
        M = np.eye(3, dtype=np.float32)
        try:
            cc, M = cv2.findTransformECC(ref, g, M, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        except cv2.error:
            continue
        if cc < 0.7:                  # a different plate (or a bad view) does not align well
            continue
        aligned.append(cv2.warpPerspective(r, M, (W, H), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                                           borderMode=cv2.BORDER_REPLICATE).astype(np.float32))
    if len(aligned) == 1:
        return rs[ref_i], 1
    return np.median(np.stack(aligned), axis=0).astype(np.uint8), len(aligned)


def deskew(img):
    """Rotate so the character rows are level: the OCR models expect a straight plate."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(5, int(img.shape[0] * 0.12)) | 1
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    H, W = g.shape
    best = (-1.0, 0.0)
    for a in np.arange(-20, 20.5, 1.0):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1)
        v = float(np.var(cv2.warpAffine(th, M, (W, H)).sum(axis=1)))
        if v > best[0]:
            best = (v, a)
    if abs(best[1]) < 1.5:
        return img
    M = cv2.getRotationMatrix2D((W / 2, H / 2), best[1], 1)
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)


def _variants(img):
    """Slightly different looks at one image: padding, small rotations, horizontal squeeze."""
    h, w = img.shape[:2]
    out = []
    for pad in (0, int(0.06 * h) + 1):
        base = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else img
        bh, bw = base.shape[:2]
        for ang in (-3, 0, 3):
            r = base
            if ang:
                M = cv2.getRotationMatrix2D((bw / 2, bh / 2), ang, 1)
                r = cv2.warpAffine(base, M, (bw, bh), borderMode=cv2.BORDER_REPLICATE)
            out.append(r)
            if ang == 0:
                out.append(cv2.resize(r, (int(bw * 0.88), bh)))
    return out


# --------------------------------------------------------------------------- models
def ort_options(threads=3):
    """ONNX Runtime settings for the plate models: a few CPU threads, no busy-waiting.
    By default they grab every core and starve the camera loops running beside them."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return so


class _Engine:
    """Plate detector + OCR ensemble, loaded once per process and shared."""

    _inst = None
    _lock = threading.Lock()

    def __init__(self):
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import create_detector

        # 0.3: below this the detector also boxes stickers, signs and bumper text
        self.detector = create_detector(DETECTOR_MODEL, conf_thresh=0.3, sess_options=ort_options())
        self.ocrs = []
        for m in OCR_MODELS:
            o = LicensePlateRecognizer(m, device="cpu", sess_options=ort_options())
            self.ocrs.append((m, o, getattr(o.config, "image_color_mode", "rgb")))
        self.run_lock = threading.Lock()

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._inst is None:
                cls._inst = _Engine()
            return cls._inst

    def detect(self, bgr, min_w=24):
        """Plate boxes on an image (usually a vehicle crop): [(x1, y1, x2, y2, conf)]."""
        with self.run_lock:
            dets = self.detector.predict(bgr)
        out = []
        for d in dets:
            b = d.bounding_box
            if b.x2 - b.x1 >= min_w and b.y2 - b.y1 >= 8:
                out.append((int(b.x1), int(b.y1), int(b.x2), int(b.y2), float(d.confidence)))
        return sorted(out, key=lambda b: -b[4])

    def ocr(self, images):
        """Every model reads every image: [(text, mean char prob, char probs)] per model per image."""
        if not images:
            return []
        out = []
        with self.run_lock:
            for _name, o, mode in self.ocrs:
                if mode == "grayscale":
                    batch = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)[..., None] for i in images]
                else:
                    batch = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in images]
                for p in o.run(batch, return_confidence=True):
                    text = re.sub(r"[^A-Z0-9]", "", (p.plate or "").upper())
                    probs = np.asarray(p.char_probs, np.float32)[:len(p.plate or "")]
                    out.append((text, float(probs.mean()) if probs.size else 0.0))
        return out


def plate_box_crop(bgr, box, pad=(0.05, 0.10)):
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    H, W = bgr.shape[:2]
    return bgr[max(0, int(y1 - pad[1] * h)):min(H, int(y2 + pad[1] * h)),
               max(0, int(x1 - pad[0] * w)):min(W, int(x2 + pad[0] * w))]


# --------------------------------------------------------------------------- voting
# Indian plate colour schemes: (plate, characters) -> vehicle category
PLATE_TYPES = {
    "white": "private (black on white)",
    "green": "electric (white on green)",
    "yellow": "commercial/taxi (black on yellow)",
    "black": "rental (yellow on black)",
}


def plate_colour(img):
    """Which of the four schemes a plate crop is: white, green, yellow or black (its
    background colour). The plate background is most of the middle of the crop; the
    characters are the other side of the brightness split."""
    h, w = img.shape[:2]
    core = img[int(0.2 * h):int(0.8 * h), int(0.1 * w):int(0.9 * w)]
    if core.size == 0:
        return "white"
    g = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bright = g > t
    bg = bright if bright.mean() >= 0.5 else ~bright
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    hue, sat, val = (float(np.median(hsv[..., i][bg])) for i in range(3))
    if 35 <= hue <= 95 and sat >= 60:
        return "green"
    if 12 <= hue <= 34 and sat >= 80 and val >= 90:
        return "yellow"
    if val < 90:
        return "black"
    return "white"


def normalise_plate(img, colour=None):
    """Dark characters on a light plate, from the colour channel with the most contrast
    for this scheme. The OCR models are trained mostly on black-on-white plates."""
    colour = colour or plate_colour(img)
    b, g, r = cv2.split(img)
    if colour == "green":            # white text: high red; green plate: low red
        ch = 255 - r
    elif colour == "yellow":         # yellow plate: high red and green; black text: low
        ch = np.minimum(r, g)
    elif colour == "black":          # yellow text on black
        ch = 255 - np.minimum(r, g)
    else:
        ch = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ch = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)


def _reads_for(engine, img, weight, variants=True):
    """All OCR reads of one plate image (and of its two lines), each with a weight.

    Coloured and dark plates are read mainly from a black-on-white version; the original
    colours still vote, at half weight.
    """
    colour = plate_colour(img)
    if colour == "white":
        return _reads_one(engine, img, weight, variants)
    return _reads_one(engine, normalise_plate(img, colour), weight, variants) + \
        [(t, w * 0.5) for t, w in _reads_one(engine, img, weight, variants)]


def _reads_one(engine, img, weight, variants=True):
    reads = []
    var = _variants(img) if variants else [img]
    for text, conf in engine.ocr(var):
        reads.append((text, conf * weight))
    if is_two_line(img):
        tops, bots = [], []
        for v in var:
            t, b = _line_split(v)
            tops.append(t)
            bots.append(b)
        rt, rb = engine.ocr(tops), engine.ocr(bots)
        for (t, ct), (b, cb) in zip(rt, rb):
            if t and b:
                reads.append((t + b, min(ct, cb) * weight * 1.15))
    return reads


def vote(reads):
    """Character-level vote over many OCR strings inside the Indian plate templates.

    Returns dict(norm, agreement per char, support, total) or None.
    """
    by_tmpl = collections.defaultdict(list)       # template -> [(plate, weight)]
    for text, w in reads:
        if w <= 0 or not text:
            continue
        text = _fix_state(text)
        for name, kinds in _TEMPLATES:
            hit = _fit(text, kinds)
            if hit:
                by_tmpl[kinds].append((hit[0], w * (1.0 if hit[1] == 0 else 0.6 if hit[1] == 1 else 0.3)))
    if not by_tmpl:
        return None
    kinds, items = max(by_tmpl.items(), key=lambda kv: sum(w for _, w in kv[1]))
    total = sum(w for _, w in items)
    # a dropped or extra character moves a read to another pattern: the winner must hold
    # most of the evidence, or the plate is not certain
    share = total / sum(w for its in by_tmpl.values() for _, w in its)
    chars, agree, alts = [], [], []
    for i in range(len(kinds)):
        c = collections.Counter()
        for plate, w in items:
            c[plate[i]] += w
        ch, cw = c.most_common(1)[0]
        chars.append(ch)
        agree.append(cw / total)
        # every value this position could be (all fit the slot: letter or digit), most likely first
        alts.append([[a, round(aw / total, 2)] for a, aw in c.most_common(3)])
    norm = _fix_state("".join(chars))
    if not valid_plate(norm):
        return None
    support = sum(1 for p, _ in items if p == norm)
    return {"norm": norm, "agree": [round(a, 2) for a in agree], "support": support,
            "reads": len(items), "weight": round(total, 2), "share": round(share, 2),
            "chars": [{"c": norm[i], "p": round(agree[i], 2), "alt": alts[i]} for i in range(len(norm))]}


UNCLEAR = 0.7          # a character with less agreement than this is shown as unclear


def unclear_note(chars):
    """'10th: 6 (55%) or 0 (40%)' for every unclear character; '' when all are clear."""
    out = []
    for i, ch in enumerate(chars or []):
        if ch["p"] >= UNCLEAR:
            continue
        opts = " or ".join(f"{a} ({round(p * 100)}%)" for a, p in ch["alt"])
        n = i + 1
        suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        out.append(f"{n}{suffix}: {opts}")
    return "; ".join(out)


def read_plate_views(crops, thorough=True, min_agree=0.7, min_reads=6):
    """Read one vehicle's plate from several plate crops.

    thorough=False is the quick read done on every passing vehicle (~0.3 s);
    thorough=True adds varied looks and per-frame reads (~2-3 s), used for culprits.
    Returns dict(norm, display, conf, status, agree, fused, crop, used, raws).
    status: ok / uncertain / unreadable.
    """
    eng = _Engine.get()
    crops = [c for c in crops if c is not None and c.size]
    if not crops:
        return None
    crops = sorted(crops, key=lambda c: -(_sharpness(c) * min(1.0, c.shape[1] / 120.0)))
    usable = [c for c in crops if _clipped(c) < 0.25] or crops
    # one plate shape only: a one-line plate and a two-line plate are not the same plate
    two = [c for c in usable if is_two_line(c)]
    one = [c for c in usable if not is_two_line(c)]
    usable = two if len(two) > len(one) else one or two
    colour = collections.Counter(plate_colour(c) for c in usable[:8]).most_common(1)[0][0]
    fused, used = fuse_crops(usable[:12])
    reads = []
    if fused is not None and used >= 2:
        reads += _reads_for(eng, fused, 1.0 + 0.1 * min(used, 8), variants=thorough)
        if thorough:
            straight = deskew(fused)
            if straight is not fused:
                reads += _reads_for(eng, straight, 1.0 + 0.1 * min(used, 8), variants=True)
    for c in usable[:3 if thorough else 1]:
        reads += _reads_for(eng, c, 0.6, variants=False)
    if not thorough:
        min_reads = 3
    raws = sorted({t for t, _ in reads if 6 <= len(t) <= 11})
    v = vote(reads)
    photo = fused if fused is not None and used >= 2 else usable[0]
    if v is None:
        return {"norm": None, "display": None, "conf": None, "status": "unreadable", "agree": None,
                "fused": photo, "crop": usable[0], "used": used, "raws": raws,
                "colour": colour, "plate_type": PLATE_TYPES[colour]}
    low = min(v["agree"])
    # ok = every character backed by a clear majority AND the reads agree on the plate's
    # length; a near-unanimous read may have some reads in another pattern
    sure = (low >= 0.95 and v["share"] >= 0.6) or (low >= min_agree and v["share"] >= 0.85)
    # and the number must come from at least three frames: two can share the same blur
    status = "ok" if sure and v["reads"] >= min_reads and used >= 3 else "uncertain"
    return {"norm": v["norm"], "display": display_plate(v["norm"]), "conf": round(low, 3), "status": status,
            "agree": v["agree"], "support": v["support"], "reads": v["reads"], "chars": v["chars"],
            "fused": photo, "crop": usable[0], "used": used, "raws": raws,
            "colour": colour, "plate_type": PLATE_TYPES[colour]}


def find_plates(vehicle_bgr, min_w=24, inner=None):
    """Plate crops on one vehicle image, best first: [(crop, box, score)].

    inner: the vehicle's own box inside this image (x1, y1, x2, y2). The image may be
    padded around it so a plate on the bumper edge is not cut; a plate belongs to the
    vehicle when its centre is inside that box and not in its top fifth (a vehicle ahead).
    """
    eng = _Engine.get()
    H, W = vehicle_bgr.shape[:2]
    ix1, iy1, ix2, iy2 = inner if inner is not None else (0, 0, W, H)
    iw, ih = max(ix2 - ix1, 1), max(iy2 - iy1, 1)
    out = []
    for b in eng.detect(vehicle_bgr, min_w=min_w):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        rx, ry = (cx - ix1) / iw, (cy - iy1) / ih
        if not (0.0 <= rx <= 1.0 and 0.2 <= ry <= 1.05):
            continue
        if b[0] <= 0 or b[2] >= W - 1:      # cut by the picture itself
            continue
        out.append((plate_box_crop(vehicle_bgr, b), b, b[4] * (1.0 - 0.5 * abs(rx - 0.5))))
    return sorted(out, key=lambda o: -o[2])


class PlateReader:
    """Single-image reader kept for the overview camera snapshots."""

    def __init__(self):
        self.error = None
        self.status = "pending"

    def start(self):
        threading.Thread(target=self._ensure, daemon=True).start()

    def _ensure(self):
        try:
            _Engine.get()
            self.status = "ready"
            return True
        except Exception as e:  # noqa: BLE001 - a missing model must not stop detection
            self.error = str(e)
            self.status = "unavailable"
            return False

    def read(self, bgr):
        """Return {norm, display, conf, crop, raws}. display is set only for a confident read."""
        if bgr is None or getattr(bgr, "size", 0) == 0:
            self.status = "unreadable"
            return None
        if not self._ensure():
            return None
        plates = find_plates(bgr)
        if not plates:
            self.status = "unreadable"
            return None
        r = read_plate_views([plates[0][0]])
        ok = r and r["status"] == "ok"
        self.status = "ok" if ok else "unreadable"
        return {"norm": r["norm"] if ok else None, "display": r["display"] if ok else None,
                "conf": r["conf"] if ok else None, "crop": plates[0][0], "raws": r["raws"] if r else []}
