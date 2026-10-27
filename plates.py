"""Read a registration number from a vehicle snapshot.

A plate is a bright rectangle (white or yellow) with dark characters, often on
the lower half of the vehicle. Two-line plates, usual on two-wheelers, are
split and read as two lines. The result is kept only when it fits an Indian
registration pattern. Detection still runs if the OCR package is missing.
"""
from __future__ import annotations

import os
import re
import tempfile
import threading

import cv2
import numpy as np

MODEL_NAME = "cct-s-v2-global-model"
_INDIA = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}
_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0", "D": "0"}


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def _nms(boxes, iou_thr=0.4):
    boxes = sorted(boxes, key=lambda b: -b[1])
    kept = []
    for box, score in boxes:
        if all(_iou(box, k[0]) < iou_thr for k in kept):
            kept.append((box, score))
    return kept


def _bright(bgr):
    if bgr is None or bgr.size == 0:
        return 0.0
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def _take(shape, x, y, bw, bh, prior, bgr):
    h, w = shape[:2]
    if bw < 28 or bh < 10 or bw > 0.75 * w or bh > 0.45 * h:
        return None
    aspect = bw / float(bh)
    if aspect < 1.25 or aspect > 7.2:
        return None
    if y + bh > h or x + bw > w:
        return None
    crop = bgr[y:y + bh, x:x + bw]
    if _bright(crop) < 90:
        return None
    cy = (y + bh / 2) / h
    score = prior + (0.55 if cy > 0.45 else -0.15 if cy < 0.22 else 0.1)
    return [x, y, x + bw, y + bh], score


def _yellow_boxes(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # commercial plates are a strong yellow; sandy gravel is a weak one
    mask = cv2.inRange(hsv, (12, 110, 110), (40, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        hit = _take(bgr.shape, x, y, bw, bh, 1.6, bgr)
        if hit:
            out.append(hit)
    return out


def _canny_boxes(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 170)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        peri = cv2.arcLength(c, True)
        if peri < 40:
            continue
        approx = cv2.approxPolyDP(c, 0.045 * peri, True)
        if not 4 <= len(approx) <= 8:
            continue
        hit = _take(bgr.shape, x, y, bw, bh, 1.15, bgr)
        if hit:
            out.append(hit)
    return out


def _text_boxes(bgr):
    """Rows of dark character strokes on a lighter background: what a plate is, whatever
    its colour or lighting. Works on a copy scaled to a fixed width so the kernels fit."""
    h, w = bgr.shape[:2]
    if w < 40 or h < 20:
        return []
    S = 640.0 / w
    img = cv2.resize(bgr, (640, max(20, int(h * S))), interpolation=cv2.INTER_AREA if S < 1 else cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
    gx = np.abs(cv2.Sobel(hat, cv2.CV_32F, 1, 0, ksize=3))
    if gx.max() <= 0:
        return []
    gx = (255 * gx / gx.max()).astype(np.uint8)
    gx = cv2.GaussianBlur(gx, (5, 5), 0)
    gx = cv2.morphologyEx(gx, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)))
    _, th = cv2.threshold(gx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = cv2.erode(th, None, iterations=1)
    th = cv2.dilate(th, None, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ih, iw = gray.shape
    out = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 24 or bh < 8 or bw > 0.6 * iw or bh > 0.35 * ih:
            continue
        aspect = bw / float(bh)
        if not (1.3 <= aspect <= 7.0):
            continue
        roi = gray[y:y + bh, x:x + bw]
        strokes = hat[y:y + bh, x:x + bw]
        # plate: lighter background, strong dark strokes, many vertical transitions
        bg_light = float(np.percentile(roi, 70))
        ink = float(strokes.mean())
        cols = (strokes > max(20, np.percentile(strokes, 75))).any(axis=0)
        transitions = int(np.count_nonzero(np.diff(cols.astype(np.int8)) != 0))
        if bg_light < 60 or ink < 6 or transitions < 6:
            continue
        cy = (y + bh / 2) / ih
        score = 1.2 + min(1.0, ink / 40.0) + min(0.6, transitions / 30.0) + (0.3 if cy > 0.45 else -0.2)
        X1, Y1 = int(x / S), int(y / S)
        X2, Y2 = int(min(w, (x + bw) / S)), int(min(h, (y + bh) / S))
        out.append(([X1, Y1, X2, Y2], score))
    return out


def propose_plates(bgr, limit=8):
    """Return up to `limit` (crop, box, score) candidates, best first."""
    h, w = bgr.shape[:2]
    found = []
    for box, score in _nms(_text_boxes(bgr) + _yellow_boxes(bgr) + _canny_boxes(bgr)):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        x1 = int(max(0, x1 - 0.08 * bw))
        y1 = int(max(0, y1 - 0.12 * bh))
        x2 = int(min(w, x2 + 0.08 * bw))
        y2 = int(min(h, y2 + 0.12 * bh))
        if x2 - x1 < 24 or y2 - y1 < 12:
            continue
        found.append((bgr[y1:y2, x1:x2], (x1, y1, x2, y2), score))
    found.sort(key=lambda f: -f[2])
    return found[:limit]


STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK",
    "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB", "BH",
}
# letters an OCR model commonly mixes up on plates
_CONFUSABLE = {"E": "LF", "L": "EI", "F": "EP", "K": "XR", "X": "K", "H": "NM", "N": "HM", "M": "NH",
               "O": "DQ", "D": "O", "Q": "O", "V": "UY", "U": "V", "P": "RF", "R": "PB", "B": "R",
               "C": "G", "G": "C", "T": "I", "I": "TL", "J": "I", "W": "V"}


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


def format_indian(raw):
    """Return (canonical plate, substitutions) or None.

    The segmentation that rewrites the fewest characters wins, so a string is
    not forced into a plate by swapping half of it. The state code must be a real one.
    """
    text = _fix_state(re.sub(r"[^A-Z0-9]", "", (raw or "").upper()))
    best = None
    for dlen in (1, 2):
        for llen in (1, 2, 3):
            if 2 + dlen + llen + 4 != len(text):
                continue
            slots = [(0, 2, "L"), (2, 2 + dlen, "D"), (2 + dlen, 2 + dlen + llen, "L"),
                     (2 + dlen + llen, len(text), "D")]
            out, fixes, ok = [], 0, True
            for a, b, kind in slots:
                table = _LETTER if kind == "L" else _DIGIT
                for ch in text[a:b]:
                    fixed = table.get(ch, ch)
                    if fixed != ch:
                        fixes += 1
                    if kind == "L" and not fixed.isalpha():
                        ok = False
                    if kind == "D" and not fixed.isdigit():
                        ok = False
                    out.append(fixed)
            if not ok:
                continue
            plate = "".join(out)
            if not _INDIA.match(plate) or plate[:2] not in STATE_CODES:
                continue
            if best is None or fixes < best[0]:
                best = (fixes, plate)
    if best is None or best[0] > 1:
        return None
    return best[1], best[0]


def edit_distance(a, b):
    """Levenshtein distance, small strings only."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def display_plate(norm):
    m = re.match(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{4})$", norm or "")
    if not m:
        return norm or ""
    return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"


def _fit_aspect(bgr, target=2.0):
    """Pad so width/height is near the OCR model's 2:1 input. Stops wide plates being squashed."""
    h, w = bgr.shape[:2]
    aspect = w / max(h, 1)
    if abs(aspect - target) < 0.15:
        return bgr
    if aspect > target:
        need = int(round(w / target))
        pad = max(0, need - h)
        top = pad // 2
        return cv2.copyMakeBorder(bgr, top, pad - top, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    need = int(round(h * target))
    pad = max(0, need - w)
    left = pad // 2
    return cv2.copyMakeBorder(bgr, 0, 0, left, pad - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def _tighten(bgr):
    """Crop down to the dark character rows so the model sees the plate, not the bumper."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    energy = hat.mean(axis=1)
    if energy.size == 0 or energy.max() < 1:
        return bgr
    rows = np.where(energy > energy.max() * 0.35)[0]
    if len(rows) < 3:
        return bgr
    y1, y2 = int(rows[0]), int(rows[-1])
    pad = max(2, int(0.2 * (y2 - y1 + 1)))
    return bgr[max(0, y1 - pad):min(bgr.shape[0], y2 + pad + 1)]


def _plate_like(text):
    if len(text) < 5 or len(text) > 12:
        return False
    return any(c.isalpha() for c in text) and any(c.isdigit() for c in text)


def _line_cut(gray):
    h = gray.shape[0]
    if h < 18:
        return None
    dark = gray < np.percentile(gray, 42)
    ink = dark.mean(axis=1)
    lo, hi = int(h * 0.30), int(h * 0.70)
    if hi <= lo:
        return None
    cut = lo + int(np.argmin(ink[lo:hi]))
    if ink[cut] > np.percentile(ink, 40):
        return None
    return cut


class PlateReader:
    def __init__(self, model=MODEL_NAME):
        self.model_name = model
        self._ocr = None
        self._lock = threading.Lock()
        self.error = None
        self.status = "pending"

    def start(self):
        threading.Thread(target=self._ensure, daemon=True).start()

    def _ensure(self):
        if self._ocr is not None:
            return True
        with self._lock:
            if self._ocr is not None:
                return True
            if self.error:
                return False
            try:
                from fast_plate_ocr import LicensePlateRecognizer
                self._ocr = LicensePlateRecognizer(self.model_name, device="cpu")
                self.status = "ready"
                return True
            except Exception as e:  # noqa: BLE001 - a missing model must not stop detection
                self.error = str(e)
                self.status = "unavailable"
                return False

    def _ocr_bgr(self, bgr):
        if bgr is None or bgr.size == 0 or bgr.shape[0] < 8 or bgr.shape[1] < 12:
            return "", 0.0
        bgr = _fit_aspect(bgr, 2.0)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(path, bgr)
            pred = self._ocr.run(path, return_confidence=True)[0]
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        text = re.sub(r"[^A-Z0-9]", "", (getattr(pred, "plate", "") or "").upper())
        probs = getattr(pred, "char_probs", None)
        conf = float(np.mean(probs)) if probs is not None and len(probs) else 0.0
        if not text:
            conf = 0.0
        return text, conf

    def _variants(self, crop):
        """Several looks at the same crop. A number is kept only when two of them agree."""
        reads = [self._ocr_bgr(crop), self._ocr_bgr(_tighten(crop))]
        h, w = crop.shape[:2]
        if h >= 22 and w / max(h, 1) < 2.4:
            cut = _line_cut(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) or (h // 2)
            if 0.28 * h < cut < 0.72 * h:
                top, tc = self._ocr_bgr(crop[:cut + 1])
                bot, bc = self._ocr_bgr(crop[cut:])
                if top and bot:
                    reads.append((top + bot, (tc + bc) / 2))
        return reads

    def read(self, bgr):
        """Return {norm, display, conf, crop}. display is set only when two reads agree.

        crop is set whenever a plate-like region was found, so the photo is kept
        even when the number itself is not confident.
        """
        if bgr is None or getattr(bgr, "size", 0) == 0:
            self.status = "unreadable"
            return None
        if not self._ensure():
            self.status = "unavailable"
            return None
        best_num = None
        best_crop = None
        best_crop_score = -1.0
        raws = []                                   # every plate-like text seen, for cross-frame voting
        for crop, _box, prior in propose_plates(bgr):
            reads = self._variants(crop)
            raws.extend(_fix_state(t) for t, c in reads if c >= 0.5 and _plate_like(t))
            agreed = {}
            for text, conf in reads:
                if conf < 0.7:
                    continue
                parsed = format_indian(text)
                if not parsed or parsed[1] != 0:
                    continue
                agreed.setdefault(parsed[0], []).append(conf)
            for norm, confs in agreed.items():
                if len(confs) < 2:
                    continue
                conf = float(sum(confs) / len(confs))
                score = conf + prior * 0.02
                if best_num is None or score > best_num[0]:
                    best_num = (score, norm, conf, crop)
            for text, conf in reads:
                if not _plate_like(text):
                    continue
                score = len(text) * conf + prior
                if score > best_crop_score:
                    best_crop, best_crop_score = crop, score
        if best_num:
            _, norm, conf, crop = best_num
            self.status = "ok"
            return {"norm": norm, "display": display_plate(norm), "conf": round(conf, 3), "crop": crop,
                    "raws": raws}
        if best_crop is not None:
            self.status = "unreadable"
            return {"norm": None, "display": None, "conf": None, "crop": best_crop, "raws": raws}
        self.status = "unreadable"
        return {"norm": None, "display": None, "conf": None, "crop": None, "raws": raws} if raws else None
