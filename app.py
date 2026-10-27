"""
WasteWatch dashboard - upload a video, draw the watched zone, run littering detection,
and review evidence (vehicle snapshot + action GIF) for every waste-throwing event.

Run:  python app.py      then open http://localhost:5000
"""
import csv
import io
import json
import os
import re
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict

import cv2
from flask import Flask, Response, abort, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from urllib.parse import urlparse

from v1 import detector as v1_detector
from v1 import plate_cam as v1_plate_cam
from v2 import detector as v2_detector
from v2 import plate_cam as v2_plate_cam
from v2.plate_cam import auto_offset, is_stream

# The dashboard runs either version, for demos and comparison.
#   v1/ - plate camera searched once per event; classic plate finder, one OCR model;
#         sync from the traffic (file names as fallback)
#   v2/ - plate camera read continuously; trained plate detector, multi-frame fusion,
#         three OCR models voting per character; sync from the cameras' burned-in clocks
VERSIONS = {
    "v1": {"label": "v1", "detector": v1_detector, "sync": v1_plate_cam.resolve_offset},
    "v2": {"label": "v2", "detector": v2_detector, "sync": v2_plate_cam.resolve_offset},
}

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, "uploads")
OUTPUTS = os.path.join(BASE, "outputs")
SOURCE = os.path.join(BASE, "source")     # sample pair folders, one recorded pair per folder
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3  # 2 GB uploads

JOBS: dict[str, dict] = {}
RUN_LOCK = threading.Lock()
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".ts"}


# ------------------------------------------------------------------ helpers
STREAM_EXT = ".stream"   # a saved live camera: a text file holding its rtsp:// URL
REF_EXT = ".ref"         # a clip used in place (pair folder): a text file holding its path
# NVR channel of the wide camera; the other channel of a pair is the zoomed plate camera
WIDE_CHANNEL = os.environ.get("WW_WIDE_CHANNEL", "004")


def source_of(path):
    """What OpenCV should open: the file itself, the URL inside a saved .stream entry, or
    the clip a .ref entry points to."""
    if path and (path.endswith(STREAM_EXT) or path.endswith(REF_EXT)):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return path


def display_name(path):
    name = os.path.basename(path).split("__", 1)[1]
    for ext in (STREAM_EXT, REF_EXT):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def _channel(name):
    """NVR channel in an export name, e.g. 'VIGI NVR2016H(UN)(004)_...' -> '004'."""
    m = re.findall(r"\((\d{2,4})\)", os.path.basename(name))
    return m[-1] if m else None


def assign_roles(names):
    """Which of two clips is the wide camera and which the zoomed plate camera.
    Returns (wide index, zoomed index, how it was decided)."""
    ch = [_channel(n) for n in names]
    if all(ch) and ch[0] != ch[1]:
        if WIDE_CHANNEL in ch:
            w = ch.index(WIDE_CHANNEL)
        else:
            w = 0 if int(ch[0]) < int(ch[1]) else 1
        return w, 1 - w, f"camera channel ({ch[w]} = wide, {ch[1 - w]} = zoomed)"
    low = [n.lower() for n in names]
    for w in (0, 1):
        if any(k in low[w] for k in ("wide", "overview")) or any(k in low[1 - w] for k in ("zoom", "plate")):
            return w, 1 - w, "file names"
    return 0, 1, "not clear from the names: check, and use Swap if wrong"


def register_clip(abs_path):
    """Make a clip on disk selectable without copying it (a .ref entry). Reuses an entry
    for the same file. Returns the same info as an upload, or None if unreadable."""
    abs_path = os.path.abspath(abs_path)
    for f in os.listdir(UPLOADS):
        if f.endswith(REF_EXT) and "__" in f:
            p = os.path.join(UPLOADS, f)
            try:
                with open(p, encoding="utf-8") as fh:
                    if os.path.abspath(fh.read().strip()) == abs_path:
                        vid = f.split("__", 1)[0]
                        _, _, info = video_info(p)
                        return dict(video_id=vid, name=display_name(p), **info)
            except OSError:
                continue
    vid = uuid.uuid4().hex[:10]
    # same name style as an uploaded file
    p = os.path.join(UPLOADS, f"{vid}__{secure_filename(os.path.basename(abs_path))}{REF_EXT}")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(abs_path)
    ok, frame, info = video_info(p)
    if not ok:
        os.remove(p)
        return None
    cv2.imwrite(os.path.join(UPLOADS, f"{vid}.jpg"), frame)
    return dict(video_id=vid, name=display_name(p), **info)


def _videos_in(folder):
    return sorted(f for f in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in VIDEO_EXT)


def find_stream(url):
    """A live camera already saved with this exact URL, if any."""
    for f in os.listdir(UPLOADS):
        if not (f.endswith(STREAM_EXT) and "__" in f):
            continue
        path = os.path.join(UPLOADS, f)
        try:
            with open(path, encoding="utf-8") as fh:
                saved = fh.read().strip()
        except OSError:
            continue
        if saved == url:
            return path
    return None


def video_info(path):
    src = source_of(path)
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if is_stream(src) else cv2.VideoCapture(src)
    if is_stream(src):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
    ok, frame = cap.read()
    info = {
        "fps": round(cap.get(cv2.CAP_PROP_FPS) or 0, 2),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duration"] = round(info["frames"] / info["fps"], 1) if info["fps"] else 0
    return ok, frame, info


def find_video(vid):
    for f in os.listdir(UPLOADS):
        if f.startswith(vid + "__"):
            return os.path.join(UPLOADS, f)
    return None


def job_public(j):
    keys = ("id", "version", "video_id", "video_name", "plate_video_id", "plate_video_name", "plate_offset",
            "plate_offset_auto", "plate_sync", "live", "status", "progress", "total", "fps", "error", "started",
            "finished", "settings", "zone")
    pub = {k: j.get(k) for k in keys} | {"events": j.get("events", [])}
    if pub.get("settings"):
        pub["settings"] = {k: v for k, v in pub["settings"].items() if k != "plate_source"}  # may hold a password
    return pub


def load_job_from_disk(jid):
    """A finished (or interrupted) analysis from outputs/<jid>. A folder with only
    events.json (e.g. written by the command-line tool) is shown too."""
    jid = secure_filename(jid)
    folder = os.path.join(OUTPUTS, jid)
    p = os.path.join(folder, "job.json")
    ep = os.path.join(folder, "events.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                j = json.load(f)
        except (OSError, ValueError):
            return None
    elif os.path.exists(ep):
        j = {"id": jid, "status": "done", "video_name": jid,
             "finished": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ep)))}
    else:
        return None
    j["id"] = jid
    if os.path.exists(ep):
        try:
            with open(ep) as f:
                j["events"] = json.load(f)
        except (OSError, ValueError):
            j["events"] = []
    # saved as running, but not running in this dashboard: it was cut off by a restart
    if j.get("status") in ("queued", "syncing", "running") and jid not in JOBS:
        j["status"] = "interrupted"
    return j


def save_job_meta(j):
    with open(os.path.join(OUTPUTS, j["id"], "job.json"), "w") as f:
        json.dump({k: v for k, v in job_public(j).items() if k != "events"}, f, indent=2)


def get_job(jid):
    return JOBS.get(jid) or load_job_from_disk(jid)


# ------------------------------------------------------------------ activity log
LOG_KINDS = (("[match]", "match"), ("[plate]", "plate"), ("plate camera", "plate"), ("[event", "event"),
             ("sync", "sync"), ("[rider]", "event"), ("[baseline]", "info"))


def _log_kind(msg):
    low = msg.lower()
    for key, kind in LOG_KINDS:
        if key in low:
            return kind
    return "info"


def job_logger(job):
    """Log function for one analysis: printed, kept for the dashboard, and saved to log.txt."""
    job.setdefault("_log", [])
    lock = threading.Lock()
    path = os.path.join(OUTPUTS, job["id"], "log.txt")

    def log(msg):
        msg = " ".join(str(msg).split("\n")).replace("\t", " ")
        print(f"[{job['id']}] {msg}")
        with lock:
            line = {"n": len(job["_log"]), "t": time.strftime("%H:%M:%S"), "kind": _log_kind(msg), "msg": msg}
            job["_log"].append(line)
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{line['t']}\t{line['kind']}\t{msg}\n")
    return log


def read_log(jid):
    j = JOBS.get(jid)
    if j and "_log" in j:
        return j["_log"]
    p = os.path.join(OUTPUTS, secure_filename(jid), "log.txt")
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for n, raw in enumerate(f):
            parts = raw.rstrip("\n").split("\t", 2)
            if len(parts) == 3:
                out.append({"n": n, "t": parts[0], "kind": parts[1], "msg": parts[2]})
    return out


# ------------------------------------------------------------------ worker
def run_job(job, video_path, cfg):
    with RUN_LOCK:
        job["status"] = "running"
        job["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_job_meta(job)
        try:
            log = job_logger(job)
            if cfg.plate_source and not job.get("live"):
                # sync the two clips from the traffic in them (NVR export names can be wrong)
                job["status"] = "syncing"
                save_job_meta(job)
                sync = VERSIONS[job.get("version", "v2")]["sync"]
                off, method, score = sync(source_of(video_path), cfg.plate_source,
                                          job["video_name"], job["plate_video_name"], log=log)
                cfg.plate_offset = off + job.get("plate_adjust", 0.0)
                job["plate_offset"] = round(cfg.plate_offset, 2)
                job["plate_sync"] = {"method": method, "score": score, "found": off,
                                     "adjust": job.get("plate_adjust", 0.0)}
                job["settings"]["plate_offset"] = cfg.plate_offset
                log(f"plate camera sync: {cfg.plate_offset:+.2f}s ({method}, match {score})")
                job["status"] = "running"
                save_job_meta(job)
            log(f"running {VERSIONS[job.get('version', 'v2')]['label']}")
            det = VERSIONS[job.get("version", "v2")]["detector"].WasteDetector(cfg, os.path.join(OUTPUTS, job["id"]),
                                                                                 log=log)
            job["_det"] = det
            job["events"] = det.events

            def prog(i, n, fps):
                job["progress"], job["total"], job["fps"] = i, n, round(fps, 1)

            det.run(source_of(video_path), progress=prog)
            job["status"] = "stopped" if det.stop_flag else "done"
        except Exception as e:  # noqa
            traceback.print_exc()
            job["status"], job["error"] = "error", str(e)
        job["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_job_meta(job)


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scene")
def scene():
    return render_template("scene.html")


@app.post("/api/upload")
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="No file"), 400
    name = secure_filename(f.filename) or "video.mp4"
    if os.path.splitext(name)[1].lower() not in VIDEO_EXT:
        return jsonify(error="Unsupported video type"), 400
    vid = uuid.uuid4().hex[:10]
    path = os.path.join(UPLOADS, f"{vid}__{name}")
    f.save(path)
    ok, frame, info = video_info(path)
    if not ok:
        os.remove(path)
        return jsonify(error="Could not read video"), 400
    cv2.imwrite(os.path.join(UPLOADS, f"{vid}.jpg"), frame)
    return jsonify(video_id=vid, name=name, **info)


def _pair_reply(infos, names, folder):
    w, z, how = assign_roles(names)
    return jsonify(folder=folder, wide=infos[w], zoom=infos[z], how=how)


@app.get("/api/folders")
def pair_folders():
    """Pair folders under source/: one wide and one zoomed clip each."""
    out = []
    if os.path.isdir(SOURCE):
        for d in sorted(os.listdir(SOURCE)):
            full = os.path.join(SOURCE, d)
            if os.path.isdir(full):
                vids = _videos_in(full)
                if len(vids) == 2:
                    w, z, how = assign_roles(vids)
                    out.append({"name": d, "wide": vids[w], "zoom": vids[z], "how": how})
    return jsonify(out)


@app.post("/api/folder")
def open_folder():
    """Use a pair folder under source/ in place: both clips become selectable, roles assigned."""
    name = os.path.basename((request.get_json(force=True) or {}).get("name") or "")
    folder = os.path.join(SOURCE, name)
    if not name or not os.path.isdir(folder):
        return jsonify(error="Folder not found"), 404
    vids = _videos_in(folder)
    if len(vids) != 2:
        return jsonify(error=f"The folder must hold exactly two videos (wide and zoomed); it has {len(vids)}"), 400
    infos = [register_clip(os.path.join(folder, v)) for v in vids]
    if not all(infos):
        return jsonify(error="Could not read one of the videos"), 400
    return _pair_reply(infos, vids, name)


@app.post("/api/upload_pair")
def upload_pair():
    """A folder chosen in the browser: its two videos are uploaded together, roles assigned."""
    files = [f for f in request.files.getlist("videos")
             if f and f.filename and os.path.splitext(f.filename)[1].lower() in VIDEO_EXT]
    if len(files) != 2:
        return jsonify(error=f"Pick a folder with exactly two videos (wide and zoomed); it has {len(files)}"), 400
    infos, names = [], []
    for f in files:
        name = secure_filename(os.path.basename(f.filename)) or "video.mp4"
        vid = uuid.uuid4().hex[:10]
        path = os.path.join(UPLOADS, f"{vid}__{name}")
        f.save(path)
        ok, frame, info = video_info(path)
        if not ok:
            os.remove(path)
            return jsonify(error=f"Could not read {name}"), 400
        cv2.imwrite(os.path.join(UPLOADS, f"{vid}.jpg"), frame)
        infos.append(dict(video_id=vid, name=name, **info))
        names.append(os.path.basename(f.filename))
    folder = os.path.dirname(files[0].filename.replace("\\", "/")) or "uploaded folder"
    return _pair_reply(infos, names, folder)


@app.post("/api/stream")
def add_stream():
    """Save a live camera (rtsp:// URL) so it can be picked like an uploaded video."""
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not is_stream(url):
        return jsonify(error="Enter an rtsp:// URL"), 400
    path = find_stream(url)
    created = path is None
    if created:
        label = (data.get("name") or "").strip() or f"{urlparse(url).hostname or 'camera'}"
        name = "Live " + (secure_filename(label) or "camera") + STREAM_EXT
        vid = uuid.uuid4().hex[:10]
        path = os.path.join(UPLOADS, f"{vid}__{name}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(url)
    else:
        vid = os.path.basename(path).split("__", 1)[0]
    ok, frame, info = video_info(path)
    if not ok:
        if created:
            os.remove(path)
        return jsonify(error="Could not read a frame from that camera. Check the URL, user and password."), 400
    cv2.imwrite(os.path.join(UPLOADS, f"{vid}.jpg"), frame)
    return jsonify(video_id=vid, name=display_name(path), live=True, **info)


@app.get("/api/videos")
def videos():
    out = []
    for f in sorted(os.listdir(UPLOADS), key=lambda x: -os.path.getmtime(os.path.join(UPLOADS, x))):
        if "__" in f:
            vid = f.split("__", 1)[0]
            out.append({"video_id": vid, "name": display_name(f"x/{f}"), "live": f.endswith(STREAM_EXT)})
    return jsonify(out)


@app.get("/api/video/<vid>")
def video_meta(vid):
    p = find_video(vid)
    if not p:
        abort(404)
    _, _, info = video_info(p)
    return jsonify(video_id=vid, name=display_name(p), live=p.endswith(STREAM_EXT), **info)


@app.get("/api/sync")
def sync_offset():
    """Seconds the plate clip started after the overview clip, from the NVR file names."""
    a, b = find_video(request.args.get("video_id", "")), find_video(request.args.get("plate_id", ""))
    if not a or not b:
        abort(404)
    if a.endswith(STREAM_EXT) or b.endswith(STREAM_EXT):
        return jsonify(offset=0.0, auto=True, live=True)
    off = auto_offset(display_name(a), display_name(b))
    return jsonify(offset=off if off is not None else 0.0, auto=off is not None, live=False)


@app.get("/api/video/<vid>/frame.jpg")
def video_frame(vid):
    p = os.path.join(UPLOADS, f"{secure_filename(vid)}.jpg")
    if not os.path.exists(p):
        abort(404)
    return send_file(p, mimetype="image/jpeg")


@app.post("/api/analyze")
def analyze():
    data = request.get_json(force=True)
    vid = data.get("video_id")
    path = find_video(vid) if vid else None
    if not path:
        return jsonify(error="Video not found"), 404
    if any(j["status"] in ("queued", "syncing", "running") for j in JOBS.values()):
        return jsonify(error="An analysis is already running. Stop it or wait for it to finish."), 409
    s = data.get("settings", {})
    version = data.get("version") if data.get("version") in VERSIONS else "v2"
    Config = VERSIONS[version]["detector"].Config
    # second camera
    pvid = data.get("plate_video_id") or None
    ppath = find_video(pvid) if pvid else None
    if pvid and not ppath:
        return jsonify(error="Plate camera video not found"), 404
    if ppath and ppath == path:
        return jsonify(error="Pick a different video for the plate camera"), 400
    live = path.endswith(STREAM_EXT)
    if ppath and ppath.endswith(STREAM_EXT) != live:
        return jsonify(error="Use two recorded clips, or two live cameras - not one of each"), 400
    off_auto = None if (not ppath or live) else auto_offset(display_name(path), display_name(ppath))
    plate_adjust = float(data.get("plate_adjust") or 0.0)
    plate_offset = plate_adjust          # clips: the real offset is worked out when the job starts
    cfg = Config(
        model=s.get("model", "yolo11n.pt"),
        conf=float(s.get("conf", 0.30)),
        detect_stride=int(s.get("stride", 1)),
        persist_sec=float(s.get("persist", 0.8)),
        pre_roll_sec=float(s.get("pre_roll", 4.0)),
        post_roll_sec=float(s.get("post_roll", 1.5)),
        lookback_sec=float(s.get("lookback", 4.0)),
        fragment_sec=float(s.get("fragment", 1.5)),
        diff_thresh=int(s.get("sensitivity", 30)),
        zone=data.get("zone") or None,
        plate_source=source_of(ppath) if ppath else None,
        plate_offset=plate_offset,
        # v1 searched this many seconds each side of the throw (its own default, 12 s);
        # v2 uses it as a margin around the culprit's time in view
        plate_before_sec=float(s.get("plate_after", 4.0)) if version == "v2" else 12.0,
        plate_after_sec=float(s.get("plate_after", 4.0)) if version == "v2" else 12.0,
        live_fps=float(s.get("live_fps", 10.0)),
    )
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    os.makedirs(os.path.join(OUTPUTS, jid), exist_ok=True)
    job = {"id": jid, "version": version, "video_id": vid, "video_name": display_name(path),
           "plate_video_id": pvid, "plate_video_name": display_name(ppath) if ppath else None,
           "plate_offset": round(plate_offset, 3) if ppath else None,
           "plate_offset_auto": off_auto, "plate_adjust": plate_adjust, "plate_sync": None, "live": live,
           "status": "queued", "progress": 0, "total": 0, "fps": 0, "events": [],
           "settings": asdict(cfg), "zone": cfg.zone, "error": None}
    JOBS[jid] = job
    save_job_meta(job)
    threading.Thread(target=run_job, args=(job, path, cfg), daemon=True).start()
    return jsonify(job_id=jid)


@app.get("/api/jobs")
def jobs():
    out = []
    folders = [d for d in os.listdir(OUTPUTS) if os.path.isdir(os.path.join(OUTPUTS, d))]

    def started(d):            # analyses are named <date>-<time>-<id>: newest first by that
        m = re.match(r"^(\d{8}-\d{6})", d)
        return m.group(1) if m else time.strftime("%Y%m%d-%H%M%S",
                                                   time.localtime(os.path.getmtime(os.path.join(OUTPUTS, d))))
    for jid in sorted(folders, key=started, reverse=True):
        j = get_job(jid)
        if j:
            pj = job_public(j)
            pj["event_count"] = len(pj.pop("events") or [])
            out.append(pj)
    return jsonify(out[:50])


@app.get("/api/job/<jid>")
def job_status(jid):
    j = get_job(jid)
    if not j:
        abort(404)
    return jsonify(job_public(j))


@app.post("/api/job/<jid>/stop")
def job_stop(jid):
    j = JOBS.get(jid)
    if j and j.get("_det"):
        j["_det"].stop_flag = True
    return jsonify(ok=True)


@app.get("/api/job/<jid>/preview.jpg")
def job_preview(jid):
    j = JOBS.get(jid)
    det = j.get("_det") if j else None
    if not det or det.preview is None:
        abort(404)
    return Response(det.preview, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/job/<jid>/log")
def job_log(jid):
    """Activity log lines after `since` (sync, events, camera matching, plate reads)."""
    since = int(request.args.get("since", 0) or 0)
    lines = read_log(jid)
    return jsonify(lines=lines[since:], next=len(lines))


@app.get("/files/<jid>/<path:fp>")
def job_file(jid, fp):
    return send_from_directory(os.path.join(OUTPUTS, secure_filename(jid)), fp)


def _csv_bytes(events):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["event", "waste_time", "clock_time", "plate_cam_clock", "culprit_type", "vehicle_class", "track_id", "plate",
                "plate_best_guess_unconfirmed", "plate_unclear_characters", "plate_type", "plate_confidence", "plate_source", "plate_frames_agree", "plate_cam_match",
                "plate_cam_ambiguous", "plate_overview_read", "culprit_seen_at", "gif_from", "gif_to",
                "waste_box_x1y1x2y2", "folder"])
    for e in events:
        gr = e.get("gif_range") or ["", ""]
        pc = e.get("platecam") or {}
        w.writerow([e["id"], e["time_str"], e.get("clock") or "", pc.get("frame_clock") or pc.get("frame_time") or "",
                    e["culprit_type"], e.get("vehicle_class") or "",
                    e.get("track_id") if e.get("track_id") is not None else "",
                    e.get("plate_text") or "", e.get("plate_guess") or "", e.get("plate_unclear") or "", e.get("plate_type") or "", e.get("plate_conf") if e.get("plate_conf") is not None else "",
                    e.get("plate_from") or "", e.get("plate_votes") or "",
                    pc.get("match_score", ""), "yes" if pc.get("ambiguous") else "",
                    e.get("plate_overview") or "",
                    e.get("culprit_time_str", ""), gr[0], gr[1],
                    " ".join(str(int(v)) for v in e["waste_box"]), e["dir"]])
    return buf.getvalue().encode()


@app.get("/api/job/<jid>/events.csv")
def job_csv(jid):
    j = get_job(jid)
    if not j:
        abort(404)
    return Response(_csv_bytes(j.get("events", [])), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=waste_events_{jid}.csv"})


@app.get("/api/job/<jid>/report.zip")
def job_zip(jid):
    j = get_job(jid)
    if not j:
        abort(404)
    root = os.path.join(OUTPUTS, jid)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, files in os.walk(root):
            for f in files:
                full = os.path.join(dp, f)
                z.write(full, os.path.relpath(full, root))
        z.writestr("events.csv", _csv_bytes(j.get("events", [])))
    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True,
                     download_name=f"waste_evidence_{jid}.zip")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    print(f"WasteWatch running at http://localhost:{a.port}")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)
