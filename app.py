"""
WasteWatch dashboard - upload a video, draw the watched zone, run littering detection,
and review evidence (vehicle snapshot + action GIF) for every waste-throwing event.

Run:  python app.py      then open http://localhost:5000
"""
import csv
import io
import json
import os
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

from detector import Config, WasteDetector
from plate_cam import auto_offset, is_stream, resolve_offset

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, "uploads")
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3  # 2 GB uploads

JOBS: dict[str, dict] = {}
RUN_LOCK = threading.Lock()
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".ts"}


# ------------------------------------------------------------------ helpers
STREAM_EXT = ".stream"   # a saved live camera: a text file holding its rtsp:// URL


def source_of(path):
    """What OpenCV should open: the file itself, or the URL inside a saved .stream entry."""
    if path and path.endswith(STREAM_EXT):
        with open(path) as f:
            return f.read().strip()
    return path


def display_name(path):
    name = os.path.basename(path).split("__", 1)[1]
    return name[:-len(STREAM_EXT)] if name.endswith(STREAM_EXT) else name


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
    keys = ("id", "video_id", "video_name", "plate_video_id", "plate_video_name", "plate_offset",
            "plate_offset_auto", "plate_sync", "live", "status", "progress", "total", "fps", "error", "started",
            "finished", "settings", "zone")
    pub = {k: j.get(k) for k in keys} | {"events": j.get("events", [])}
    if pub.get("settings"):
        pub["settings"] = {k: v for k, v in pub["settings"].items() if k != "plate_source"}  # may hold a password
    return pub


def load_job_from_disk(jid):
    p = os.path.join(OUTPUTS, jid, "job.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        j = json.load(f)
    ep = os.path.join(OUTPUTS, jid, "events.json")
    if os.path.exists(ep):
        with open(ep) as f:
            j["events"] = json.load(f)
    return j


def save_job_meta(j):
    with open(os.path.join(OUTPUTS, j["id"], "job.json"), "w") as f:
        json.dump({k: v for k, v in job_public(j).items() if k != "events"}, f, indent=2)


def get_job(jid):
    return JOBS.get(jid) or load_job_from_disk(jid)


# ------------------------------------------------------------------ worker
def run_job(job, video_path, cfg):
    with RUN_LOCK:
        job["status"] = "running"
        job["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_job_meta(job)
        try:
            log = lambda m: print(f"[{job['id']}] {m}")
            if cfg.plate_source and not job.get("live"):
                # sync the two clips from the traffic in them (NVR export names can be wrong)
                job["status"] = "syncing"
                save_job_meta(job)
                off, method, score = resolve_offset(video_path, cfg.plate_source,
                                                    job["video_name"], job["plate_video_name"], log=log)
                cfg.plate_offset = off + job.get("plate_adjust", 0.0)
                job["plate_offset"] = round(cfg.plate_offset, 2)
                job["plate_sync"] = {"method": method, "score": score, "found": off,
                                     "adjust": job.get("plate_adjust", 0.0)}
                job["settings"]["plate_offset"] = cfg.plate_offset
                log(f"plate camera sync: {cfg.plate_offset:+.2f}s ({method}, match {score})")
                job["status"] = "running"
                save_job_meta(job)
            det = WasteDetector(cfg, os.path.join(OUTPUTS, job["id"]), log=log)
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
        plate_before_sec=float(s.get("plate_after", 12.0)),
        plate_after_sec=float(s.get("plate_after", 12.0)),
        live_fps=float(s.get("live_fps", 10.0)),
    )
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    os.makedirs(os.path.join(OUTPUTS, jid), exist_ok=True)
    job = {"id": jid, "video_id": vid, "video_name": display_name(path),
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
    for jid in sorted(os.listdir(OUTPUTS), reverse=True):
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


@app.get("/files/<jid>/<path:fp>")
def job_file(jid, fp):
    return send_from_directory(os.path.join(OUTPUTS, secure_filename(jid)), fp)


def _csv_bytes(events):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["event", "waste_time", "culprit_type", "vehicle_class", "track_id", "plate",
                "plate_confidence", "plate_source", "plate_frames_agree", "plate_cam_match",
                "plate_cam_ambiguous", "plate_overview_read", "culprit_seen_at", "gif_from", "gif_to",
                "waste_box_x1y1x2y2", "folder"])
    for e in events:
        gr = e.get("gif_range") or ["", ""]
        pc = e.get("platecam") or {}
        w.writerow([e["id"], e["time_str"], e["culprit_type"], e.get("vehicle_class") or "",
                    e.get("track_id") if e.get("track_id") is not None else "",
                    e.get("plate_text") or "", e.get("plate_conf") if e.get("plate_conf") is not None else "",
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


# ------------------------------------------------------------------ event log
@app.get("/events")
def events_page():
    return render_template("events.html")


def _event_when(job, ev):
    """(sortable timestamp, date, time text) for an event.

    Live: the event time is a clock time on the day the job ran. Clips: the job's
    date and the position in the clip, because the clip's real recording time is
    not reliably known.
    """
    started = job.get("started") or ""
    day = started[:10] if started else job["id"][:8]
    if job.get("live"):
        t = (ev.get("time_str") or "")[:8]
        stamp = f"{day} {t}"
        return stamp, day, ev.get("time_str") or ""
    stamp = f"{started or day} +{ev.get('time', 0):09.2f}"
    return stamp, day, f"clip {ev.get('time_str', '')}"


def _all_events():
    out = []
    for jid in sorted(os.listdir(OUTPUTS), reverse=True):
        if not os.path.isdir(os.path.join(OUTPUTS, jid)):
            continue
        j = get_job(jid)
        if not j:
            continue
        for ev in j.get("events") or []:
            stamp, day, when = _event_when(j, ev)
            pc = ev.get("platecam") or {}
            out.append({
                "job": jid, "id": ev["id"], "dir": ev["dir"], "stamp": stamp, "date": day, "when": when,
                "video": j.get("video_name"), "plate_video": j.get("plate_video_name"),
                "live": bool(j.get("live")), "job_status": j.get("status"),
                "culprit_type": ev.get("culprit_type"), "vehicle_class": ev.get("vehicle_class"),
                "track_id": ev.get("track_id"), "plate_text": ev.get("plate_text"),
                "plate_norm": ev.get("plate_norm"), "plate_conf": ev.get("plate_conf"),
                "plate_from": ev.get("plate_from"), "plate_votes": ev.get("plate_votes"),
                "plate_status": ev.get("plate_status"), "plate_overview": ev.get("plate_overview"),
                "ambiguous": bool(pc.get("ambiguous")), "match_method": pc.get("method"),
                "platecam_status": pc.get("status"), "hidden": bool(ev.get("hidden")),
                "files": ev.get("files") or {}, "gif_ready": bool(ev.get("gif_ready")),
                "time_str": ev.get("time_str"), "culprit_time_str": ev.get("culprit_time_str"),
                "gif_range": ev.get("gif_range"),
            })
    out.sort(key=lambda e: e["stamp"], reverse=True)
    return out


def _filter_events(evs, args):
    q = (args.get("q") or "").strip().upper().replace(" ", "")
    d_from, d_to = args.get("from") or "", args.get("to") or ""
    kind = args.get("kind") or ""
    vclass = args.get("vclass") or ""
    res = []
    for e in evs:
        if q:
            hay = " ".join(str(x or "") for x in (e["plate_norm"], e["plate_text"], e["video"], e["job"],
                                                   e["vehicle_class"], e["plate_overview"])).upper().replace(" ", "")
            if q not in hay:
                continue
        if d_from and e["date"].replace("-", "") < d_from.replace("-", ""):
            continue
        if d_to and e["date"].replace("-", "") > d_to.replace("-", ""):
            continue
        if kind == "plate" and not e["plate_text"]:
            continue
        if kind == "noplate" and e["plate_text"]:
            continue
        if kind == "review" and not (e["ambiguous"] or not e["plate_text"]):
            continue
        if vclass and (e["vehicle_class"] or e["culprit_type"]) != vclass:
            continue
        res.append(e)
    return res


@app.get("/api/events")
def api_events():
    evs = _filter_events(_all_events(), request.args)
    limit = min(int(request.args.get("limit", 500)), 5000)
    plates = {}
    for e in evs:
        if e["plate_norm"]:
            plates[e["plate_norm"]] = plates.get(e["plate_norm"], 0) + 1
    return jsonify(events=evs[:limit], total=len(evs), repeat_plates=plates)


def _event_record(jid, eid):
    j = get_job(jid)
    if not j:
        return None, None
    for ev in j.get("events") or []:
        if int(ev["id"]) == int(eid):
            return j, ev
    return j, None


def _event_summary(j, ev):
    stamp, day, when = _event_when(j, ev)
    pc = ev.get("platecam") or {}
    lines = [
        "WasteWatch - littering event evidence",
        "",
        f"Analysis        : {j['id']}  ({'live cameras' if j.get('live') else 'recorded clips'})",
        f"Date            : {day}",
        f"Waste at rest   : {when}",
        f"Overview video  : {j.get('video_name') or ''}",
        f"Plate camera    : {j.get('plate_video_name') or 'not used'}",
        f"Culprit         : {ev.get('culprit_type')} {ev.get('vehicle_class') or ''} (track #{ev.get('track_id')})",
        f"Culprit at spot : {ev.get('culprit_time_str') or ''}",
        f"Number plate    : {ev.get('plate_text') or 'not read'}",
        f"Plate read from : {ev.get('plate_from') or ''}  {ev.get('plate_votes') or ''}",
        f"Plate confidence: {ev.get('plate_conf') if ev.get('plate_conf') is not None else ''}",
    ]
    if ev.get("plate_overview"):
        lines.append(f"Overview read   : {ev['plate_overview']}")
    if pc:
        lines.append(f"Plate cam match : {pc.get('status')}  method={pc.get('method') or ''}  "
                     f"score={pc.get('match_score', '')}  {'AMBIGUOUS - check the photos' if pc.get('ambiguous') else ''}")
    lines += ["", "Files:"]
    names = {"gif": "action.gif - the throw, from before it happened to after",
             "vehicle": "vehicle.jpg - best photo of the culprit (overview camera)",
             "scene": "scene.jpg - full frame when the culprit was at the spot",
             "waste": "waste.jpg - close-up of the waste",
             "plate": "plate crop the number was read from",
             "pc_scene": "plate_cam.jpg - plate camera frame, matched vehicle boxed",
             "pc_vehicle": "plate_cam_vehicle.jpg - matched vehicle in the plate camera",
             "pc_plate": "plate crop from the plate camera"}
    seen = set()
    for k, fn in (ev.get("files") or {}).items():
        if fn in seen:
            continue
        seen.add(fn)
        lines.append(f"  {fn:28s} {names.get(k, '').split(' - ', 1)[-1]}")
    return "\n".join(lines) + "\n"


def _add_event_to_zip(z, j, ev, prefix=""):
    root = os.path.join(OUTPUTS, j["id"], ev["dir"])
    if os.path.isdir(root):
        for f in sorted(os.listdir(root)):
            full = os.path.join(root, f)
            if os.path.isfile(full):
                z.write(full, prefix + f)
    z.writestr(prefix + "event.json", json.dumps(ev, indent=2))
    z.writestr(prefix + "summary.txt", _event_summary(j, ev))


@app.get("/api/job/<jid>/event/<int:eid>/evidence.zip")
def event_zip(jid, eid):
    jid = secure_filename(jid)
    j, ev = _event_record(jid, eid)
    if not ev:
        abort(404)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        _add_event_to_zip(z, j, ev)
    mem.seek(0)
    plate = (ev.get("plate_norm") or "noplate")
    return send_file(mem, mimetype="application/zip", as_attachment=True,
                     download_name=f"evidence_{jid}_event{int(eid):02d}_{plate}.zip")


@app.post("/api/events/evidence.zip")
def events_zip():
    """Selected events in one ZIP: a folder per event, plus a CSV of all of them."""
    items = (request.get_json(force=True) or {}).get("items") or []
    if not items:
        return jsonify(error="Select at least one event"), 400
    mem = io.BytesIO()
    rows = []
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for it in items[:500]:
            jid = secure_filename(str(it.get("job", "")))
            j, ev = _event_record(jid, it.get("id", -1))
            if not ev:
                continue
            folder = f"{jid}_event{int(ev['id']):02d}_{ev.get('plate_norm') or 'noplate'}/"
            _add_event_to_zip(z, j, ev, folder)
            rows.append((j, ev))
        z.writestr("events.csv", _log_csv(rows))
    if not rows:
        return jsonify(error="None of the selected events were found"), 404
    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True,
                     download_name=f"waste_evidence_{time.strftime('%Y%m%d-%H%M')}_{len(rows)}events.zip")


def _log_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date", "time", "analysis", "event", "overview_video", "plate_video", "culprit_type",
                "vehicle_class", "plate", "plate_confidence", "plate_source", "plate_frames_agree",
                "plate_cam_ambiguous", "plate_cam_method", "overview_plate_read", "folder"])
    for j, ev in rows:
        _stamp, day, when = _event_when(j, ev)
        pc = ev.get("platecam") or {}
        w.writerow([day, when, j["id"], ev["id"], j.get("video_name") or "", j.get("plate_video_name") or "",
                    ev.get("culprit_type") or "", ev.get("vehicle_class") or "", ev.get("plate_text") or "",
                    ev.get("plate_conf") if ev.get("plate_conf") is not None else "", ev.get("plate_from") or "",
                    ev.get("plate_votes") or "", "yes" if pc.get("ambiguous") else "", pc.get("method") or "",
                    ev.get("plate_overview") or "", f"{j['id']}/{ev['dir']}"])
    return buf.getvalue().encode()


@app.get("/api/events.csv")
def events_csv():
    evs = _filter_events(_all_events(), request.args)
    rows = []
    for e in evs:
        j, ev = _event_record(e["job"], e["id"])
        if ev:
            rows.append((j, ev))
    return Response(_log_csv(rows), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=waste_event_log_{time.strftime('%Y%m%d-%H%M')}.csv"})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    print(f"WasteWatch running at http://localhost:{a.port}")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)
