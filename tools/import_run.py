"""Add a command-line analysis to the dashboard's Previous analyses.

    python -m tools.import_run <run folder> --version v2 --pair source/2026-10-26_15-22-54 \
        [--log run.log] [--seconds 75]

<run folder> is the --out folder of `python -m v1.detector` / `python -m v2.detector`.
It is copied into outputs/<date-time-id>/ with a job.json in the same format the
dashboard writes, so it opens like any other analysis. Existing analyses are not touched.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import asdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import app  # noqa: E402  (dashboard helpers: register_clip, video_info, log kinds)
from werkzeug.utils import secure_filename  # noqa: E402

_OFFSET = re.compile(r"plate camera (?:offset|sync): ([+-]?\d+(?:\.\d+)?)s \((.+?), match ([\d.]+)\)")


def _clip(path, name):
    """The clip as a selectable video, or just its name when the file is gone."""
    if path and os.path.exists(path):
        info = app.register_clip(path)
        if info:
            return info
    return {"video_id": None, "name": name, "frames": 0}


def import_run(run_dir, version, pair=None, log_file=None, seconds=None, wide_name=None, zoom_name=None):
    ev_path = os.path.join(run_dir, "events.json")
    if not os.path.exists(ev_path):
        raise SystemExit(f"no events.json in {run_dir}")
    finished = os.path.getmtime(ev_path)
    started = finished - (seconds or 0)
    # already imported? (same run folder) -> skip, so running this twice adds nothing
    for d in os.listdir(app.OUTPUTS):
        p = os.path.join(app.OUTPUTS, d, "job.json")
        if os.path.exists(p):
            with open(p) as f:
                if json.load(f).get("imported_from") == os.path.abspath(run_dir):
                    print(f"already in Previous analyses as {d}")
                    return d
    wide = zoom = None
    if pair:
        pair_dir = os.path.join(BASE, pair) if not os.path.isabs(pair) else pair
        vids = app._videos_in(pair_dir)
        w, z, _how = app.assign_roles(vids)
        wide = _clip(os.path.join(pair_dir, vids[w]), vids[w])
        zoom = _clip(os.path.join(pair_dir, vids[z]), vids[z])
    if wide_name:
        wide = _clip(None, wide_name)
    if zoom_name:
        zoom = _clip(None, zoom_name)
    frames = wide["frames"] if wide else 0

    jid = time.strftime("%Y%m%d-%H%M%S", time.localtime(started)) + "-" + uuid.uuid4().hex[:4]
    dst = os.path.join(app.OUTPUTS, jid)
    shutil.copytree(run_dir, dst)

    lines, sync = [], None
    if log_file and os.path.exists(log_file):
        with open(log_file, encoding="utf-8", errors="replace") as f:
            raw = f.read().replace("\r", "\n")
        for msg in raw.split("\n"):
            msg = msg.strip()
            if not msg or re.match(r"^\d+/\d+ frames", msg) or "Warning" in msg or "onnxruntime" in msg:
                continue
            m = _OFFSET.search(msg)
            if m:
                sync = (float(m.group(1)), m.group(2), float(m.group(3)))
            lines.append(msg)
    stamp = time.strftime("%H:%M:%S", time.localtime(started))
    with open(os.path.join(dst, "log.txt"), "w", encoding="utf-8") as f:
        for msg in lines:
            f.write(f"{stamp}\t{app._log_kind(msg)}\t{msg}\n")

    cfg = asdict(app.VERSIONS[version]["detector"].Config())
    cfg.pop("plate_source", None)
    if sync:
        cfg["plate_offset"] = sync[0]
    with open(ev_path) as f:
        n_events = len(json.load(f))
    job = {
        "id": jid, "version": version,
        "video_id": wide["video_id"], "video_name": secure_filename(wide["name"]),
        "plate_video_id": zoom["video_id"] if zoom else None,
        "plate_video_name": secure_filename(zoom["name"]) if zoom else None,
        "plate_offset": sync[0] if sync else None, "plate_offset_auto": None,
        "plate_sync": {"method": sync[1], "score": sync[2], "found": sync[0], "adjust": 0.0} if sync else None,
        "live": False, "status": "done",
        "progress": frames, "total": frames,
        "fps": round(frames / seconds, 1) if seconds and frames else 0,
        "error": None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finished)),
        "settings": cfg, "zone": None,
        "imported_from": os.path.abspath(run_dir),
    }
    with open(os.path.join(dst, "job.json"), "w") as f:
        json.dump(job, f, indent=2)
    print(f"added {jid}: {version} · {wide['name']}{' + ' + zoom['name'] if zoom else ''} · {n_events} event(s)")
    return jid


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add a command-line analysis to Previous analyses")
    ap.add_argument("run_dir")
    ap.add_argument("--version", choices=("v1", "v2"), required=True)
    ap.add_argument("--pair", help="pair folder the run used, e.g. source/2026-10-26_15-22-54")
    ap.add_argument("--wide", help="wide clip name, when its file is no longer there")
    ap.add_argument("--zoom", help="zoomed clip name, when its file is no longer there")
    ap.add_argument("--log", help="the run's console output, kept as its activity log")
    ap.add_argument("--seconds", type=float, help="how long the run took")
    a = ap.parse_args()
    if not a.pair and not a.wide:
        ap.error("give --pair or --wide")
    import_run(a.run_dir, a.version, a.pair, a.log, a.seconds, a.wide, a.zoom)
