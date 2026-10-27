# WasteWatch: Littering Detection Dashboard

Upload a CCTV video, mark the dumping area (drain, roadside, vacant plot), and the system
flags every time someone on a bike or car throws waste. For each event you get:

| File | What it is |
|---|---|
| `vehicle.jpg` | Best snapshot of the culprit vehicle with its rider (largest, clearest frame, usually showing the plate) |
| `action.gif` | The full action. Starts **4 s before** the waste landed, so the throw itself is included, even if the vehicle has already left the frame |
| `scene.jpg` | Full frame at the moment the culprit was next to the spot |
| `waste.jpg` | Close-up of the dropped waste |
| `events.json` / `events.csv` | Timeline of all events |

## Install & run

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                                        # open http://localhost:5000
```
The YOLO model (`yolo11n.pt`, ~5 MB) downloads automatically on first run.
GPU: install the CUDA build of PyTorch and it is used automatically (CPU ≈ 5 fps for 720p, GPU 30+ fps).

CLI (no dashboard):
```bash
python detector.py video.mp4 --out outputs/test --zone "[[0.55,0.22],[0.78,0.22],[1,0.45],[1,1],[0.22,1]]"
```

## How it works

1. **Track every road user**: YOLO11 + ByteTrack follow persons, bikes, motorcycles, cars, buses and trucks.
   A rider sitting on a two-wheeler is merged with the bike into one evidence box. Each track
   keeps its full position history and its best snapshot, so the evidence survives after the vehicle leaves.
2. **Find new waste**: a background model of the watched zone looks for a *new object that appears and
   stays still* (default 0.8 s). Vehicles, people and their shadows are masked out. Moving shadows, and "ghosts"
   left when something moves away, are rejected (edge test).
3. **Attribute**: when waste is confirmed, the system looks **back** up to 4 s to find which vehicle/person
   was closest to that spot. Vehicles take priority over pedestrians.
4. **Evidence**: a rolling 15 s frame buffer builds the GIF from *before* the throw to *after* it,
   with the culprit boxed in red and the waste in yellow.

## Two cameras: overview + plate camera

The overview camera sees the throw. A second, zoomed **plate camera** on the exit path
reads the number. Nothing changes in how waste is detected; the plate camera only adds
the plate.

How it works (`plate_cam.py`):
1. When an event is confirmed, the culprit's time at the spot and the time it left the
   overview give a search window (1 s before → *Plate cam search after exit*, default 6 s, after).
2. YOLO runs on the plate camera in that window and follows every vehicle.
3. The culprit is the vehicle with the same **type** (two-wheeler vs car/bus/truck),
   seen at the right **time**, with the closest **colour** to the overview snapshot
   (rider's clothes count for two-wheelers).
4. Its plate is read on **every frame** it is in view (sharpest, largest first). A number
   is accepted only when **two or more frames agree**. A confident plate-camera read
   replaces the overview read; the overview read is kept in the CSV for comparison.

Each event then also has:

| File | What it is |
|---|---|
| `plate_cam.jpg` | Plate camera frame with the matched vehicle boxed |
| `plate_cam_vehicle.jpg` | The matched vehicle, close up |
| `plate_cam_plate.jpg` | The plate crop the number was read from |

`events.json` gets `plate_from` (`plate_cam` / `clip` / overview), `plate_votes`
(e.g. `3/12` frames agree) and a `platecam` block with the match score. `ambiguous: true`
means two similar vehicles passed together — check the photo before issuing a notice.

### Recorded clips (testing)
1. Export **both channels for the same time period** from the NVR (locally, not through the mobile app).
2. Upload the overview clip as usual, then pick/upload the plate clip under **Plate camera**.
3. Sync is automatic: when the analysis starts, the traffic in both clips is matched
   (status shows *Syncing cameras*). NVR export names are only a fallback — they can carry
   the export time instead of the recording time. If you still see the wrong vehicle, use
   **Adjust** (seconds; + means the plate clip started later).

### False-alert protection
- The first 3 s of every video/stream are used to learn the empty scene (median of frames),
  so a vehicle or its shadow at the start is not mistaken for waste. No events in that time.
- A "new object" that is the same ground under different light (a shadow moving off gravel,
  sun on existing litter) is rejected by a texture/brightness test.
- An object that is seen and then vanishes (thrown into bushes) only counts if it moved.
- Grass/flutter checks use only the changed pixels, so a bag landing in the grass is not
  dismissed as waving grass.

### Plate reading
- Plates are found by their character strokes (works for white, yellow, shaded or grey plates).
- The state code must be a real one; a single OCR look-alike (KE → KL) is corrected.
- The plate camera reads up to 40 frames of the matched vehicle. A number is accepted when
  two frames read it exactly, or one sure read is backed by near-identical reads
  (shown as e.g. `1 exact + 4 near of 40`).

CLI:
```bash
python detector.py overview.mp4 --plate plate.mp4 --out outputs/test --zone "[[...]]"
# --plate-offset 5.0   override the automatic sync
# --plate-adjust -0.4  fine-tune it
```

### Live cameras (RTSP)
Add both cameras under **Add a live camera (RTSP)**, pick the overview in step 1 and the
plate camera under *Plate camera*. Frames are time-stamped on arrival, so both cameras
must reach this PC over the LAN.
```bash
python detector.py "rtsp://user:pass@192.168.1.64:554/stream1" --plate "rtsp://user:pass@192.168.1.65:554/stream1"
```
VIGI cameras: main stream is usually `rtsp://user:pass@IP:554/stream1`. Live mode must keep
up with the camera. *Live processing fps* (default 10) sets how many overview frames per second
are analysed; on CPU use 6–8, on a GPU 25. The log warns when frames are still being dropped.
The plate camera is buffered separately at full resolution, 8 frames per second, for the last 25 s.

### Plate camera checklist
- Points along the near lane at vehicles **leaving** the dumping spot (rear plates).
- Plate at least **130 px wide** for cars and **~100 px** for two-wheeler plates in the
  capture zone. Below ~100 px the OCR guesses and the two-frame rule rejects it.
- Shutter 1/1000 s (day), 1/500 s (night), IR at night, H.265 (not H.265+), 25 fps, max bitrate.
- Set both cameras to NTP / the NVR's time so their clocks match.

## Tuning (Detection settings panel)

| Setting | When to change |
|---|---|
| Watched zone | Always draw one in production. Excludes far-away traffic and parked objects |
| Waste must stay still | Raise (1.5 s) on windy scenes with fluttering leaves/paper |
| Change sensitivity | Lower (20) for small/dark waste at night; raise (40) for noisy cameras |
| Culprit look-back | Raise for slow-moving scenes where waste is found long after the throw |
| GIF pre-roll | How many seconds before the waste landed the clip starts |
| Detect every N frames | 2 = roughly 2x faster on CPU, slightly less accurate tracking |
| Model | `yolo11s/m` detect small/distant vehicles better (slower) |

## Next steps worth adding
- Alerts (WhatsApp/SMS/email) on each event
- A dedicated plate detector model if the plate camera view has many vehicles at once
