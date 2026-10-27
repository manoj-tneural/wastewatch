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

## Two versions: v1 and v2

The dashboard runs either one (switch next to *Start analysis*), so the same clips can be
compared. Both share the throw detection and the general fixes (faster track search,
clean live view, camera-clock times, matching status).

| | v1 (`v1/`) | v2 (`v2/`) |
|---|---|---|
| Plate camera | searched once per event, after it | read continuously; vehicles kept in memory |
| Plate finding | classic (character strokes, colour, edges) | trained plate detector |
| Plate reading | one OCR model, two frames must agree | multi-frame fusion, three OCR models, per-character vote; v1 reader as second opinion |
| Plate colours | grey only | white, green (EV), yellow (taxi), black (rental) |
| Partial plates | not shown | unclear characters with likely values |
| Camera sync | traffic, file names as fallback | camera clocks, fine-tuned by traffic |
| Overview plate reads | yes | no when a plate camera is connected (it always wins) |

CLI: `python -m v1.detector …` or `python -m v2.detector …` (same options).

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
python -m v2.detector video.mp4 --out outputs/test --zone "[[0.55,0.22],[0.78,0.22],[1,0.45],[1,1],[0.22,1]]"
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

How it works (`v2/plate_cam.py`):
1. The plate camera runs all the time in its own thread. Every vehicle that passes is
   tracked on every frame; its sharpest plate crops and best photos are kept **in memory
   only**, and it gets a quick read as it leaves the view. Live, vehicles are forgotten
   after 90 s (*plate_retention_sec*); nothing is written to disk unless the vehicle is
   matched to a littering event.
2. When an event is confirmed, the culprit is picked from the vehicles that passed while
   it was in the overview (± *Plate cam margin*, default 4 s): same **type** (two-wheeler
   vs car/bus/truck), right **time**, closest **colour** to the overview snapshot (rider's
   clothes count for two-wheelers).
3. Its plate gets a thorough read (see *Plate reading*). The original reader also reads
   the same frames as a second opinion. If the two disagree, the plate is reported as a
   best guess, not as the number.

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
2. Put the two clips in one folder. Under **Pair folder**, pick it from *Sample pairs in
   source/…* (folders inside `source/`, used in place) or click **Choose folder…** (uploads
   both). The wide and zoomed clips are told apart by the NVR channel in the file name
   (`WW_WIDE_CHANNEL`, default `004`, is the wide camera); **⇄ Swap** fixes it if needed.
   You can still upload the overview clip and pick the plate clip separately.
3. Sync is automatic: when the analysis starts, the clock burned into each camera's
   picture is read (to the frame, from when the seconds digit changes), then fine-tuned by
   matching the traffic in both clips (status shows *Syncing cameras*). NVR export names
   are only a last resort — they can carry the export time instead of the recording time. If you still see the wrong vehicle, use
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
- A trained plate detector (YOLOv9, `open-image-models`) finds the plate on each vehicle
  view. The camera's own text overlays are masked so they are never read as a plate.
- The plate crops of one vehicle (up to 12) are aligned and median-stacked. A plate is
  sharp on only a few frames; the stacked image is clearer than any single frame.
- Two-line plates (usual on two-wheelers) are split and each line is read.
- Three OCR models read the stacked plate, a straightened copy, and slightly varied
  versions of both. The reads are voted **character by character** inside the Indian
  plate patterns (standard and BH series); the state code must be a real one and the
  district cannot be 0.
- A plate is reported as the number (`ok`) only when every character has a clear majority
  and the reads agree on its length. Otherwise it is a **best guess** (`uncertain`),
  shown with a warning and kept in its own CSV column, for a person to check against the
  plate photo.
- Look-alike characters on blurred plates (H/M; I/L/J/1/7; 3/6/8/9/0; 4/A) need an 85%
  majority to count as clear, against 70% for others, and an unclear one lists its
  look-alikes that fit the slot. Series letters are never I or O (Indian rule), so such a
  read becomes the look-alike letter.
- A vehicle's plate candidates are grouped by where they sit on it (rear middle, rear
  left, side); the most seen groups are read and the clearer result is kept, so a
  sticker or lettering elsewhere on the vehicle is not read as the plate.
- Partial reads show which characters are clear. An unclear character is shown as its
  most likely value (highlighted), with the other possible values and their share of the
  reads, e.g. `10th: 6 (55%) or 0 (43%)`. Only values that fit the slot are offered: a
  digit position gives digits, a letter position letters.
- The v1 single-frame reader (`v2/plates_legacy.py`) is kept as a second opinion.

CLI:
```bash
python -m v2.detector overview.mp4 --plate plate.mp4 --out outputs/test --zone "[[...]]"
# --plate-offset 5.0   override the automatic sync
# --plate-adjust -0.4  fine-tune it
```

### Live cameras (RTSP)
Add both cameras under **Add a live camera (RTSP)**, pick the overview in step 1 and the
plate camera under *Plate camera*. Frames are time-stamped on arrival, so both cameras
must reach this PC over the LAN.
```bash
python -m v2.detector "rtsp://user:pass@192.168.1.64:554/stream1" --plate "rtsp://user:pass@192.168.1.65:554/stream1"
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
