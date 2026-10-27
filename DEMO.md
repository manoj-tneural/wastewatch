# Running a WasteWatch demo

The dashboard runs two versions on the same footage:

- **v1** (`v1/`): plate camera searched once per throw, classic plate finder with one OCR model,
  sync from the traffic.
- **v2** (`v2/`): plate camera read continuously, trained plate detector with multi-frame
  fusion and per-character voting (four plate colour schemes, unclear characters with their
  likely values), sync from the camera clocks.

Both versions use the same throw detection. The difference you can show is in matching
the vehicle across the two cameras and reading its plate, plus speed.

## Sample pairs

Each folder in `source/` holds one recorded pair from the same minute. It is named after
the wide camera's start time (camera clock). `(004)` is the wide camera, `(005)` the zoomed
plate camera.

| Folder | Length | Good for |
|---|---|---|
| `2026-10-26_15-22-54` | 12 s | **Best first demo.** Short. The pillion rider on the motorcycle with the green bag throws it onto the footpath at 15:23:03; plate KL 07 CL 2696 |
| `2026-10-26_15-23-59` | 20 s | Same staged bike; busier traffic |
| `2026-10-26_15-28-11` | 20 s | Trucks and cars; harsher light |
| `2026-10-26_15-21-37` | 15 s | Same staged bike |

## Before the audience arrives (once, about 15 minutes)

1. Open a terminal in `D:\wastewatch` and start the dashboard:
   ```
   python app.py
   ```
   Wait for `WasteWatch running at http://localhost:5000`.
2. Open **http://localhost:5000** in Chrome or Edge.
3. **Warm-up run.** The first analysis downloads and loads the models (plate detector,
   OCR, clock reader), which adds a minute or two. Run v2 once on
   `2026-10-26_15-22-54` (steps 1-7 below) so this does not happen in front of people.
4. **v1 results are already there.** **Previous analyses** holds a v1 run of every sample
   pair, so you do not need to run v1 live. That matters because v1 is slow: about 4 minutes
   for the 12 s pair, against 75 s for v2. To show a fresh v1 run anyway, start it before the
   audience arrives.

## The demo, step by step

1. **Pick the pair folder.** At the top of step 1, under *Pair folder*, choose the folder
   from **Sample pairs in source/…** (e.g. `2026-10-26_15-22-54`). Both cameras fill in:
   the wide clip as *Overview*, the zoomed clip as *Plate camera*. Nothing is copied.
   - For a folder somewhere else (a USB stick, another drive), click **Choose folder…**
     instead; its two videos are uploaded together.
2. **Check the roles.** The line under the folder says which file is wide and which is
   zoomed, and how that was decided (from the NVR camera channel: `(004)` wide, `(005)`
   zoomed). If they are the wrong way round, click **⇄ Swap**. The two cameras are synced
   automatically from the clocks burned into the pictures when the analysis starts.
3. **Watched zone.** Click **Left shoulder** (the footpath and verge on the left of this
   camera), or draw the dumping area by clicking points on the frame.
4. **Choose the version.** Next to *Start analysis*, click **v1** or **v2**.
5. **Start analysis.** Point out, as it runs:
   - the live view: vehicles near the litter zone, and the culprit boxed in red with its
     waste for a few seconds after a throw;
   - *Syncing cameras*, then the offset that was found;
   - events appearing in **Evidence** as they are detected.
6. **Match log.** One row per throw: the clock time, a snapshot of the throw, the same
   vehicle in the zoomed camera (vehicle and plate boxed), the plate image, and the OCR
   result. A confirmed plate is shown in full. A partly read plate shows its unclear
   characters highlighted, with their likely values and look-alikes (e.g.
   `10th: 6 (55%) or 0 (43%), looks like 3/8/9`). Until the plate camera has found the
   vehicle, the row says *matching in the plate camera…*; it fills in within seconds.
7. **How the plate was found.** On an event card, open *How this plate was found*. It
   shows the vehicle that was picked on the plate camera (out of how many passed at that
   time), how it scored on type, colour and timing, and what each plate reader said.
8. **Run the other version.** The videos stay selected. Switch to the other version and
   click **Start analysis** again.
9. **Compare.** Use **Previous analyses** (top right). Each entry is labelled `V1` or `V2`.
   Switch between the two runs of the same pair and compare:
   - how many plates are read, and whether they are right (check against the plate photo);
   - the match log, which v1 cannot fill in as fully;
   - processing time (the *Speed* figure, or the job times).
10. **Evidence pack.** **⬇ CSV** gives every event in a spreadsheet: clock time, plate,
    best guess and unclear characters, plate type. **⬇ All evidence (ZIP)** adds the photos,
    GIFs and the activity log.

## What to say, and not claim

- v2 reads plates from many frames of the zoomed camera and only reports a number as
  confirmed when the reads agree. When it is not sure, it says so and shows the likely
  characters. It does not guess silently.
- Throw detection is the same in both versions and is the next part being improved. On
  these clips it also raises events that are not throws. Be ready to say this if asked;
  do not present every event as a real throw.
- Night plates in this footage are unreadable: headlight glare and motion blur. That is a
  camera setting (shutter, IR, highlight compensation), not software.

## If something goes wrong

| Problem | Fix |
|---|---|
| Page does not open | Check the terminal still shows `WasteWatch running`; restart `python app.py` |
| "An analysis is already running" | Wait for it or click **■ Stop** |
| Plate camera shows "no footage for this time" | The two clips are not from the same time: use a pair folder, or re-pick the zoomed clip |
| Wide and zoomed the wrong way round | Click **⇄ Swap** under the folder. For a different NVR, start the dashboard with `set WW_WIDE_CHANNEL=001` (the wide camera's channel) first |
| "The folder must hold exactly two videos" | A pair folder holds one wide and one zoomed clip, nothing else |
| First run is slow | Models are loading: that is why the warm-up run is done in advance |
