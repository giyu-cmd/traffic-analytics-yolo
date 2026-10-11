# YOLO Traffic Analytics System: Design and Build Plan

PBL project · October 2026

## 1. Architecture and scope

### Goal

Build a video-based traffic analytics tool around YOLO that detects and classifies vehicles, tracks them with stable IDs, estimates traffic density and vehicle speed, flags wrong-way driving, and benchmarks the model choices behind it.

### In scope

- Vehicle detection and classification (car, bus, truck, motorcycle, bicycle, plus auto-rickshaw if the footage needs it)
- Consistent vehicle IDs through multi-object tracking (ByteTrack)
- Per-class vehicle counting and traffic density estimation
- Speed estimation using per-region pixel-to-metre calibration
- Wrong-way (opposite direction) violation detection
- Performance and accuracy comparison between YOLO versions and ResNet-based models

### Proposed out of scope (confirm with your guide)

- Number plate recognition, helmet detection and red-light violations
- Tracking vehicles across multiple cameras
- Deployment on edge hardware

### Pipeline

1. Video input (file, CCTV or RTSP stream)
2. Preprocessing (resize, region-of-interest crop)
3. YOLO detector (boxes and classes)
4. ByteTrack (persistent track IDs)
5. Per-track analytics: counting, density, speed, wrong-way detection
6. Outputs: annotated video, dashboard, violation log

Two supporting parts sit alongside the pipeline:

- Scene config: regions, count lines, lane direction and pixel-to-metre mappings for each camera. The analytics stage reads it.
- Benchmark harness: runs the same test videos through swappable detectors and trackers and records accuracy and speed.

### Modules at a glance

- Counting: counts each tracked vehicle once when it crosses a virtual line, per class.
- Density: number of vehicles in a region, normalised by the region's real length in metres, mapped to low, medium or high.
- Speed: converts each vehicle's ground position to metres and divides displacement by elapsed time.
- Wrong-way: compares each vehicle's movement direction with the allowed lane direction.

## 2. Architectural decisions

Each entry gives the choice and the reason in one line.

- Language: Python 3.10 or newer. It has the best ecosystem for computer vision and deep learning.
- Detector: Ultralytics YOLO (v8, v10, v11). One API covers training, validation, tracking and export, so swapping versions for the benchmark is a one-line change.
- Tracker: ByteTrack. It is built into Ultralytics, needs no separate appearance network, and also associates low-confidence boxes, which helps through occlusion. DeepSORT and BoT-SORT stay as comparison points in the benchmark.
- Classification baseline: ResNet-18 and ResNet-50 from torchvision. ResNet cannot locate vehicles, so it is compared on cropped vehicle images and in a two-stage pipeline.
- Non-YOLO detector baseline: Faster R-CNN with a ResNet-50 backbone from torchvision. It gives a standard two-stage detector to compare against.
- Video input and output: OpenCV. It is the standard tool for reading frames and drawing overlays.
- Calibration maths: OpenCV homography and simple two-point scale. Perspective makes one global pixel-to-metre value wrong.
- Config format: one YAML file per camera. It is human-readable, version-controllable and easy to load.
- Dashboard: Streamlit. It is the fastest route to a working live dashboard in Python.
- Storage: SQLite for events, with CSV export. A single file, queryable and easy to share.
- Evaluation tooling: Ultralytics validation for mAP, and py-motmetrics for MOTA, IDF1 and ID switches.
- Training compute: Colab or Kaggle GPU for fine-tuning, local machine for inference and the dashboard.

## 3. Build plan with inputs and outputs

Each phase's output is the next phase's input, so the order matters.

### Data contracts

These records are passed between phases. Fixing them early keeps modules independent.

- Detection record: frame\_id, timestamp\_s, class, confidence, x1, y1, x2, y2
- Track record: frame\_id, timestamp\_s, track\_id, class, confidence, bounding box, ground\_point (bottom-centre of the box, x and y in pixels)
- Event record: event\_id, type (count or violation), track\_id, class, frame\_id, timestamp\_s, region\_id, details (speed, direction angle), clip\_path

### Phase 0: Setup and data

- Input: raw traffic videos (own footage or a public dataset) and the list of target classes
- Work: create the repo structure (data, configs, src, results), extract frames, label or import labels, split train, validation and test by video rather than by frame to avoid leakage
- Output: YOLO-format dataset with data.yaml, a split manifest, and a fixed set of benchmark test videos
- Done when: every split contains all classes and the test videos are never used for training

### Phase 1: Detection and classification

- Input: the Phase 0 dataset and pretrained YOLO weights
- Work: fine-tune, validate, then run inference on the test videos
- Output: fine-tuned weights, detection records for each test video, validation metrics (mAP@0.5, mAP@0.5:0.95, precision and recall per class) and measured FPS
- Done when: all target classes are detected on the test videos and per-class mAP is recorded

### Phase 2: Tracking with ByteTrack

- Input: the detection stream (frames plus detections) and the tracker configuration
- Work: run ByteTrack on the detections, tune the confidence thresholds and track buffer, and check a short clip by hand for ID switches
- Output: track records, an annotated clip with IDs, and the ID-switch count on the hand-checked clip
- Done when: IDs stay stable through typical occlusion on the test clip and the tracker settings are saved

### Phase 3: Scene configuration and calibration tool

- Input: one reference frame per camera, user clicks on that frame, and real-world distances the user supplies
- Work: build the calibration tool described in Section 4
- Output: a camera config file with regions, count lines, lane directions and pixel-to-metre mappings, plus a preview overlay image
- Done when: a known real distance measured elsewhere in the frame is reproduced within the target error

### Phase 4: Counting and density

- Input: track records, count lines and regions from the camera config
- Work: count each track once per line crossing, per class. Density is the number of vehicles in a region divided by the region's length in metres, then mapped to low, medium or high using configurable thresholds
- Output: per-class counts per time window, density per region over time, and the density level label
- Done when: counts are compared against manual counts on the test videos and the accuracy percentage is recorded

### Phase 5: Speed estimation

- Input: track records, region mappings and the video frame rate
- Work: convert each ground point to metres using the mapping of the region it is in, compute displacement over a short window of frames, smooth it, and convert to km/h (metres per second multiplied by 3.6)
- Output: speed per vehicle written to the event table and drawn on the annotated video
- Done when: speed error (MAE) is measured against reference speeds from a source with known ground truth

### Phase 6: Wrong-way detection

- Input: track records and the lane direction vector of each region
- Work: compute each vehicle's movement vector over a window and compare it with the lane direction. Flag a violation when the movement is opposed for several consecutive frames and the vehicle has moved a minimum distance, so stopped vehicles are ignored
- Output: violation events with timestamp, track ID, saved video clip and a snapshot
- Done when: precision and recall are measured on clips containing wrong-way driving, and stationary vehicles produce no false flags
- Note: real wrong-way footage is rare. Playing a normal clip in reverse is a quick way to produce test material, but report it as synthetic

### Phase 7: Outputs layer

- Input: track records, events and metrics from earlier phases
- Output: annotated video (boxes, IDs, class, speed), a Streamlit dashboard (live counts, density, speed distribution, violation list) and SQLite plus CSV logs
- Done when: the dashboard runs end to end on a test video without manual steps

### Phase 8: Benchmark and evaluation

- Input: the fixed test videos and the candidate models: YOLOv8, v10 and v11 in several sizes, Faster R-CNN with ResNet-50, ResNet-18 and ResNet-50 classifiers, YOLO classification variants, and the trackers ByteTrack, DeepSORT and BoT-SORT
- Output: result tables
  - Detection: mAP@0.5, mAP@0.5:0.95, precision, recall, FPS, latency, parameter count
  - Classification: top-1 accuracy, per-class F1, confusion matrix, inference time
  - End-to-end pipeline: single-stage YOLO versus YOLO detect followed by ResNet classify, with class accuracy and total FPS
  - Tracking: MOTA, IDF1, ID switches
  - Modules: counting error, speed MAE, wrong-way precision and recall
- Fairness rules: same dataset split, input resolution, thresholds and hardware for every model, all models fine-tuned from standard pretrained weights for the same number of epochs, and FPS measured on one GPU with warm-up frames excluded

### Phase 9: Report and demo

- Input: all result tables, annotated clips and dashboard
- Output: final report and a short demo video

## 4. Pixel-to-metre configuration system

### Why one value is not enough

In a road camera view, one pixel covers less ground near the camera than far away, so the number of metres per pixel changes across the image. A single global value gives wrong speeds. The system therefore lets you define regions of the image, each with its own mapping.

### Two calibration methods per region

- Scale method (quick): click two points along the road and type the real distance between them. The tool stores metres per pixel for that region. This works for short, roughly flat segments.
- Homography method (accurate): click four corners of a rectangle on the road whose real width and length you know, such as a lane width and a marked length. The tool computes a transform that maps image points to top-down road coordinates in metres. This handles perspective properly.

Default to the homography method for any region with visible perspective, and use the scale method for quick checks.

### The configuration tool

A small tool with a reference frame loaded on screen, built with OpenCV windows first and optionally moved into Streamlit later.

1. Load a reference frame for the camera.
2. Draw a region polygon with clicks (for example near, middle and far sections of the road, or one polygon per carriageway).
3. Choose a method for that region and click the calibration points.
4. Type the real distances.
5. Optionally draw count lines and an arrow showing the allowed lane direction.
6. Preview an overlay: a metre grid drawn over the road and the measured distance of any line you draw.
7. Save everything to the camera's YAML file.

### Config file example

```yaml
camera_id: cam01
reference_frame: refs/cam01.jpg
fps: 30
regions:
  - id: near
    polygon: [[120,700],[1180,700],[1080,420],[260,420]]
    calibration:
      method: homography
      # clicked in order: near-left, near-right, far-right, far-left
      image_points: [[210,690],[1090,690],[1000,430],[330,430]]
      real_size_m: [7.0, 30.0]   # width across road, length along road
    lane_direction: [0, -1]      # allowed travel direction in image space
  - id: far
    polygon: [[260,420],[1080,420],[980,250],[360,250]]
    calibration:
      method: scale
      image_points: [[600,400],[600,300]]
      real_distance_m: 12.0
    lane_direction: [0, -1]
count_lines:
  - id: line_a
    points: [[200,600],[1100,600]]
```

### How the runtime uses it

- For each track, take the ground point (bottom-centre of the box), because the road surface is the plane being calibrated.
- Find which region polygon contains that point and apply that region's mapping.
- Convert positions to metres, compute displacement over a short window, divide by elapsed time from the frame rate, and convert to km/h.
- If a vehicle crosses from one region to another, compute each step with the region it is in at that frame and sum the distances.
- Use one lane direction per carriageway so two-way roads are handled by defining one region per direction.

### Validation and maintenance

- After calibrating, measure a known real distance elsewhere in the frame (for example a lane marking length) and compare it with the tool's result. A proposed target is within about 5 percent.
- Store the reference frame next to the config. If the camera is moved or zoomed, recalibrate, because the old mapping no longer applies.
- Keep one YAML file per camera so the same code runs on different footage.
- Known limitation: the bottom-centre of a bounding box is an approximation of road contact, and it is least accurate for tall vehicles seen from a high angle.

### Inputs and outputs of the tool

- Input: a reference frame, the user's clicks, and real-world distances
- Output: the camera YAML file, a preview overlay image with the metre grid, and a short calibration report showing the check-distance error

## 5. Open items to settle early

- Dataset: own footage or a public dataset. This decides how large the ResNet and YOLO classification comparison can be.
- Ground truth for speed: how reference speeds will be obtained for the MAE measurement.
- Compute: confirm Colab or Kaggle GPU access for fine-tuning.
- Scope: confirm the out-of-scope items and the tracker choice with your guide.
- Cameras: whether the project uses a single camera view or several.
