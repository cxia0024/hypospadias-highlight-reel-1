# Hypospadias Surgical Video Highlight Reel Framework

A four-phase pipeline for generating summary highlight reels from full-length intraoperative hypospadias repair videos.

## Pipeline Phases

1. **Object Detection** — Detect surgical instruments, anatomical structures, and key entities in video frames using a fine-tuned object detection model.
2. **Frame-Level Captioning** — Generate descriptive captions for individual frames conditioned on detected objects and spatial context.
3. **Clip-Level Captioning** — Aggregate frame captions into temporally coherent clip-level narratives that capture surgical actions and transitions.
4. **Report & Highlight Reel Synthesis** — Produce a structured surgical report and compile a highlight reel by selecting the most informative clips.

## Project Structure

```
├── config/
│   └── default.yaml            # Pipeline configuration
├── src/
│   ├── pipeline.py             # Orchestrates the four phases
│   ├── phase1_detection.py     # Object detection
│   ├── phase2_frame_caption.py # Frame-level captioning
│   ├── phase3_clip_caption.py  # Clip-level captioning
│   ├── phase4_synthesis.py     # Report + highlight reel
│   ├── video_io.py             # Video reading/writing utilities
│   └── models/
│       └── __init__.py
├── cli.py                      # CLI entry point
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python cli.py --config config/default.yaml --input video.mp4 --output output/
```
