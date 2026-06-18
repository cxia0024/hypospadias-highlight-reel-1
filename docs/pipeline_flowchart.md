# Hypospadias Highlight Reel — Pipeline Flowchart

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     HYPOSPADIAS HIGHLIGHT REEL PIPELINE                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────┐
    │                    📹  INPUT: Raw Surgical Video                │
    │               (Full-length hypospadias repair, e.g. 2 hrs)     │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      FRAME SAMPLING                            │
    │                                                                │
    │   video_io.sample_frames()                                     │
    │   • Extracts frames at configurable FPS (default: 1/sec)       │
    │   • 2-hr video → ~7,200 frames (vs 432,000 at 60fps)          │
    │                                                                │
    │   Output: list[Frame]  ─  image + index + timestamp            │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
          ════════════════════════════════════════════
          ║          PHASE 1: OBJECT DETECTION       ║
          ════════════════════════════════════════════
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                     YOLOv8 (Fine-Tuned)                        │
    │                                                                │
    │   Detects per frame:                                           │
    │   ┌─────────────────────┐  ┌──────────────────────┐            │
    │   │    INSTRUMENTS      │  │   ANATOMY             │            │
    │   │  • scissors         │  │  • urethral_plate     │            │
    │   │  • forceps          │  │  • glans              │            │
    │   │  • needle_driver    │  │  • skin_flap          │            │
    │   │  • suture           │  │  • dartos_flap        │            │
    │   │  • catheter         │  │                       │            │
    │   └─────────────────────┘  └──────────────────────┘            │
    │                                                                │
    │   Output: list[FrameDetections]  ─  bboxes + classes + conf    │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
          ════════════════════════════════════════════
          ║     PHASE 1.5: SURGICAL PHASE RECOGNITION ║
          ════════════════════════════════════════════
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              ResNet-18  ──►  Bidirectional GRU                  │
    │              (spatial)       (temporal context)                 │
    │                                                                │
    │   Classifies each frame into one of 8 surgical phases:         │
    │                                                                │
    │   ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌─────────────┐ │
    │   │  1. Prep │─►│2. Deglove│─►│ 3. Urethral│─►│4. Tubular-  │ │
    │   │          │  │          │  │    Plate    │  │   ization   │ │
    │   │  wt: 0.2 │  │  wt: 0.5 │  │  Incision  │  │  wt: 1.0   │ │
    │   │          │  │          │  │  wt: 0.8   │  │  ★ highest  │ │
    │   └──────────┘  └──────────┘  └────────────┘  └─────────────┘ │
    │         │              │              │              │         │
    │         ▼              ▼              ▼              ▼         │
    │   ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌─────────────┐ │
    │   │5. Water- │─►│6. Glans- │─►│  7. Skin   │─►│ 8. Dressing │ │
    │   │ proofing │  │  plasty  │  │  Closure   │  │             │ │
    │   │  wt: 0.7 │  │  wt: 0.9 │  │  wt: 0.6  │  │   wt: 0.1  │ │
    │   └──────────┘  └──────────┘  └────────────┘  └─────────────┘ │
    │                                                                │
    │   Post-processing:                                             │
    │   ┌────────────────┐  ┌───────────────┐  ┌──────────────────┐  │
    │   │   Temporal      │  │   Monotonic   │  │    Argmax        │  │
    │   │   Smoothing     │─►│   Prior       │─►│    Selection     │  │
    │   │  (15-frame avg) │  │ (no backward  │  │  (best phase +   │  │
    │   │                 │  │  transitions) │  │   confidence)    │  │
    │   └────────────────┘  └───────────────┘  └──────────────────┘  │
    │                                                                │
    │   Fallback: instrument-phase priors + temporal position        │
    │   (e.g. needle_driver + suture + urethral_plate → tubular.)   │
    │                                                                │
    │   Output: list[PhaseLabeledFrame]  ─  phase + confidence       │
    └──────────────┬──────────────────────────────────────────────────┘
                   │
          ════════════════════════════════════════════
          ║    PHASE 2: FRAME-LEVEL CAPTIONING       ║
          ════════════════════════════════════════════
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Claude Vision API                            │
    │                                                                │
    │   Per frame, sends to Claude:                                  │
    │   ┌───────────────┐  ┌─────────────────────────────────┐       │
    │   │  Frame Image  │  │  "Describe the surgical scene   │       │
    │   │  (base64 JPG) │ +│   showing needle_driver, suture,│       │
    │   │               │  │   urethral_plate"               │       │
    │   └───────────────┘  └─────────────────────────────────┘       │
    │                            │                                   │
    │                            ▼                                   │
    │   "Needle driver placing interrupted suture through            │
    │    urethral plate edges over catheter stent"                   │
    │                                                                │
    │   Output: list[CaptionedFrame]  ─  caption + timestamp         │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                       CLIP GROUPING                            │
    │                                                                │
    │   video_io.group_into_clips()                                  │
    │   Sliding window: 10s duration, 5s stride                      │
    │                                                                │
    │   ─────┬──────────┬──────────┬──────────┬──────                │
    │        │  clip 1  │          │          │                       │
    │        │──────────│──────────│          │                       │
    │        │     │  clip 2  │          │                            │
    │        │     │──────────│──────────│                            │
    │        │          │  clip 3  │          │                       │
    │   ─────┴──────────┴──────────┴──────────┴──────                │
    │        0s        5s        10s       15s       20s             │
    │                                                                │
    │   Overlap ensures no action at a boundary is missed            │
    │                                                                │
    │   Output: list[Clip]  ─  grouped frames per window             │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
          ════════════════════════════════════════════
          ║    PHASE 3: CLIP-LEVEL CAPTIONING        ║
          ════════════════════════════════════════════
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Claude Messages API                         │
    │                                                                │
    │   Per clip, three operations:                                  │
    │                                                                │
    │   ┌─ 1. Phase Resolution ────────────────────────────────────┐ │
    │   │  Majority vote across frame phase labels                 │ │
    │   │  → dominant phase + avg confidence                       │ │
    │   └──────────────────────────────────────────────────────────┘ │
    │                                                                │
    │   ┌─ 2. Caption Synthesis ───────────────────────────────────┐ │
    │   │  System: "Summarize into one coherent surgical action"   │ │
    │   │  User:   "Surgical phase: Tubularization                 │ │
    │   │           - [42.0s] Needle driver grasping suture...     │ │
    │   │           - [43.0s] Suture being pulled through...       │ │
    │   │           - [44.0s] ..."                                 │ │
    │   │  → "Surgeon tubularizes urethral plate with running      │ │
    │   │     subepithelial suture over 8Fr catheter stent"        │ │
    │   └──────────────────────────────────────────────────────────┘ │
    │                                                                │
    │   ┌─ 3. Importance Scoring ──────────────────────────────────┐ │
    │   │                                                          │ │
    │   │  ┌──────────────────┐  weight                            │ │
    │   │  │ Object diversity │   40%   unique instruments / 5     │ │
    │   │  │ Detection density│   40%   detections per frame / 3   │ │
    │   │  │ Phase weight     │   20%   e.g. tubularization = 1.0  │ │
    │   │  └──────────────────┘                                    │ │
    │   │  score = 0.4·diversity + 0.4·density + 0.2·phase_wt     │ │
    │   └──────────────────────────────────────────────────────────┘ │
    │                                                                │
    │   Output: list[CaptionedClip]                                  │
    │           ─ caption + score + surgical_phase + confidence       │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
          ════════════════════════════════════════════
          ║        PHASE 4: SYNTHESIS                ║
          ════════════════════════════════════════════
                                  │
                   ┌──────────────┼──────────────┐
                   │              │              │
                   ▼              ▼              ▼
    ┌──────────────────┐ ┌──────────────┐ ┌──────────────────────┐
    │  CLIP SELECTION  │ │    REPORT    │ │   HIGHLIGHT REEL     │
    │                  │ │  GENERATION  │ │     ASSEMBLY         │
    │  Three modes:    │ │              │ │                      │
    │                  │ │  Sections:   │ │  video_io.write_     │
    │  importance_score│ │  • Summary   │ │  highlight_reel()    │
    │  ┌────────────┐  │ │  • Surgical  │ │                      │
    │  │ Sort by    │  │ │    Phases    │ │  Seeks to each       │
    │  │ score desc │  │ │    (table)   │ │  selected segment    │
    │  │ Fill 120s  │  │ │  • Key       │ │  in source video,    │
    │  └────────────┘  │ │    Moments   │ │  copies frames to    │
    │                  │ │  • Instrument│ │  output MP4          │
    │  phase_balanced  │ │    Usage (%) │ │                      │
    │  ┌────────────┐  │ │  • Full      │ │                      │
    │  │ Equal time │  │ │    Timeline  │ │                      │
    │  │ per phase, │  │ │    (w/phase  │ │                      │
    │  │ best clips │  │ │     labels)  │ │                      │
    │  │ in each    │  │ │              │ │                      │
    │  └────────────┘  │ │              │ │                      │
    │                  │ │              │ │                      │
    │  uniform         │ │              │ │                      │
    │  ┌────────────┐  │ │              │ │                      │
    │  │ Chrono-    │  │ │              │ │                      │
    │  │ logical    │  │ │              │ │                      │
    │  └────────────┘  │ │              │ │                      │
    │                  │ │              │ │                      │
    │  Filter:         │ │              │ │                      │
    │  score ≥ 0.6     │ │              │ │                      │
    │  total ≤ 120s    │ │              │ │                      │
    └──────────────────┘ └──────┬───────┘ └──────────┬───────────┘
                                │                    │
                                ▼                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                         OUTPUT                                 │
    │                                                                │
    │   output/                                                      │
    │   ├── highlight_reel.mp4    ~2 min of key surgical moments     │
    │   └── report.md             structured operative summary       │
    │                                                                │
    │   ┌─────────────────────────────────────────────────────────┐   │
    │   │  # Surgical Highlight Report                           │   │
    │   │                                                        │   │
    │   │  ## Summary                                            │   │
    │   │  Hypospadias repair spanning 7200s. Selected 12 clips  │   │
    │   │  for 118s highlight reel.                              │   │
    │   │                                                        │   │
    │   │  ## Surgical Phases                                    │   │
    │   │  | Phase          | Time     | Duration | Importance | │   │
    │   │  |----------------|----------|----------|------------| │   │
    │   │  | Tubularization | 1800-3000| 1200s    | 0.87       | │   │
    │   │  | Glansplasty    | 3000-4200| 1200s    | 0.82       | │   │
    │   │  | ...            | ...      | ...      | ...        | │   │
    │   │                                                        │   │
    │   │  ## Key Moments                                        │   │
    │   │  - 1842s [Tubularization] (0.91): Surgeon begins...   │   │
    │   │  - 3150s [Glansplasty] (0.88): Glans wings...         │   │
    │   └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║  CLI:  python cli.py --input surgery.mp4 --config config/default.yaml      ║
║  Env:  ANTHROPIC_API_KEY required for Phases 2 & 3                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```
