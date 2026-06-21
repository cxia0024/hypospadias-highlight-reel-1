# Hypospadias Highlight Reel — Pipeline Flowchart

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     HYPOSPADIAS HIGHLIGHT REEL PIPELINE                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────┐
    │                     INPUT: Raw Surgical Video                   │
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
          ║       PHASE 1: OBJECT DETECTION          ║
          ║           (ViT Multi-Label)              ║
          ════════════════════════════════════════════
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              Vision Transformer (ViT) Detector                 │
    │                                                                │
    │   Architecture (per frame):                                    │
    │                                                                │
    │   ┌─ Step 1: Patch Extraction (Eq 1) ────────────────────────┐ │
    │   │                                                          │ │
    │   │  Image A ∈ R^{224×224}                                   │ │
    │   │       │                                                  │ │
    │   │       ▼                                                  │ │
    │   │  ┌────┬────┬────┬─···─┐                                  │ │
    │   │  │ P₁ │ P₂ │ P₃ │    │  N = (224/16)² = 196 patches    │ │
    │   │  ├────┼────┼────┤    │  Each Pₖ ∈ R^{16×16×3}           │ │
    │   │  │ P₁₅│ P₁₆│ P₁₇│    │                                  │ │
    │   │  ├────┼────┼────┤    │  Pₖ = A[i:i+a−1, j:j+b−1]       │ │
    │   │  │    │    │    │    │  k = i × (m/b) + j                │ │
    │   │  └────┴────┴────┴─···─┘                                  │ │
    │   └──────────────────────────────────────────────────────────┘ │
    │                          │                                     │
    │                          ▼                                     │
    │   ┌─ Step 2: Embedding + Positional Encoding (Eq 2) ────────┐ │
    │   │                                                          │ │
    │   │  Flatten: P̃ₖ ∈ R^{768}  (16 × 16 × 3 = 768)            │ │
    │   │                                                          │ │
    │   │  Linear projection + learnable positional embedding:     │ │
    │   │  I = [[1, P̃₁], [2, P̃₂], ..., [196, P̃₁₉₆]]             │ │
    │   │    ∈ R^{1 × 196 × 768}                                  │ │
    │   └──────────────────────────────────────────────────────────┘ │
    │                          │                                     │
    │                          ▼                                     │
    │   ┌─ Step 3: Transformer Encoder ────────────────────────────┐ │
    │   │                                                          │ │
    │   │  12-layer Transformer Encoder                            │ │
    │   │  ┌────────────────────────────────────┐                  │ │
    │   │  │  Pre-Norm → Multi-Head Attention   │  × 12 layers    │ │
    │   │  │  (12 heads, dim=768)               │                  │ │
    │   │  │  Pre-Norm → MLP (3072, GELU)       │                  │ │
    │   │  │  + Residual connections + Dropout   │                  │ │
    │   │  └────────────────────────────────────┘                  │ │
    │   │         │                                                │ │
    │   │         ▼                                                │ │
    │   │  Global Average Pooling over 196 tokens → R^{768}        │ │
    │   └──────────────────────────────────────────────────────────┘ │
    │                          │                                     │
    │                          ▼                                     │
    │   ┌─ Step 4: Classification + Thresholding (Eq 3-4) ────────┐ │
    │   │                                                          │ │
    │   │  MLP Head: 768 → 384 → GELU → 9 logits                  │ │
    │   │                    │                                     │ │
    │   │                    ▼                                     │ │
    │   │  P = σ(logits)   ── sigmoid (multi-label, NOT softmax)   │ │
    │   │                    │                                     │ │
    │   │                    ▼                                     │ │
    │   │  O = { objectᵢ | Pᵢ > 0.5 }    (threshold per class)    │ │
    │   │                                                          │ │
    │   │  9 classes detected independently:                       │ │
    │   │  ┌─────────────────────┐  ┌──────────────────────┐       │ │
    │   │  │    INSTRUMENTS      │  │   ANATOMY            │       │ │
    │   │  │  • scissors         │  │  • urethral_plate    │       │ │
    │   │  │  • forceps          │  │  • glans             │       │ │
    │   │  │  • needle_driver    │  │  • skin_flap         │       │ │
    │   │  │  • suture           │  │  • dartos_flap       │       │ │
    │   │  │  • catheter         │  │                      │       │ │
    │   │  └─────────────────────┘  └──────────────────────┘       │ │
    │   └──────────────────────────────────────────────────────────┘ │
    │                                                                │
    │   ┌─ Training: Weighted BCE Loss (Eq 5) ─────────────────────┐ │
    │   │                                                          │ │
    │   │  L = −Σ wᵢ [yᵢ log σ(ŷᵢ) + (1−yᵢ) log(1−σ(ŷᵢ))]      │ │
    │   │                                                          │ │
    │   │  wᵢ = 1/fᵢ  (inverse class frequency)                   │ │
    │   │                                                          │ │
    │   │  Handles imbalance: forceps (common) vs dartos (rare)    │ │
    │   └──────────────────────────────────────────────────────────┘ │
    │                                                                │
    │   Fallback: YOLOv8 (set model: "yolov8" in config)            │
    │                                                                │
    │   Output: list[FrameDetections]  ─  classes + confidence       │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
          ════════════════════════════════════════════
          ║   PHASE 1.5: SURGICAL PHASE RECOGNITION  ║
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
          ║          (Claude Vision API)             ║
          ════════════════════════════════════════════
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Claude Vision API                            │
    │                   (Anthropic SDK)                               │
    │                                                                │
    │   Per frame, sends to Claude:                                  │
    │   ┌───────────────┐  ┌─────────────────────────────────┐       │
    │   │  Frame Image  │  │  "Describe the surgical scene   │       │
    │   │  (base64 JPG) │ +│   showing needle_driver, suture,│       │
    │   │               │  │   urethral_plate"               │       │
    │   └───────────────┘  └─────────────────────────────────┘       │
    │           │                                                    │
    │           ▼                                                    │
    │   anthropic.Anthropic().messages.create(                       │
    │     model = "claude-sonnet-4-20250514",                        │
    │     system = "surgical video analysis assistant",              │
    │     messages = [{ image + prompt }],                           │
    │     max_tokens = 80                                            │
    │   )                                                            │
    │           │                                                    │
    │           ▼                                                    │
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
          ║        (Claude Messages API)             ║
          ════════════════════════════════════════════
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Claude Messages API                         │
    │                    (Anthropic SDK)                              │
    │                                                                │
    │   Per clip, three operations:                                  │
    │                                                                │
    │   ┌─ 1. Phase Resolution ────────────────────────────────────┐ │
    │   │  Majority vote across frame phase labels                 │ │
    │   │  → dominant phase + avg confidence                       │ │
    │   └──────────────────────────────────────────────────────────┘ │
    │                                                                │
    │   ┌─ 2. Caption Synthesis ───────────────────────────────────┐ │
    │   │                                                          │ │
    │   │  anthropic.Anthropic().messages.create(                   │ │
    │   │    model = "claude-sonnet-4-20250514",                    │ │
    │   │    system = "Summarize into one coherent surgical action",│ │
    │   │    messages = [{                                          │ │
    │   │      "Surgical phase: Tubularization                     │ │
    │   │       - [42.0s] Needle driver grasping suture...         │ │
    │   │       - [43.0s] Suture being pulled through...           │ │
    │   │       - [44.0s] ..."                                     │ │
    │   │    }],                                                   │ │
    │   │    max_tokens = 150                                      │ │
    │   │  )                                                       │ │
    │   │                                                          │ │
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
    │  score >= 0.6    │ │              │ │                      │
    │  total <= 120s   │ │              │ │                      │
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
║                         MODEL SUMMARY                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Phase 1   │ ViT (16×16 patches, 12-layer, 768-dim)  │ Object detection   ║
║            │ Weighted BCE loss (Eq 5: wᵢ = 1/fᵢ)     │ Multi-label        ║
║  ──────────┼──────────────────────────────────────────┼──────────────────  ║
║  Phase 1.5 │ ResNet-18 + Bidirectional GRU            │ Phase recognition  ║
║            │ Temporal smoothing + monotonic prior      │ 8 surgical phases  ║
║  ──────────┼──────────────────────────────────────────┼──────────────────  ║
║  Phase 2   │ Claude Vision API (Anthropic)            │ Frame captions     ║
║            │ Image + detected objects → description   │ Medical accuracy   ║
║  ──────────┼──────────────────────────────────────────┼──────────────────  ║
║  Phase 3   │ Claude Messages API (Anthropic)          │ Clip captions      ║
║            │ Frame captions + phase → coherent action │ + importance score ║
║  ──────────┼──────────────────────────────────────────┼──────────────────  ║
║  Phase 4   │ Scoring heuristic + video assembly       │ Highlight reel     ║
║            │ Phase-balanced / importance / uniform     │ + surgical report  ║
║                                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CLI:  python cli.py --input surgery.mp4 --config config/default.yaml      ║
║  Env:  ANTHROPIC_API_KEY required for Phases 2 & 3                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```
