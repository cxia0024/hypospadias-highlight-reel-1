# Methodology

## Overview

We propose a four-phase computational framework for the automated generation of surgical highlight reels from full-length intraoperative hypospadias repair videos. The pipeline processes raw video through sequential stages of (i) multi-label object detection, (ii) surgical phase recognition, (iii) hierarchical captioning at both frame and clip levels, and (iv) importance-driven highlight selection with structured report synthesis. The complete framework is illustrated in Figure 1.

## 3.1 Video Preprocessing and Frame Sampling

Given an input surgical video $V$ of duration $T$ seconds recorded at a native frame rate $r_{\text{native}}$ (typically 30–60 fps), we first perform temporal downsampling to reduce computational burden while preserving clinically relevant information. We extract a set of frames $\mathcal{F} = \{F_1, F_2, \ldots, F_M\}$ at a configurable sampling rate $r_s$ (default: 1 fps), where:

$$M = \left\lfloor \frac{T \cdot r_{\text{native}}}{r_{\text{native}} / r_s} \right\rfloor = \lfloor T \cdot r_s \rfloor$$

Each frame $F_i$ is represented as a tuple $(I_i, t_i, idx_i)$, where $I_i \in \mathbb{R}^{H \times W \times 3}$ is the RGB image, $t_i = idx_i / r_{\text{native}}$ is the timestamp in seconds, and $idx_i$ is the original frame index. For a typical 2-hour hypospadias repair, this reduces the processing load from approximately 432,000 frames (at 60 fps) to approximately 7,200 frames.

## 3.2 Phase I: Multi-Label Object Detection via Vision Transformer

### 3.2.1 Problem Formulation

The object detection phase is formulated as a multi-label classification problem rather than a spatial localization task. Given a frame image $I_i$, we seek to identify the set of surgical instruments and anatomical structures present in the scene. We define the label space $\mathcal{C} = \{c_1, c_2, \ldots, c_K\}$ comprising $K = 9$ classes across two categories:

- **Instruments**: scissors, forceps, needle driver, suture, catheter
- **Anatomy**: urethral plate, glans, skin flap, dartos flap

This formulation is motivated by two observations: (1) downstream pipeline stages require knowledge of *which* objects are present rather than *where* they are located, and (2) multi-label image-level annotation is substantially less expensive than bounding box annotation, requiring only binary presence/absence labels per class per frame.

### 3.2.2 Architecture

We employ a Vision Transformer (ViT) architecture adapted for multi-label surgical instrument detection. The model processes each frame through the following stages.

**Patch Extraction.** Given an image matrix $A \in \mathbb{R}^{n \times m}$ (resized to $224 \times 224 \times 3$), we extract $N$ non-overlapping patches. The $k$-th image patch $P_k \in \mathbb{R}^{a \times b}$, assuming $a$ and $b$ evenly divide $n$ and $m$ respectively, is defined as:

$$P_k = A_{[i:i+a-1,\; j:j+b-1]}, \quad k = i \times \frac{m}{b} + j \tag{1}$$

With patch size $a = b = 16$, we obtain $N = (224/16)^2 = 196$ patches, each of dimension $3 \times 16 \times 16 = 768$ when flattened.

**Embedding and Positional Encoding.** Each flattened patch $\tilde{P}_k \in \mathbb{R}^{768}$ is projected to the embedding dimension $d = 768$ via a linear transformation and augmented with a learnable positional embedding $E_{\text{pos}} \in \mathbb{R}^{N \times d}$:

$$I = \left[[1, \tilde{P}_1],\; [2, \tilde{P}_2],\; \ldots,\; [N, \tilde{P}_N]\right] \in \mathbb{R}^{1 \times (16 \times 16 + 1) \times 196} \tag{2}$$

More precisely, the input sequence to the transformer encoder is:

$$z_k = W_{\text{embed}} \cdot \tilde{P}_k + E_{\text{pos}}[k], \quad k = 1, \ldots, N$$

where $W_{\text{embed}} \in \mathbb{R}^{d \times d}$ is the learned projection matrix.

**Transformer Encoder.** The embedded sequence passes through an $L = 12$ layer transformer encoder. Each layer comprises pre-normalized multi-head self-attention (with $h = 12$ heads and head dimension $d_k = d/h = 64$) followed by a feed-forward network with GELU activation and hidden dimension 3072:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

$$\text{MultiHead}(Z) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W_O$$

The self-attention mechanism enables each patch to attend to all other patches, allowing the model to capture global spatial relationships between instruments and anatomy—for instance, the co-occurrence of a needle driver with suture material near the urethral plate.

**Classification.** Following the transformer encoder, we apply global average pooling over all $N$ patch tokens to obtain a single feature vector $\bar{z} \in \mathbb{R}^d$. This is passed through a two-layer classification head:

$$P = \sigma(\text{MLP}(\bar{z})) \tag{3}$$

where $\sigma$ denotes the element-wise sigmoid function and $P \in [0, 1]^K$ represents the independent probability of each class being present. The final detected object set is:

$$O = \{\text{object}_i \mid P_i > \tau\} \tag{4}$$

where $\tau = 0.5$ is the detection threshold, which may be calibrated based on empirical performance analysis.

### 3.2.3 Training with Weighted Binary Cross-Entropy

Given the inherent class imbalance within surgical datasets—where certain objects such as forceps appear in the majority of frames while others such as the dartos flap appear infrequently—we employ a weighted Binary Cross-Entropy (BCE) loss:

$$\mathcal{L} = -\sum_{i=1}^{n} w_i \left[ y_i \log \sigma(\hat{y}_i) + (1 - y_i) \log(1 - \sigma(\hat{y}_i)) \right] \tag{5}$$

where $y_i$ and $\hat{y}_i$ denote the ground truth and predicted logits respectively. Class-specific weights $w_i$ are computed as the inverse of class frequency $f_i$:

$$w_i = \frac{1}{f_i}$$

This ensures that misclassifications of rare but clinically significant structures (e.g., dartos flap during the waterproofing layer) contribute proportionally more to the training loss than misclassifications of ubiquitous instruments.

## 3.3 Phase I.5: Surgical Phase Recognition

### 3.3.1 Motivation

Hypospadias repair follows a well-defined sequence of operative steps. We define eight canonical surgical phases $\mathcal{S} = \{s_1, \ldots, s_8\}$, ordered by their expected temporal occurrence:

| Phase | Description | Highlight Weight $\omega_s$ |
|-------|-------------|:---------------------------:|
| $s_1$: Preparation | Patient positioning, catheterization, marking | 0.2 |
| $s_2$: Degloving | Circumferential subcoronal incision, penile degloving | 0.5 |
| $s_3$: Urethral Plate Incision | Midline relaxing incision of the urethral plate | 0.8 |
| $s_4$: Tubularization | Tubularized incised plate (TIP/Snodgrass) urethroplasty | 1.0 |
| $s_5$: Waterproofing | Dartos flap or tunica vaginalis coverage | 0.7 |
| $s_6$: Glansplasty | Glans wing approximation, meatoplasty | 0.9 |
| $s_7$: Skin Closure | Ventral skin coverage, byars flaps | 0.6 |
| $s_8$: Dressing | Catheter fixation, compressive dressing application | 0.1 |

Each phase is assigned a highlight weight $\omega_s \in [0, 1]$ reflecting its relative surgical significance for educational and documentation purposes. These weights were determined through consultation with pediatric urologists, with the core reconstructive steps (tubularization, glansplasty, urethral plate incision) receiving the highest weights.

### 3.3.2 Temporal Phase Classification Model

We employ a two-stage architecture for temporal phase recognition.

**Spatial Feature Extraction.** A ResNet-18 backbone extracts a spatial feature vector $\phi_i \in \mathbb{R}^{512}$ from each frame $I_i$:

$$\phi_i = \text{ResNet-18}(I_i), \quad \phi_i \in \mathbb{R}^{512}$$

**Temporal Modeling.** The sequence of spatial features $\Phi = [\phi_1, \phi_2, \ldots, \phi_M]$ is processed by a 2-layer bidirectional Gated Recurrent Unit (GRU):

$$\overrightarrow{h_i}, \overleftarrow{h_i} = \text{BiGRU}(\Phi), \quad h_i = [\overrightarrow{h_i} \| \overleftarrow{h_i}] \in \mathbb{R}^{512}$$

The bidirectional architecture allows each frame's phase prediction to be informed by both preceding and subsequent frames, capturing the temporal context of surgical actions. The hidden states are mapped to phase logits via a classification head:

$$\hat{s}_i = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot h_i + b_1) + b_2)$$

### 3.3.3 Temporal Post-Processing

Raw per-frame phase predictions are refined through two post-processing steps to enforce temporal coherence.

**Temporal Smoothing.** We apply a sliding window average of width $W = 15$ frames to suppress frame-to-frame prediction jitter:

$$\hat{s}_i^{\text{smooth}} = \frac{1}{|w_i|} \sum_{j \in w_i} \hat{s}_j, \quad w_i = \{j : |j - i| \leq \lfloor W/2 \rfloor\}$$

The smoothed probabilities are re-normalized to form a valid distribution.

**Monotonic Phase Prior.** Surgical procedures progress forward through defined stages and do not revert to earlier phases under normal circumstances. We enforce this constraint through a forward-biased transition penalty. Let $s^*_{i-1} = \arg\max_j \hat{s}_{i-1,j}^{\text{smooth}}$ denote the predicted phase at the previous frame. For frame $i$, we penalize all phases preceding $s^*_{i-1}$:

$$\hat{s}_{i,j}^{\text{adjusted}} = \begin{cases} \hat{s}_{i,j}^{\text{smooth}} \cdot (1 - \lambda) & \text{if } j < s^*_{i-1} \\ \hat{s}_{i,j}^{\text{smooth}} & \text{otherwise} \end{cases}$$

where $\lambda = 0.3$ is the transition penalty parameter. This soft constraint discourages but does not prohibit backward transitions, accommodating occasional revisits to earlier phases that may occur in complex cases.

### 3.3.4 Heuristic Fallback

When trained model weights are unavailable, we employ a heuristic phase estimator that combines instrument-phase co-occurrence priors with temporal position. For each frame $F_i$ with detected object set $O_i$ and normalized temporal position $\rho_i = i / (M - 1) \in [0, 1]$, the probability of phase $s_j$ is estimated as:

$$P(s_j \mid F_i) \propto 0.4 \cdot \underbrace{\max\!\left(0,\; 1 - 2\left|\rho_i - \frac{j}{|\mathcal{S}| - 1}\right|\right)}_{\text{temporal position score}} + 0.6 \cdot \underbrace{\frac{\sum_{c \in O_i} \pi_{j,c}}{|\Pi_j|}}_{\text{instrument co-occurrence score}}$$

where $\pi_{j,c}$ denotes the prior probability of observing instrument class $c$ during phase $s_j$, and $\Pi_j$ is the set of expected instruments for phase $j$. These priors encode domain knowledge—for instance, the co-occurrence of a needle driver and suture with the urethral plate strongly indicates the tubularization phase.

## 3.4 Phase II: Frame-Level Captioning via Vision-Language Model

For each frame $F_i$ with detected object set $O_i$, we generate a descriptive caption $d_i$ using a multimodal large language model (Claude, Anthropic). The frame image $I_i$ is encoded as a base64 JPEG and submitted alongside a structured prompt conditioned on the detection results:

$$d_i = \text{VLM}\!\left(I_i,\; \text{``Describe the surgical scene showing } O_i\text{''}\right)$$

The VLM receives both the raw visual information and the symbolic detection output, enabling it to generate medically specific descriptions grounded in the detected objects. For example, given $O_i = \{\text{needle\_driver}, \text{suture}, \text{urethral\_plate}\}$, the model might produce: *"Needle driver placing interrupted subepithelial suture through the lateral edges of the incised urethral plate over an 8Fr catheter stent."*

A system prompt constrains the model to produce concise, medically accurate descriptions limited to 80 tokens, prioritizing surgical actions and instrument-tissue interactions over generic scene description.

## 3.5 Temporal Aggregation: Clip Formation

Frame-level annotations are aggregated into temporally coherent clips using a sliding window approach. Given parameters for window duration $\Delta t$ (default: 10 seconds) and stride $\delta$ (default: 5 seconds), we partition the frame sequence into overlapping clips:

$$\mathcal{C}_k = \{F_i \mid t_{\text{start}}^{(k)} \leq t_i < t_{\text{start}}^{(k)} + \Delta t\}, \quad t_{\text{start}}^{(k)} = k \cdot \delta$$

The 50% overlap between consecutive windows ensures that surgical actions spanning a window boundary are captured in at least one complete clip, preventing loss of clinically relevant transitions.

## 3.6 Phase III: Clip-Level Captioning and Importance Scoring

### 3.6.1 Phase Resolution

Each clip $\mathcal{C}_k$ is assigned a dominant surgical phase via majority vote over its constituent frame-level phase labels:

$$s_k^* = \arg\max_{s \in \mathcal{S}} \left| \{F_i \in \mathcal{C}_k \mid \hat{s}_i = s\} \right|$$

The phase confidence for the clip is computed as the mean confidence among frames assigned to the dominant phase.

### 3.6.2 Caption Synthesis

The set of frame-level captions $\{d_i : F_i \in \mathcal{C}_k\}$ within each clip is synthesized into a coherent clip-level narrative $D_k$ using a large language model. The model receives the surgical phase context alongside the temporally ordered frame descriptions:

$$D_k = \text{LLM}\!\left(\text{``Phase: } s_k^*\text{. Summarize: } \{d_i\}_{i \in \mathcal{C}_k}\text{''}\right)$$

This produces a single coherent sentence describing the surgical action performed during the clip, for example: *"Surgeon tubularizes the urethral plate with a running subepithelial 6-0 polyglycolic acid suture over an 8Fr catheter stent."*

### 3.6.3 Importance Scoring

Each clip is assigned a composite importance score $\mathcal{I}_k \in [0, 1]$ that quantifies its relevance for inclusion in the highlight reel. The score integrates three components:

$$\mathcal{I}_k = \alpha \cdot \text{div}(\mathcal{C}_k) + \beta \cdot \text{dens}(\mathcal{C}_k) + \gamma \cdot \omega_{s_k^*} \tag{6}$$

where:

**Object diversity** measures the variety of instruments and anatomical structures detected:
$$\text{div}(\mathcal{C}_k) = \min\!\left(\frac{|\bigcup_{F_i \in \mathcal{C}_k} O_i|}{5},\; 1.0\right)$$

**Detection density** captures the level of active instrumentation:
$$\text{dens}(\mathcal{C}_k) = \min\!\left(\frac{\sum_{F_i \in \mathcal{C}_k} |O_i|}{|\mathcal{C}_k| \cdot 3},\; 1.0\right)$$

**Phase weight** $\omega_{s_k^*}$ is the highlight weight of the clip's dominant surgical phase (Table 1).

The weighting coefficients $\alpha = \beta = 0.4$ and $\gamma = 0.2$ were chosen to emphasize scene content while incorporating phase-level surgical significance. Clips with $\mathcal{I}_k < \tau_{\min} = 0.6$ are excluded from highlight consideration.

## 3.7 Phase IV: Highlight Reel Synthesis and Report Generation

### 3.7.1 Clip Selection Strategies

Three selection strategies are provided for constructing the highlight reel within a maximum duration budget $T_{\max}$ (default: 120 seconds):

**Importance-ranked selection** greedily selects clips in descending order of importance score until the duration budget is exhausted:

$$\mathcal{H} = \text{GreedySelect}\!\left(\text{sort}_{\mathcal{I}}(\mathcal{C}),\; T_{\max}\right)$$

**Phase-balanced selection** allocates an equal time budget $T_{\max} / |\mathcal{S}'|$ to each observed surgical phase $\mathcal{S}' \subseteq \mathcal{S}$, then selects the highest-scoring clips within each phase's allocation. This ensures representation of all operative stages:

$$\mathcal{H}_s = \text{GreedySelect}\!\left(\text{sort}_{\mathcal{I}}(\mathcal{C}_s),\; \frac{T_{\max}}{|\mathcal{S}'|}\right), \quad \mathcal{H} = \bigcup_{s \in \mathcal{S}'} \mathcal{H}_s$$

**Uniform selection** samples clips chronologically to produce a temporally representative summary.

In all strategies, selected clips are re-ordered chronologically in the final highlight reel to preserve the natural operative narrative.

### 3.7.2 Video Assembly

The highlight reel is assembled by seeking to each selected clip's temporal boundaries in the original video and extracting the corresponding frame sequence at native resolution and frame rate. Clips are concatenated in chronological order to produce the output video.

### 3.7.3 Structured Surgical Report

A comprehensive markdown report is generated containing:

1. **Summary**: overall procedure duration, number of clips analyzed, and highlight reel statistics
2. **Surgical phase summary**: tabulated time ranges, durations, clip counts, and average importance scores per phase
3. **Key moments**: the selected highlight clips with timestamps, phase labels, importance scores, and clip-level captions
4. **Instrument usage**: relative frequency of each detected instrument class across the entire procedure, computed as $u_c = n_c / \sum_{c'} n_{c'}$ where $n_c$ is the total number of frame-level detections of class $c$
5. **Full timeline**: every clip with its phase label, caption, and importance score

## 3.8 Implementation Details

The framework is implemented in Python with the following dependencies:

| Component | Implementation |
|-----------|---------------|
| Object detection (ViT) | PyTorch, 12-layer ViT with 768-dim embeddings, 12 attention heads |
| Phase recognition | ResNet-18 backbone + 2-layer bidirectional GRU (hidden dim 256) |
| Frame captioning | Anthropic Claude API (vision), base64-encoded JPEG input |
| Clip captioning | Anthropic Claude API (text), phase-conditioned summarization |
| Video I/O | OpenCV |

All neural network models use ImageNet-normalized preprocessing ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$). The ViT and phase recognition models are trained offline on annotated surgical video datasets; at inference time, the complete pipeline processes a 2-hour video in approximately $M$ forward passes through each model plus $M + |\mathcal{C}|$ API calls to the vision-language model.

The pipeline is configured via a YAML specification file that allows adjustment of all hyperparameters including sampling rate $r_s$, patch size, detection threshold $\tau$, temporal smoothing window $W$, transition penalty $\lambda$, clip duration $\Delta t$, importance weights $(\alpha, \beta, \gamma)$, and selection strategy.
