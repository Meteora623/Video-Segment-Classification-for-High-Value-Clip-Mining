# Phase 0 — Problem Definition (Video Segment Classification)

## 1. Background
We need to mine high-value segments from long videos. Only a small portion of a long video contains useful events (e.g., continuous motion / interaction), while most segments are uninformative. Sending all frames to expensive downstream pipelines is costly and slows iteration.

## 2. Problem Statement
Given an input video, identify time segments that likely contain "high-value events" and output a list of (start_time, end_time, label, confidence).

This is a **segment-level classification** problem (not single-label video classification and not per-frame detection).

## 3. Users & Use Cases
Primary users:
- Downstream ML pipelines (expensive models / 3D reconstruction / manual review)
- Data curation workflows (building cleaner training data)

Example use cases:
- Trigger downstream processing only for predicted positive segments
- Build a continuously improving dataset via an ML flywheel

## 4. Input / Output Contract
Input:
- video file (mp4) OR extracted frames at fixed fps
- optional metadata: fps, resolution, timestamp

Output (JSON):
{
  "segments": [
    { "start": <float>, "end": <float>, "label": "<string>", "confidence": <float> }
  ],
  "model_version": "<string>"
}

Constraints:
- segments must satisfy: 0 <= start < end <= video_duration
- confidence in [0, 1]
- overlapping segments allowed / not allowed: <choose one>

## 5. Label Taxonomy (v1)
We start with a simple taxonomy for v1:

Binary:
- POSITIVE (1): continuous high-value event (e.g., continuous motion / interaction)
- NEGATIVE (0): uninformative / background

Notes:
- v1 focuses on being a cheap gating model rather than perfect fine-grained understanding.
- taxonomy may evolve to multi-class later.

## 6. Success Metrics
Offline (proxy) metrics:
- Segment-level Precision / Recall (at IoU threshold τ = <e.g., 0.5>)
- F1 score
- False Negative Rate (FNR) on positives (important if missing events is costly)

Online (system) metrics:
- Downstream cost reduction: % decrease in segments sent to expensive pipeline
- Downstream hit rate: % of selected segments accepted by downstream pipeline
- Serving latency: P95 < <e.g., 50ms> per clip request
- Reliability: error rate < <e.g., 0.1%>

## 7. Data Assumptions & Availability
Initial data source:
- <where do videos come from?>

Labeling approach (v1):
- weak labels from heuristics OR small manually labeled set
- expected label noise: <low/medium/high>

Dataset split:
- time-based split to avoid leakage (train on earlier videos, test on later)
- optional source-based split to test generalization

## 8. System Constraints & Trade-offs
Compute constraints:
- prefer CPU-friendly inference; GPU optional for offline training
- model should be lightweight (cheap gating)

Trade-offs:
- We prefer higher recall over precision / or prefer precision over recall: <choose and justify>
- We accept some false positives to avoid missing true events (or vice versa)

Failure modes:
- Distribution shift (different video sources, fps, compression)
- Data quality issues (blur, low light, camera shake)
- Concept drift (new event patterns)

## 9. Non-goals (v1)
- Fine-grained action recognition across dozens of classes
- Frame-level object detection / tracking as the primary output
- Perfect segmentation boundaries (v1 allows coarse boundaries)

## 10. Milestones (v0 -> v1)
v0 (baseline):
- simple clip extraction + cheap features + baseline classifier + offline eval

v1 (system):
- FastAPI serving + logging + basic monitoring + retraining loop
