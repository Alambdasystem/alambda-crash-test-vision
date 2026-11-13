# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alambda Systems
"""Detect and associate one vehicle in a controlled crash-test video.

Only actual YOLOX detections are exported. Missing detections remain missing;
no optical-flow continuation, interpolation, speed, distance, or fault is inferred.
"""
from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


class Detector:
    """YOLOX-S ONNX decoding adapted from the Apache-2.0 OpenCV Zoo example."""

    def __init__(self, weights, confidence=0.20, nms=0.45):
        self.net = cv2.dnn.readNet(str(weights))
        self.confidence, self.nms = confidence, nms
        grids, strides = [], []
        for stride in (8, 16, 32):
            yy, xx = np.mgrid[:640 // stride, :640 // stride]
            grids.append(np.stack([xx, yy], axis=-1).reshape(-1, 2))
            strides.append(np.full((len(grids[-1]), 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def run(self, frame):
        height, width = frame.shape[:2]
        scale = min(640 / height, 640 / width)
        nw, nh = round(width * scale), round(height * scale)
        padded = np.full((640, 640, 3), 114, np.uint8)
        padded[:nh, :nw] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (nw, nh))
        self.net.setInput(padded.transpose(2, 0, 1).astype(np.float32)[None])
        raw = self.net.forward()[0]
        class_ids = raw[:, 5:].argmax(axis=1)
        scores = raw[:, 4] * raw[np.arange(len(raw)), class_ids + 5]
        take = (scores >= self.confidence) & np.isin(class_ids, list(VEHICLE_CLASSES))
        if not np.any(take):
            return []
        rows, scores, class_ids = raw[take], scores[take], class_ids[take]
        centers = (rows[:, :2] + self.grid[take]) * self.stride[take]
        sizes = np.exp(np.clip(rows[:, 2:4], -20, 20)) * self.stride[take]
        xywh = np.concatenate([(centers - sizes / 2) / scale, sizes / scale], axis=1)
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.confidence, self.nms)
        detections = []
        for i in np.asarray(keep, dtype=int).ravel():
            x, y, w, h = xywh[i]
            box = np.array([max(0., x), max(0., y), min(float(width), x + w), min(float(height), y + h)])
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            cid = int(class_ids[i])
            detections.append({"bbox": box, "confidence": float(scores[i]),
                               "class_id": cid, "class_name": VEHICLE_CLASSES[cid]})
        return detections


def iou(a, b):
    left, top = np.maximum(a[:2], b[:2])
    right, bottom = np.minimum(a[2:], b[2:])
    overlap = max(0., right - left) * max(0., bottom - top)
    area_a, area_b = np.prod(a[2:] - a[:2]), np.prod(b[2:] - b[:2])
    return float(overlap / (area_a + area_b - overlap + 1e-9))


def select_vehicle(detections, width, height, roi, min_area, previous, gap):
    """Start with the largest eligible vehicle; retain nearby overlapping detections."""
    eligible = []
    for d in detections:
        box = d["bbox"]
        center = (box[:2] + box[2:]) / 2
        area = float(np.prod(box[2:] - box[:2])) / (width * height)
        if (area >= min_area and roi[0] <= center[0] / width <= roi[2]
                and roi[1] <= center[1] / height <= roi[3]):
            eligible.append(d)
    if not eligible:
        return None
    if previous is None:
        return max(eligible, key=lambda d: (float(np.prod(d["bbox"][2:] - d["bbox"][:2])), d["confidence"]))
    diagonal = np.hypot(width, height)
    old_center = (previous[:2] + previous[2:]) / 2
    ranked = []
    for d in eligible:
        center = (d["bbox"][:2] + d["bbox"][2:]) / 2
        distance = float(np.linalg.norm(center - old_center) / diagonal)
        if distance <= min(.65, .20 + .02 * gap):
            score = 1.5 * iou(previous, d["bbox"]) - 2.0 * distance + .15 * d["confidence"]
            ranked.append((score, d))
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "assets/source.mp4")
    parser.add_argument("--weights", type=Path, default=ROOT / "assets/yolox_s.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "output/tracks.json")
    parser.add_argument("--confidence", type=float, default=.20)
    parser.add_argument("--min-area-ratio", type=float, default=.05)
    parser.add_argument("--roi", type=float, nargs=4, default=[0., 0., 1., 1.], metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or not 0 <= args.min_area_ratio < 1:
        parser.error("Confidence must be in (0,1), and minimum area ratio in [0,1).")
    if not (0 <= args.roi[0] < args.roi[2] <= 1 and 0 <= args.roi[1] < args.roi[3] <= 1):
        parser.error("ROI must be normalized X1 Y1 X2 Y2 within [0,1].")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Model not found: {args.weights}")
    cv2.setNumThreads(4)
    detector = Detector(args.weights, args.confidence)
    cap = cv2.VideoCapture(str(args.source))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Cannot read valid video metadata: {args.source}")
    frames, previous, last_seen = [], None, -1
    candidate_count, detected_count = 0, 0
    class_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    started = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n = len(frames)
        candidates = detector.run(frame)
        candidate_count += len(candidates)
        chosen = select_vehicle(candidates, width, height, args.roi, args.min_area_ratio, previous, n - last_seen)
        objects = []
        if chosen is not None:
            previous, last_seen = chosen["bbox"].copy(), n
            box = [round(float(x), 2) for x in chosen["bbox"]]
            point = [round((box[0] + box[2]) / 2, 2), round((box[1] + box[3]) / 2, 2)]
            objects.append({"id": "V1", "label": "vehicle", "class_id": chosen["class_id"],
                            "class_name": chosen["class_name"], "confidence": round(chosen["confidence"], 4),
                            "bbox": box, "point": point, "observation": "detected"})
            detected_count += 1
            class_counts[chosen["class_name"]] += 1
        frames.append({"frame": n, "source_time_s": round(n / fps, 6), "objects": objects})
        if n % 30 == 0:
            print(json.dumps({"frame": n, "expected_frames": expected, "detected_frames": detected_count,
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    cap.release()
    if not frames:
        raise ValueError("Video contained no readable frames")
    limitations = [
        "Experimental vehicle detection on a controlled crash-test clip; no independent accuracy benchmark.",
        "V1 is a temporary single-video association, not a persistent or verified vehicle identity.",
        "All boxes are actual detector outputs; missing detections are omitted and no positions are interpolated.",
        "Bounding-box center is an image-space visualization point, not a physical landmark or center of mass.",
        "Crushing, occlusion, debris, and motion blur can cause missed, fragmented, or inaccurate detections.",
        "No calibrated speed, distance, acceleration, impact force, causation, or fault is inferred.",
        "Timestamps describe this source file's playback time; capture frame rate may differ.",
    ]
    data = {
        "video": {"width": width, "height": height, "fps": fps, "frames": len(frames), "duration_s": len(frames) / fps},
        "method": {
            "detector": "OpenCV Zoo YOLOX-S ONNX (COCO pretrained)",
            "detector_source": "https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox",
            "classes": VEHICLE_CLASSES, "confidence_threshold": args.confidence, "nms_iou_threshold": .45,
            "class_selection": "Highest-scoring COCO class per proposal; retain car (2), bus (5), or truck (7).",
            "selection": "Largest eligible vehicle initially, then overlap and center proximity to the last actual detection.",
            "roi_normalized": args.roi, "min_bbox_area_ratio": args.min_area_ratio,
            "continuations": False, "point": "Bounding-box center in source-image pixels.",
            "limitations": limitations,
        },
        "frames": frames,
    }
    summary = {
        "frames": len(frames), "reported_source_frames": expected, "fps": fps,
        "duration_s": len(frames) / fps, "detected_frames": detected_count,
        "missing_frames": len(frames) - detected_count, "coverage_fraction": detected_count / len(frames),
        "vehicle_candidate_detections": candidate_count, "accepted_class_counts": class_counts,
        "interpolated_or_continued_observations": 0, "elapsed_s": round(time.monotonic() - started, 2),
        "selection_roi_normalized": args.roi, "limitations": limitations,
        "note": "Coverage describes whether a detection was emitted, not accuracy or identity correctness.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    (args.output.parent / "processing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
