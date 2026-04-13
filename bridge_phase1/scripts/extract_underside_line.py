import json
from pathlib import Path

import cv2
import numpy as np


def bbox_from_pred(pred):
    """Convert Roboflow center bbox (x,y,w,h) to corner bbox dict {x1,y1,x2,y2,conf}."""
    x = float(pred["x"])
    y = float(pred["y"])
    w = float(pred["width"])
    h = float(pred["height"])
    return {
        "x1": x - w / 2.0,
        "y1": y - h / 2.0,
        "x2": x + w / 2.0,
        "y2": y + h / 2.0,
        "conf": float(pred.get("confidence", 0.0)),
    }


def best_prediction(preds, class_name="Bridge"):
    best = None
    best_conf = -1.0
    for p in preds:
        if p.get("class") != class_name:
            continue
        c = float(p.get("confidence", 0.0))
        if c > best_conf:
            best_conf = c
            best = p
    return best, best_conf


def clamp_bbox(b, w, h):
    """Clamp bbox dict {x1,y1,x2,y2} to image bounds and return ints."""
    x1 = int(max(0, min(w - 1, round(b["x1"]))))
    y1 = int(max(0, min(h - 1, round(b["y1"]))))
    x2 = int(max(0, min(w - 1, round(b["x2"]))))
    y2 = int(max(0, min(h - 1, round(b["y2"]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def pick_underside_line(lines, roi_h, roi_w,
                        max_slope=0.15,
                        min_len_frac=0.35):
    """
    Choose best near-horizontal line, biased toward lower part of ROI.
    Returns (x1,y1,x2,y2,score) in ROI coords or None.
    """
    if lines is None:
        return None

    best = None
    best_score = -1e18
    min_len = min_len_frac * roi_w

    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, l)
        dx = x2 - x1
        dy = y2 - y1
        adx = abs(dx)
        if adx < 2:
            continue

        slope = abs(dy) / (adx + 1e-9)
        if slope > max_slope:
            continue

        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue

        y_mid = (y1 + y2) / 2.0
        # Score: prefer longer and lower lines
        score = length + 0.6 * y_mid

        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2, best_score)

    return best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--class_name", default="Bridge")
    ap.add_argument("--conf_th", type=float, default=0.35)

    ap.add_argument("--canny1", type=int, default=50)
    ap.add_argument("--canny2", type=int, default=150)

    ap.add_argument("--max_slope", type=float, default=0.15)
    ap.add_argument("--min_len_frac", type=float, default=0.35)

    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.in_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    out_lines = []

    for rec in records:
        frame_idx = int(rec.get("frame_idx", 0))
        preds = rec.get("predictions", [])

        best, best_conf = best_prediction(preds, class_name=args.class_name)
        if best is None or best_conf < args.conf_th:
            rec["underside_line"] = None
            rec["underside_y"] = None
            rec["underside_status"] = "no_pred_or_low_conf"
            out_lines.append(json.dumps(rec))
            continue

        bbox = bbox_from_pred(best)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            rec["underside_line"] = None
            rec["underside_y"] = None
            rec["underside_status"] = "frame_read_failed"
            out_lines.append(json.dumps(rec))
            continue

        H, W = frame.shape[:2]
        clamped = clamp_bbox(bbox, W, H)
        if clamped is None:
            rec["underside_line"] = None
            rec["underside_y"] = None
            rec["underside_status"] = "invalid_bbox"
            out_lines.append(json.dumps(rec))
            continue

        x1, y1, x2, y2 = clamped
        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, args.canny1, args.canny2)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=max(20, int((x2 - x1) * args.min_len_frac)),
            maxLineGap=15
        )

        pick = pick_underside_line(
            lines,
            roi_h=(y2 - y1),
            roi_w=(x2 - x1),
            max_slope=args.max_slope,
            min_len_frac=args.min_len_frac
        )

        if pick is None:
            rec["underside_line"] = None
            rec["underside_y"] = None
            rec["underside_status"] = "no_good_line"
            out_lines.append(json.dumps(rec))
            continue

        lx1, ly1, lx2, ly2, score = pick

        # ROI -> full image coords
        gx1 = lx1 + x1
        gy1 = ly1 + y1
        gx2 = lx2 + x1
        gy2 = ly2 + y1

        rec["underside_line"] = {"x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2, "score": float(score)}
        rec["underside_y"] = float((gy1 + gy2) / 2.0)
        rec["underside_status"] = "ok"

        # Optional: store bbox we used (useful for debugging later)
        rec["bridge_bbox_used"] = bbox

        out_lines.append(json.dumps(rec))

    cap.release()
    Path(args.out_jsonl).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print("✅ Wrote:", args.out_jsonl)


if __name__ == "__main__":
    main()



# python scripts/extract_underside_line.py \
#   --video video.MOV \
#   --in_jsonl outputs/video_detections_every_1s.jsonl \
#   --out_jsonl outputs/video_detections_every_1s_with_underside.jsonl