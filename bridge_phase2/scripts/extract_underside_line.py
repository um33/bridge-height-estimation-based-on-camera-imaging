import json
from pathlib import Path

import cv2
import numpy as np


def bbox_from_pred(pred):
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
    x1 = int(max(0, min(w - 1, round(b["x1"]))))
    y1 = int(max(0, min(h - 1, round(b["y1"]))))
    x2 = int(max(0, min(w - 1, round(b["x2"]))))
    y2 = int(max(0, min(h - 1, round(b["y2"]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def make_inner_roi(
    x1, y1, x2, y2,
    x_margin_frac=0.04,
    y_top_frac=0.48,
    y_bottom_frac=0.98
):
    bw = x2 - x1
    bh = y2 - y1

    ix1 = x1 + int(round(x_margin_frac * bw))
    ix2 = x2 - int(round(x_margin_frac * bw))
    iy1 = y1 + int(round(y_top_frac * bh))
    iy2 = y1 + int(round(y_bottom_frac * bh))

    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def pick_underside_line(
    lines,
    roi_h,
    roi_w,
    max_slope=0.06,
    min_len_frac=0.40,
    min_y_frac=0.35,
    max_center_offset_frac=0.33,
    lowest_band_frac=0.12
):
    """
    Prefer the true bridge underside:
    - near-horizontal
    - long enough
    - not too high in ROI
    - not too far to one side
    - among valid candidates, choose from the lowest band first
    """

    if lines is None:
        return None

    min_len = min_len_frac * roi_w
    roi_center_x = 0.5 * roi_w

    candidates = []

    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, l)

        # normalize so x1 <= x2
        if x2 < x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 2:
            continue

        slope = abs(dy) / (abs(dx) + 1e-9)
        if slope > max_slope:
            continue

        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue

        y_mid = 0.5 * (y1 + y2)
        x_mid = 0.5 * (x1 + x2)

        # reject lines too high in ROI
        if y_mid < min_y_frac * roi_h:
            continue

        # reject lines whose midpoint is too far from ROI center
        center_offset = abs(x_mid - roi_center_x)
        if center_offset > max_center_offset_frac * roi_w:
            continue

        candidates.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "length": length,
            "slope": float(slope),
            "y_mid": float(y_mid),
            "x_mid": float(x_mid),
            "center_offset": float(center_offset),
        })

    if not candidates:
        return None

    # Step 1: keep only candidates close to the lowest valid line
    max_y_mid = max(c["y_mid"] for c in candidates)
    band_thresh = max_y_mid - lowest_band_frac * roi_h
    lowest_band = [c for c in candidates if c["y_mid"] >= band_thresh]

    # Step 2: among the lowest-band lines, prefer long + centered + low + flat
    best = None
    best_score = -1e18

    for c in lowest_band:
        score = (
            2.2 * c["length"]
            + 1.5 * c["y_mid"]
            - 2.0 * c["center_offset"]
            - 500.0 * c["slope"]
        )

        if score > best_score:
            best_score = score
            best = dict(c)
            best["score"] = float(score)
            best["candidate_count"] = len(candidates)
            best["lowest_band_count"] = len(lowest_band)

    return best


def draw_overlay(frame, bbox_used, inner_roi, picked_line, out_path):
    vis = frame.copy()

    if bbox_used is not None:
        cv2.rectangle(
            vis,
            (bbox_used["x1"], bbox_used["y1"]),
            (bbox_used["x2"], bbox_used["y2"]),
            (0, 255, 255),
            2
        )

    if inner_roi is not None:
        cv2.rectangle(
            vis,
            (inner_roi["x1"], inner_roi["y1"]),
            (inner_roi["x2"], inner_roi["y2"]),
            (255, 255, 0),
            2
        )

    if picked_line is not None:
        cv2.line(
            vis,
            (picked_line["x1"], picked_line["y1"]),
            (picked_line["x2"], picked_line["y2"]),
            (0, 255, 0),
            3
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


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

    ap.add_argument("--max_slope", type=float, default=0.06)
    ap.add_argument("--min_len_frac", type=float, default=0.40)

    ap.add_argument("--x_margin_frac", type=float, default=0.03)
    ap.add_argument("--y_top_frac", type=float, default=0.58)
    ap.add_argument("--y_bottom_frac", type=float, default=0.92)

    ap.add_argument("--save_overlays", action="store_true")
    ap.add_argument("--overlay_dir", default="outputs/underside_overlays_v2")

    args = ap.parse_args()

    records = [
        json.loads(l)
        for l in Path(args.in_jsonl).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

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
            rec["bridge_bbox_used"] = None
            rec["underside_debug"] = {
                "reason": "no_pred_or_low_conf",
                "best_conf": float(best_conf) if best is not None else None
            }
            out_lines.append(json.dumps(rec))
            continue

        bbox = bbox_from_pred(best)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            rec["underside_line"] = None
            rec["underside_y"] = None
            rec["underside_status"] = "frame_read_failed"
            rec["bridge_bbox_used"] = None
            rec["underside_debug"] = {"reason": "frame_read_failed"}
            out_lines.append(json.dumps(rec))
            continue

        H, W = frame.shape[:2]
        clamped = clamp_bbox(bbox, W, H)
        if clamped is None:
            rec["underside_line"] = None
            rec["underside_y"] = None
            rec["underside_status"] = "invalid_bbox"
            rec["bridge_bbox_used"] = None
            rec["underside_debug"] = {"reason": "invalid_bbox"}
            out_lines.append(json.dumps(rec))
            continue

        x1, y1, x2, y2 = clamped
        bbox_used = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "conf": float(best_conf)
        }

        inner = make_inner_roi(
            x1, y1, x2, y2,
            x_margin_frac=args.x_margin_frac,
            y_top_frac=args.y_top_frac,
            y_bottom_frac=args.y_bottom_frac
        )

        if inner is None:
            rec["underside_line"] = None
            rec["underside_y"] = None
            rec["underside_status"] = "invalid_inner_roi"
            rec["bridge_bbox_used"] = bbox_used
            rec["underside_debug"] = {"reason": "invalid_inner_roi"}
            out_lines.append(json.dumps(rec))
            continue

        ix1, iy1, ix2, iy2 = inner
        roi = frame[iy1:iy2, ix1:ix2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, args.canny1, args.canny2)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=50,
            minLineLength=max(20, int((ix2 - ix1) * args.min_len_frac)),
            maxLineGap=20
        )

        pick = pick_underside_line(
            lines,
            roi_h=(iy2 - iy1),
            roi_w=(ix2 - ix1),
            max_slope=args.max_slope,
            min_len_frac=args.min_len_frac
        )

        if pick is None:
            rec["underside_line"] = None
            rec["underside_y"] = None
            rec["underside_status"] = "no_good_line"
            rec["bridge_bbox_used"] = bbox_used
            rec["underside_debug"] = {
                "reason": "no_good_line",
                "bbox_used": bbox_used,
                "inner_roi": {"x1": ix1, "y1": iy1, "x2": ix2, "y2": iy2}
            }
            out_lines.append(json.dumps(rec))
            continue

        gx1 = pick["x1"] + ix1
        gy1 = pick["y1"] + iy1
        gx2 = pick["x2"] + ix1
        gy2 = pick["y2"] + iy1

        line_full = {
            "x1": int(gx1),
            "y1": int(gy1),
            "x2": int(gx2),
            "y2": int(gy2),
            "score": float(pick["score"]),
            "length": float(pick["length"]),
            "slope": float(pick["slope"]),
        }

        rec["underside_line"] = line_full
        rec["underside_y"] = float((gy1 + gy2) / 2.0)
        rec["underside_status"] = "ok"
        rec["bridge_bbox_used"] = bbox_used
        rec["underside_debug"] = {
            "bbox_used": bbox_used,
            "inner_roi": {"x1": ix1, "y1": iy1, "x2": ix2, "y2": iy2},
            "picked_line_roi": pick
        }

        if args.save_overlays:
            overlay_path = Path(args.overlay_dir) / f"frame_{frame_idx:06d}.jpg"
            draw_overlay(
                frame=frame,
                bbox_used=bbox_used,
                inner_roi={"x1": ix1, "y1": iy1, "x2": ix2, "y2": iy2},
                picked_line=line_full,
                out_path=overlay_path
            )

        out_lines.append(json.dumps(rec))

    cap.release()
    Path(args.out_jsonl).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"✅ Wrote: {args.out_jsonl}")


if __name__ == "__main__":
    main()


# python scripts/extract_underside_line.py \
#   --video video.MOV \
#   --in_jsonl outputs/video_detections_every_1s.jsonl \
#   --out_jsonl outputs/extract_underside_lines.jsonl \
#   --y_top_frac 0.48 \
#   --y_bottom_frac 0.98 \
#   --max_slope 0.10 \
#   --min_len_frac 0.30 \
#   --save_overlays
