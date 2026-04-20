import json
from pathlib import Path

import cv2
import numpy as np


def load_selected_frames(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Failed to read frame {idx}")
        frames.append((int(idx), frame.copy()))

    cap.release()
    return frames


def load_jsonl_by_frame(jsonl_path):
    records = {}
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        records[int(rec["frame_idx"])] = rec
    return records


def detect_features_in_mask(gray, mask, max_corners=250, quality_level=0.01, min_distance=8, block_size=7):
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask,
        blockSize=block_size,
    )
    return pts


def track_points(prev_gray, curr_gray, prev_pts):
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_params
    )

    if curr_pts is None or status is None:
        return None, None, None

    good_prev = prev_pts[status.flatten() == 1]
    good_curr = curr_pts[status.flatten() == 1]
    good_err = err[status.flatten() == 1] if err is not None else None
    return good_prev, good_curr, good_err


def summarize_motion(prev_pts, curr_pts):
    disp = curr_pts - prev_pts
    dx = disp[:, 0]
    dy = disp[:, 1]
    mag = np.sqrt(dx * dx + dy * dy)

    return {
        "tracked_points": int(len(prev_pts)),
        "mean_dx": float(np.mean(dx)) if len(dx) else None,
        "mean_dy": float(np.mean(dy)) if len(dy) else None,
        "median_dx": float(np.median(dx)) if len(dx) else None,
        "median_dy": float(np.median(dy)) if len(dy) else None,
        "mean_mag": float(np.mean(mag)) if len(mag) else None,
        "median_mag": float(np.median(mag)) if len(mag) else None,
        "max_mag": float(np.max(mag)) if len(mag) else None,
    }


def draw_tracks(frame, prev_pts, curr_pts, color=(0, 255, 0), max_draw=120):
    vis = frame.copy()
    n = min(len(prev_pts), max_draw)

    for i in range(n):
        x1, y1 = prev_pts[i].ravel()
        x2, y2 = curr_pts[i].ravel()

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))

        cv2.arrowedLine(vis, p1, p2, color, 2, tipLength=0.25)
        cv2.circle(vis, p2, 2, (0, 0, 255), -1)

    return vis


def print_stats(label, prev_idx, curr_idx, stats):
    if stats is None:
        print(f"{label} {prev_idx} -> {curr_idx} | tracking failed")
        return

    print(
        f"{label} {prev_idx} -> {curr_idx} | "
        f"tracked={stats['tracked_points']} | "
        f"mean_dx={stats['mean_dx']:.2f} | "
        f"mean_dy={stats['mean_dy']:.2f} | "
        f"mean_mag={stats['mean_mag']:.2f} | "
        f"median_mag={stats['median_mag']:.2f}"
    )


def build_road_trapezoid_mask(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array([
        [int(0.38 * w), int(0.58 * h)],
        [int(0.62 * w), int(0.58 * h)],
        [int(0.72 * w), int(0.82 * h)],
        [int(0.28 * w), int(0.82 * h)],
    ], dtype=np.int32)

    cv2.fillPoly(mask, [pts], 255)
    return mask, pts


def build_underside_band_mask(shape, bbox, underside_line, band_half_height=25, x_margin=15):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if bbox is None or underside_line is None:
        return mask, None

    x1 = int(round(float(bbox["x1"])))
    y1 = int(round(float(bbox["y1"])))
    x2 = int(round(float(bbox["x2"])))
    y2 = int(round(float(bbox["y2"])))

    lx1 = int(round(float(underside_line["x1"])))
    ly1 = int(round(float(underside_line["y1"])))
    lx2 = int(round(float(underside_line["x2"])))
    ly2 = int(round(float(underside_line["y2"])))

    ux1 = max(0, min(lx1, lx2) - x_margin)
    ux2 = min(w - 1, max(lx1, lx2) + x_margin)
    uy = int(round((ly1 + ly2) / 2.0))

    by1 = max(0, uy - band_half_height)
    by2 = min(h - 1, uy + band_half_height)

    ux1 = max(ux1, x1)
    ux2 = min(ux2, x2)
    by1 = max(by1, y1)
    by2 = min(by2, y2)

    if ux2 <= ux1 or by2 <= by1:
        return mask, None

    cv2.rectangle(mask, (ux1, by1), (ux2, by2), 255, -1)
    return mask, {"x1": ux1, "y1": by1, "x2": ux2, "y2": by2}


def save_mask_overlay(frame, road_poly, bridge_band_rect, out_path):
    vis = frame.copy()

    if road_poly is not None:
        cv2.polylines(vis, [road_poly], isClosed=True, color=(255, 255, 0), thickness=2)

    if bridge_band_rect is not None:
        cv2.rectangle(
            vis,
            (bridge_band_rect["x1"], bridge_band_rect["y1"]),
            (bridge_band_rect["x2"], bridge_band_rect["y2"]),
            (0, 255, 255),
            2,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--jsonl", required=True, help="JSONL with bridge_bbox_used and underside_line")
    ap.add_argument("--frames", required=True, help="Comma-separated frame indices")
    ap.add_argument("--out_dir", default="outputs/motion_diagnostics_structured")
    ap.add_argument("--band_half_height", type=int, default=25)
    ap.add_argument("--x_margin", type=int, default=15)
    args = ap.parse_args()

    frame_indices = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    if len(frame_indices) < 2:
        raise ValueError("Provide at least 2 frame indices")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl_by_frame(args.jsonl)
    frames = load_selected_frames(args.video, frame_indices)

    first_idx, first_frame = frames[0]
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    first_rec = records.get(first_idx, {})
    bbox = first_rec.get("bridge_bbox_used")
    underside_line = first_rec.get("underside_line")

    road_mask, road_poly = build_road_trapezoid_mask(first_frame.shape)
    bridge_mask, bridge_band_rect = build_underside_band_mask(
        first_frame.shape,
        bbox=bbox,
        underside_line=underside_line,
        band_half_height=args.band_half_height,
        x_margin=args.x_margin,
    )

    save_mask_overlay(first_frame, road_poly, bridge_band_rect, out_dir / f"masks_{first_idx:06d}.jpg")

    road_pts = detect_features_in_mask(first_gray, road_mask, max_corners=200)
    bridge_pts = detect_features_in_mask(first_gray, bridge_mask, max_corners=200)

    if road_pts is None or len(road_pts) == 0:
        print("No road-trapezoid features detected in first frame")
        road_pts = None
    else:
        print(f"Initial road-trapezoid features on frame {first_idx}: {len(road_pts)}")

    if bridge_pts is None or len(bridge_pts) == 0:
        print("No underside-band bridge features detected in first frame")
        bridge_pts = None
    else:
        print(f"Initial underside-band bridge features on frame {first_idx}: {len(bridge_pts)}")

    prev_gray = first_gray
    prev_idx = first_idx

    for i in range(1, len(frames)):
        curr_idx, curr_frame = frames[i]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if road_pts is not None and len(road_pts) > 0:
            road_prev, road_curr, _ = track_points(prev_gray, curr_gray, road_pts)
            if road_prev is not None and len(road_prev) > 0:
                road_stats = summarize_motion(road_prev.reshape(-1, 2), road_curr.reshape(-1, 2))
                print_stats("ROAD   ", prev_idx, curr_idx, road_stats)
                road_overlay = draw_tracks(curr_frame, road_prev, road_curr, color=(0, 255, 0), max_draw=120)
                cv2.imwrite(str(out_dir / f"road_tracks_{prev_idx:06d}_to_{curr_idx:06d}.jpg"), road_overlay)
                road_pts = road_curr.reshape(-1, 1, 2)
            else:
                print(f"ROAD    {prev_idx} -> {curr_idx} | tracking failed")
                road_pts = None

        if bridge_pts is not None and len(bridge_pts) > 0:
            bridge_prev, bridge_curr, _ = track_points(prev_gray, curr_gray, bridge_pts)
            if bridge_prev is not None and len(bridge_prev) > 0:
                bridge_stats = summarize_motion(bridge_prev.reshape(-1, 2), bridge_curr.reshape(-1, 2))
                print_stats("BRIDGE ", prev_idx, curr_idx, bridge_stats)
                bridge_overlay = draw_tracks(curr_frame, bridge_prev, bridge_curr, color=(255, 0, 255), max_draw=120)
                cv2.imwrite(str(out_dir / f"bridge_tracks_{prev_idx:06d}_to_{curr_idx:06d}.jpg"), bridge_overlay)
                bridge_pts = bridge_curr.reshape(-1, 1, 2)
            else:
                print(f"BRIDGE  {prev_idx} -> {curr_idx} | tracking failed")
                bridge_pts = None

        prev_gray = curr_gray
        prev_idx = curr_idx

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()