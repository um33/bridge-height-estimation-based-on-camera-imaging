import os
import json
from pathlib import Path

import numpy as np
import yaml

def load_intrinsics(path: str):
    """Load camera intrinsics from YAML and return (K, dist)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)  # not used in this baseline
    return K, dist, data

def pixel_to_ray(K, u, v):
    """
    Convert pixel (u,v) to a normalized ray direction in a Y-up convention.
    OpenCV pixels have +v downward, so we flip sign to make +Y upward.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) / fx
    y = -(v - cy) / fy  # flip vertical axis
    ray = np.array([x, y, 1.0], dtype=np.float64)
    ray /= np.linalg.norm(ray)
    return ray

def rot_x(theta):
    """Rotation matrix around x-axis (pitch). theta in radians."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)

def estimate_distance_on_road(K, img_w, img_h, cam_height_m=1.2, pitch_deg=0, v_frac=0.95):
    """
    Estimate ground distance by intersecting a chosen image pixel ray with the road plane.
    Road plane: Y = 0 (Y-up world).
    Camera center: (0, cam_height_m, 0).
    We choose pixel (u=mid, v=v_frac*H) which is usually on the road.
    Returns ground distance in meters or None.
    """
    # Choose a road pixel (stable baseline)
    u = img_w / 2.0
    v = img_h * v_frac

    # Ray in camera coords (Y-up convention)
    ray_cam = pixel_to_ray(K, u, v)

    # Rotate by pitch (downward pitch -> larger pitch_deg)
    theta = np.deg2rad(pitch_deg)
    R = rot_x(theta)
    ray_world = R @ ray_cam

    # Camera origin in world coords
    C = np.array([0.0, cam_height_m, 0.0], dtype=np.float64)

    # Intersect with road plane Y=0: (C + t*ray).y = 0
    denom = ray_world[1]
    if abs(denom) < 1e-9:
        return None  # ray parallel to ground

    t = -C[1] / denom
    if t <= 0:
        return None  # intersection behind camera (bad geometry / wrong assumptions)

    P = C + t * ray_world

    # Ground distance (horizontal distance in XZ plane)
    dist = float(np.sqrt(P[0]**2 + P[2]**2))
    return dist

def bbox_bottom_center(pred):
    """
    Roboflow bbox is center-based: x,y,width,height (pixels).
    Bottom-center pixel of bbox.
    """
    u = float(pred["x"])
    v = float(pred["y"]) + float(pred["height"]) / 2.0
    return u, v

def bbox_corners(pred):
    """Convert center bbox to corners (x1,y1,x2,y2) in pixels."""
    x = float(pred["x"]); y = float(pred["y"])
    w = float(pred["width"]); h = float(pred["height"])
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return x1, y1, x2, y2

def best_prediction(preds, class_name="Bridge"):
    """Pick highest-confidence prediction of a class (defaults to 'Bridge')."""
    best = None
    best_conf = -1.0
    for p in preds:
        if p.get("class") != class_name:
            continue
        conf = float(p.get("confidence", 0.0))
        if conf > best_conf:
            best_conf = conf
            best = p
    return best, best_conf

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Input JSONL from video sampling + detection")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL with added baseline distance + bridge pixel info")
    ap.add_argument("--intrinsics", default="configs/intrinsics.yaml", help="Intrinsics YAML path")
    ap.add_argument("--cam_height", type=float, default=1.4, help="Camera height above road in meters")
    ap.add_argument("--pitch_deg", type=float, default=10.0, help="Downward pitch estimate in degrees")
    ap.add_argument("--v_frac", type=float, default=0.95, help="Road pixel vertical fraction (0.0 top -> 1.0 bottom)")
    ap.add_argument("--class_name", default="Bridge", help="Class name to use for bridge predictions")
    ap.add_argument("--conf_th", type=float, default=0.35, help="Min confidence to keep bridge pixel info")
    args = ap.parse_args()

    K, dist, intr_data = load_intrinsics(args.intrinsics)
    img_w = int(intr_data["image_width"])
    img_h = int(intr_data["image_height"])

    lines = Path(args.in_jsonl).read_text(encoding="utf-8").splitlines()
    out_lines = []

    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)

        # 1) Baseline stable distance to a road pixel (bottom-center)
        d = estimate_distance_on_road(
            K, img_w, img_h,
            cam_height_m=args.cam_height,
            pitch_deg=args.pitch_deg,
            v_frac=args.v_frac
        )
        rec["distance_m_baseline"] = d
        rec["distance_params"] = {
            "cam_height_m": args.cam_height,
            "pitch_deg": args.pitch_deg,
            "v_frac": args.v_frac
        }

        # 2) Store bridge bbox pixel info (for later Phase 3 underside extraction)
        preds = rec.get("predictions", [])
        best, best_conf = best_prediction(preds, class_name=args.class_name)

        bridge_px = None
        bridge_bbox = None

        if best is not None and best_conf >= args.conf_th:
            u_b, v_b = bbox_bottom_center(best)
            x1, y1, x2, y2 = bbox_corners(best)
            bridge_px = {"u": u_b, "v": v_b, "conf": best_conf}
            bridge_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": best_conf}

        rec["bridge_pixel_bottom_center"] = bridge_px
        rec["bridge_bbox"] = bridge_bbox
        rec["bridge_params"] = {
            "class_name": args.class_name,
            "conf_th": args.conf_th
        }

        out_lines.append(json.dumps(rec))

    Path(args.out_jsonl).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print("✅ Wrote:", args.out_jsonl)

if __name__ == "__main__":
    main()


# python scripts/add_distance_to_jsonl.py \
#   --in_jsonl outputs/video_detections_every_1s.jsonl \
#   --out_jsonl outputs/bridge_detections_every_1s_with_bridge_distance.jsonl \
#   --cam_height 1.7 \
#   --pitch_deg 0 \ 
#   --conf_th 0.35