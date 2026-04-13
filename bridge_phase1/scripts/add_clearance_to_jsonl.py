import json
from pathlib import Path
import numpy as np
import yaml

def load_intrinsics(path: str):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist, data

def pixel_to_ray_y_up(K, u, v):
    fx = K[0,0]; fy = K[1,1]
    cx = K[0,2]; cy = K[1,2]
    x = (u - cx) / fx
    y = -(v - cy) / fy
    ray = np.array([x, y, 1.0], dtype=np.float64)
    ray /= np.linalg.norm(ray)
    return ray

def rot_x(theta_rad):
    c = np.cos(theta_rad); s = np.sin(theta_rad)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def intersect_ground(ray_world, cam_height_m):
    C = np.array([0.0, cam_height_m, 0.0], dtype=np.float64)
    denom = ray_world[1]
    if abs(denom) < 1e-9:
        return None
    t = -C[1] / denom
    if t <= 0:
        return None
    return C + t * ray_world

def ground_distance(P):
    return float(np.sqrt(P[0]**2 + P[2]**2))

def underside_midpoint(rec):
    ul = rec.get("underside_line")
    if not ul:
        return None
    u = 0.5 * (float(ul["x1"]) + float(ul["x2"]))
    v = 0.5 * (float(ul["y1"]) + float(ul["y2"]))
    return u, v

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--intrinsics", default="configs/intrinsics.yaml")
    ap.add_argument("--cam_height", type=float, default=1.7)
    ap.add_argument("--pitch_deg", type=float, default=0.0)

    # NEW: choose a road pixel to get a stable distance reference
    ap.add_argument("--v_frac", type=float, default=0.95,
                    help="road reference pixel vertical position as fraction of image height (0-1)")

    args = ap.parse_args()

    K, dist, intr = load_intrinsics(args.intrinsics)
    img_w = int(intr["image_width"])
    img_h = int(intr["image_height"])

    theta = np.deg2rad(args.pitch_deg)
    R = rot_x(theta)

    out_lines = []
    for line in Path(args.in_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)

        # 1) Stable forward distance: intersect road using bottom-center pixel
        u_r = img_w / 2.0
        v_r = img_h * args.v_frac
        ray_r = R @ pixel_to_ray_y_up(K, u_r, v_r)
        Pg = intersect_ground(ray_r, cam_height_m=args.cam_height)
        road_dist = ground_distance(Pg) if Pg is not None else None

        # 2) Underside ray -> estimate clearance at same forward distance (road_dist)
        clearance = None
        uv = underside_midpoint(rec)
        if uv is not None and road_dist is not None:
            u_u, v_u = uv
            ray_u = R @ pixel_to_ray_y_up(K, u_u, v_u)

            # need ray to point forward
            if ray_u[2] > 1e-6:
                t = road_dist / ray_u[2]
                C = np.array([0.0, args.cam_height, 0.0], dtype=np.float64)
                P_u = C + t * ray_u
                clearance = float(P_u[1])  # height above road plane Y=0

        rec["road_dist_m"] = road_dist
        rec["clearance_m_estimate"] = clearance
        rec["clearance_params"] = {
            "cam_height_m": args.cam_height,
            "pitch_deg": args.pitch_deg,
            "v_frac": args.v_frac,
            "method": "road_pixel_ground_dist + underside_midpoint_ray_at_same_Z"
        }

        out_lines.append(json.dumps(rec))

    Path(args.out_jsonl).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print("✅ Wrote:", args.out_jsonl)

if __name__ == "__main__":
    main()


# python scripts/add_clearance_to_jsonl.py \
#   --in_jsonl outputs/video_detections_every_1s_with_underside.jsonl \
#   --out_jsonl outputs/video_detections_every_1s_with_clearance.jsonl \
#   --cam_height 1.7 \
#   --pitch_deg 0 \
#   --v_frac 0.85