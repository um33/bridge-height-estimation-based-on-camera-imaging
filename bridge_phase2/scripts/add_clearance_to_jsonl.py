import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_intrinsics(path: str):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return K, dist, data


def rot_x(theta_rad: float):
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c,   -s],
        [0.0, s,    c]
    ], dtype=np.float64)


def estimate_horizon_y(K, pitch_deg):
    fy = K[1, 1]
    cy = K[1, 2]
    theta = np.deg2rad(pitch_deg)
    return float(cy - fy * np.tan(theta))


def undistort_pixel(K, dist, u, v):
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, K, dist, P=K)
    u_corr, v_corr = undist[0, 0]
    return float(u_corr), float(v_corr)


def pixel_to_ray_y_up(K, u, v):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) / fx
    y = -(v - cy) / fy
    ray = np.array([x, y, 1.0], dtype=np.float64)
    ray /= np.linalg.norm(ray)
    return ray


def pixel_to_world_ray(K, dist, R, u, v):
    u_corr, v_corr = undistort_pixel(K, dist, u, v)
    ray_cam = pixel_to_ray_y_up(K, u_corr, v_corr)
    ray_world = R @ ray_cam
    ray_world /= np.linalg.norm(ray_world)
    return ray_world, (u_corr, v_corr)


def intersect_ground(ray_world, cam_height_m):
    C = np.array([0.0, cam_height_m, 0.0], dtype=np.float64)
    denom = ray_world[1]

    if abs(denom) < 1e-9:
        return None

    t = -C[1] / denom
    if t <= 0:
        return None

    return C + t * ray_world


def underside_midpoint(rec):
    ul = rec.get("underside_line")
    if not ul:
        return None
    u = 0.5 * (float(ul["x1"]) + float(ul["x2"]))
    v = 0.5 * (float(ul["y1"]) + float(ul["y2"]))
    return u, v


def get_bbox_used(rec):
    bbox = rec.get("bridge_bbox_used")
    if bbox is None:
        return None
    try:
        return {
            "x1": float(bbox["x1"]),
            "y1": float(bbox["y1"]),
            "x2": float(bbox["x2"]),
            "y2": float(bbox["y2"]),
            "conf": float(bbox.get("conf", 0.0)),
        }
    except Exception:
        return None


def choose_horizon_aware_anchor_pixel(
    rec,
    img_h,
    horizon_y,
    anchor_offset_px=220.0,
    horizon_margin_px=40.0,
    bottom_margin_px=30.0
):
    """
    Choose anchor below the bridge and below the horizon.
    """
    uv_under = underside_midpoint(rec)
    if uv_under is None:
        return None

    u_mid, v_under = uv_under

    min_valid_y = horizon_y + horizon_margin_px
    proposed_y = v_under + anchor_offset_px
    v_anchor = max(proposed_y, min_valid_y)
    v_anchor = min(v_anchor, img_h - bottom_margin_px)

    if v_anchor <= min_valid_y:
        return None

    return float(u_mid), float(v_anchor)


def estimate_clearance_for_record(
    rec, K, dist, R, cam_height_m, img_h, pitch_deg,
    anchor_offset_px, horizon_margin_px, bottom_margin_px
):
    C = np.array([0.0, cam_height_m, 0.0], dtype=np.float64)

    uv_under = underside_midpoint(rec)
    if uv_under is None:
        return {
            "status": "no_underside",
            "clearance_m": None
        }

    horizon_y = estimate_horizon_y(K, pitch_deg)

    uv_anchor = choose_horizon_aware_anchor_pixel(
        rec=rec,
        img_h=img_h,
        horizon_y=horizon_y,
        anchor_offset_px=anchor_offset_px,
        horizon_margin_px=horizon_margin_px,
        bottom_margin_px=bottom_margin_px
    )

    if uv_anchor is None:
        return {
            "status": "no_anchor_pixel",
            "clearance_m": None,
            "horizon_y": horizon_y
        }

    u_under, v_under = uv_under
    u_anchor, v_anchor = uv_anchor

    ray_anchor, (u_anchor_ud, v_anchor_ud) = pixel_to_world_ray(K, dist, R, u_anchor, v_anchor)
    Pg = intersect_ground(ray_anchor, cam_height_m=cam_height_m)

    if Pg is None:
        return {
            "status": "anchor_no_ground_intersection",
            "clearance_m": None,
            "horizon_y": horizon_y,
            "anchor_pixel": [u_anchor, v_anchor],
            "anchor_pixel_undistorted": [u_anchor_ud, v_anchor_ud],
            "ray_anchor_world": ray_anchor.tolist()
        }

    z_ref = float(Pg[2])
    if z_ref <= 1e-6:
        return {
            "status": "anchor_nonpositive_z",
            "clearance_m": None,
            "horizon_y": horizon_y,
            "anchor_pixel": [u_anchor, v_anchor],
            "anchor_pixel_undistorted": [u_anchor_ud, v_anchor_ud],
            "ground_point": Pg.tolist(),
            "z_ref_m": z_ref
        }

    ray_under, (u_under_ud, v_under_ud) = pixel_to_world_ray(K, dist, R, u_under, v_under)

    if abs(ray_under[2]) < 1e-9:
        return {
            "status": "underside_bad_z_component",
            "clearance_m": None,
            "horizon_y": horizon_y,
            "underside_pixel": [u_under, v_under],
            "underside_pixel_undistorted": [u_under_ud, v_under_ud],
            "z_ref_m": z_ref
        }

    t = z_ref / ray_under[2]
    if t <= 0:
        return {
            "status": "underside_negative_t",
            "clearance_m": None,
            "horizon_y": horizon_y,
            "underside_pixel": [u_under, v_under],
            "underside_pixel_undistorted": [u_under_ud, v_under_ud],
            "z_ref_m": z_ref
        }

    P_under = C + t * ray_under
    clearance = float(P_under[1])

    return {
        "status": "ok",
        "clearance_m": clearance,
        "horizon_y": horizon_y,
        "anchor_pixel": [u_anchor, v_anchor],
        "anchor_pixel_undistorted": [u_anchor_ud, v_anchor_ud],
        "underside_pixel": [u_under, v_under],
        "underside_pixel_undistorted": [u_under_ud, v_under_ud],
        "ground_point": Pg.tolist(),
        "z_ref_m": z_ref,
        "anchor_ground_distance_m": float(np.sqrt(Pg[0] ** 2 + Pg[2] ** 2)),
        "underside_point_estimated": P_under.tolist(),
        "ray_anchor_world": ray_anchor.tolist(),
        "ray_under_world": ray_under.tolist()
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--intrinsics", default="configs/intrinsics.yaml")
    ap.add_argument("--cam_height", type=float, required=True)
    ap.add_argument("--pitch_deg", type=float, default=0.0)
    ap.add_argument("--anchor_offset_px", type=float, default=220.0)
    ap.add_argument("--horizon_margin_px", type=float, default=40.0)
    ap.add_argument("--bottom_margin_px", type=float, default=30.0)
    args = ap.parse_args()

    K, dist, intr = load_intrinsics(args.intrinsics)
    img_h = int(intr["image_height"])

    theta = np.deg2rad(args.pitch_deg)
    R = rot_x(theta)

    out_lines = []

    for line in Path(args.in_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        rec = json.loads(line)

        result = estimate_clearance_for_record(
            rec=rec,
            K=K,
            dist=dist,
            R=R,
            cam_height_m=args.cam_height,
            img_h=img_h,
            pitch_deg=args.pitch_deg,
            anchor_offset_px=args.anchor_offset_px,
            horizon_margin_px=args.horizon_margin_px,
            bottom_margin_px=args.bottom_margin_px
        )

        rec["clearance_m_estimate"] = result.get("clearance_m")
        rec["clearance_status"] = result.get("status")
        rec["clearance_debug"] = {
            "cam_height_m": args.cam_height,
            "pitch_deg": args.pitch_deg,
            "anchor_offset_px": args.anchor_offset_px,
            "horizon_margin_px": args.horizon_margin_px,
            "bottom_margin_px": args.bottom_margin_px,
            "method": "horizon_aware_road_anchor_same_Z_as_underside",
            "horizon_y": result.get("horizon_y"),
            "anchor_pixel": result.get("anchor_pixel"),
            "anchor_pixel_undistorted": result.get("anchor_pixel_undistorted"),
            "underside_pixel": result.get("underside_pixel"),
            "underside_pixel_undistorted": result.get("underside_pixel_undistorted"),
            "z_ref_m": result.get("z_ref_m"),
            "anchor_ground_distance_m": result.get("anchor_ground_distance_m"),
            "ground_point": result.get("ground_point"),
            "underside_point_estimated": result.get("underside_point_estimated"),
            "ray_anchor_world": result.get("ray_anchor_world"),
            "ray_under_world": result.get("ray_under_world")
        }

        out_lines.append(json.dumps(rec))

    Path(args.out_jsonl).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"✅ Wrote: {args.out_jsonl}")


if __name__ == "__main__":
    main()

