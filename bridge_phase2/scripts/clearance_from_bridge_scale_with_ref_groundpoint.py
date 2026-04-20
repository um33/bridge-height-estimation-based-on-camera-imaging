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


def load_records(jsonl_path):
    records = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return sorted(records, key=lambda r: int(r.get("frame_idx", 0)))


def is_valid_record(rec):
    return (
        rec.get("underside_status") == "ok"
        and rec.get("bridge_bbox_used") is not None
        and rec.get("underside_line") is not None
    )


def underside_midpoint(rec):
    ul = rec.get("underside_line")
    if not ul:
        return None
    u = 0.5 * (float(ul["x1"]) + float(ul["x2"]))
    v = 0.5 * (float(ul["y1"]) + float(ul["y2"]))
    return u, v


def underside_length(rec):
    ul = rec.get("underside_line")
    if not ul:
        return None
    dx = float(ul["x2"]) - float(ul["x1"])
    dy = float(ul["y2"]) - float(ul["y1"])
    return float(np.hypot(dx, dy))


def moving_average(values, window=3):
    out = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        chunk = [v for v in values[left:i + 1] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def save_reference_overlay(video_path, ref_frame, u_ref, v_ref, underside_uv, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ref_frame))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return

    vis = frame.copy()

    if underside_uv is not None:
        ux, uy = underside_uv
        cv2.circle(vis, (int(round(ux)), int(round(uy))), 7, (0, 255, 0), -1)

    cv2.circle(vis, (int(round(u_ref)), int(round(v_ref))), 7, (0, 0, 255), -1)
    cv2.line(
        vis,
        (int(round(u_ref)), int(round(v_ref))),
        (int(round(u_ref)), int(round(v_ref - 60))),
        (0, 0, 255),
        2,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--intrinsics", default="configs/intrinsics.yaml")
    ap.add_argument("--cam_height", type=float, required=True)
    ap.add_argument("--pitch_deg", type=float, default=0.0)
    ap.add_argument("--ref_frame", type=int, required=True)
    ap.add_argument("--road_y_ref", type=float, required=True,
                    help="Chosen road pixel y in the reference frame, under the bridge opening")
    ap.add_argument("--road_x_mode", default="underside_midpoint",
                    choices=["underside_midpoint", "image_center"])
    ap.add_argument("--frame_min", type=int, default=None)
    ap.add_argument("--frame_max", type=int, default=None)
    ap.add_argument("--use_smoothing", action="store_true")
    ap.add_argument("--overlay_out", default="outputs/reference_groundpoint_overlay.jpg")
    args = ap.parse_args()

    K, dist, intr = load_intrinsics(args.intrinsics)
    img_w = int(intr["image_width"])

    theta = np.deg2rad(args.pitch_deg)
    R = rot_x(theta)
    C = np.array([0.0, args.cam_height, 0.0], dtype=np.float64)

    records = load_records(args.jsonl)

    if args.frame_min is not None:
        records = [r for r in records if int(r["frame_idx"]) >= args.frame_min]
    if args.frame_max is not None:
        records = [r for r in records if int(r["frame_idx"]) <= args.frame_max]

    ref_rec = None
    for rec in records:
        if int(rec["frame_idx"]) == args.ref_frame and is_valid_record(rec):
            ref_rec = rec
            break

    if ref_rec is None:
        raise RuntimeError(f"Reference frame {args.ref_frame} is missing or invalid.")

    ref_under_uv = underside_midpoint(ref_rec)
    ref_under_len = underside_length(ref_rec)
    if ref_under_uv is None or ref_under_len is None or ref_under_len <= 1e-9:
        raise RuntimeError("Reference frame missing underside midpoint or valid underside length.")

    if args.road_x_mode == "underside_midpoint":
        u_ref = float(ref_under_uv[0])
    else:
        u_ref = img_w / 2.0

    v_ref = float(args.road_y_ref)

    # Estimate reference depth from one road point in the reference frame
    road_ray, (u_ref_ud, v_ref_ud) = pixel_to_world_ray(K, dist, R, u_ref, v_ref)
    Pg = intersect_ground(road_ray, cam_height_m=args.cam_height)
    if Pg is None:
        raise RuntimeError("Chosen reference road point does not intersect the road plane. Pick a lower road_y_ref.")

    z_ref_m = float(Pg[2])
    if z_ref_m <= 1e-9:
        raise RuntimeError("Estimated z_ref_m is non-positive. Pick a better reference road point.")

    save_reference_overlay(
        video_path=args.video,
        ref_frame=args.ref_frame,
        u_ref=u_ref,
        v_ref=v_ref,
        underside_uv=ref_under_uv,
        out_path=args.overlay_out
    )

    raw_depth_proxy = []
    for rec in records:
        if is_valid_record(rec):
            curr_len = underside_length(rec)
            raw_depth_proxy.append(ref_under_len / curr_len if curr_len and curr_len > 1e-9 else None)
        else:
            raw_depth_proxy.append(None)

    smoothed_depth_proxy = moving_average(raw_depth_proxy, window=3)

    out_lines = []

    print("\nCLEARANCE FROM BRIDGE SCALE + REFERENCE GROUND POINT\n")
    print(f"Estimated z_ref_m from frame {args.ref_frame}: {z_ref_m:.3f} m")
    print(f"Reference road pixel: u={u_ref:.1f}, v={v_ref:.1f}")
    print("\nframe | time | under_len | rel_depth | Z_t | clearance | status")
    print("-" * 95)

    for i, rec in enumerate(records):
        rec_out = dict(rec)
        rel_depth = smoothed_depth_proxy[i] if args.use_smoothing else raw_depth_proxy[i]

        if not is_valid_record(rec):
            rec_out["clearance_m_estimate"] = None
            rec_out["clearance_status"] = "no_valid_underside_or_bbox"
            rec_out["clearance_debug"] = {
                "method": "bridge_scale_plus_reference_groundpoint",
                "ref_frame": args.ref_frame,
                "estimated_z_ref_m": z_ref_m,
                "pitch_deg": args.pitch_deg,
                "cam_height_m": args.cam_height,
                "relative_depth_proxy": None,
                "z_t_m": None,
            }
            out_lines.append(json.dumps(rec_out))
            continue

        uv = underside_midpoint(rec)
        if uv is None or rel_depth is None:
            rec_out["clearance_m_estimate"] = None
            rec_out["clearance_status"] = "no_relative_depth"
            rec_out["clearance_debug"] = {
                "method": "bridge_scale_plus_reference_groundpoint",
                "ref_frame": args.ref_frame,
                "estimated_z_ref_m": z_ref_m,
                "pitch_deg": args.pitch_deg,
                "cam_height_m": args.cam_height,
                "relative_depth_proxy": rel_depth,
                "z_t_m": None,
            }
            out_lines.append(json.dumps(rec_out))
            continue

        z_t = float(z_ref_m * rel_depth)

        u, v = uv
        ray_world, (u_ud, v_ud) = pixel_to_world_ray(K, dist, R, u, v)

        if abs(ray_world[2]) < 1e-9:
            rec_out["clearance_m_estimate"] = None
            rec_out["clearance_status"] = "bad_ray_z"
            out_lines.append(json.dumps(rec_out))
            continue

        t = z_t / ray_world[2]
        if t <= 0:
            rec_out["clearance_m_estimate"] = None
            rec_out["clearance_status"] = "negative_t"
            out_lines.append(json.dumps(rec_out))
            continue

        P_under = C + t * ray_world
        clearance = float(P_under[1])

        rec_out["clearance_m_estimate"] = clearance
        rec_out["clearance_status"] = "ok"
        rec_out["clearance_debug"] = {
            "method": "bridge_scale_plus_reference_groundpoint",
            "ref_frame": args.ref_frame,
            "estimated_z_ref_m": z_ref_m,
            "reference_ground_pixel": [u_ref, v_ref],
            "reference_ground_pixel_undistorted": [u_ref_ud, v_ref_ud],
            "reference_ground_point_3d": Pg.tolist(),
            "pitch_deg": args.pitch_deg,
            "cam_height_m": args.cam_height,
            "used_smoothing": bool(args.use_smoothing),
            "underside_len": underside_length(rec),
            "ref_underside_len": ref_under_len,
            "relative_depth_proxy": rel_depth,
            "z_t_m": z_t,
            "underside_pixel": [u, v],
            "underside_pixel_undistorted": [u_ud, v_ud],
            "ray_world": ray_world.tolist(),
            "underside_point_estimated": P_under.tolist(),
        }

        print(
            f"{int(rec['frame_idx']):5d} | "
            f"{float(rec.get('time_sec', 0.0)):.1f} | "
            f"{underside_length(rec):9.2f} | "
            f"{rel_depth:9.3f} | "
            f"{z_t:6.2f} | "
            f"{clearance:9.3f} | ok"
        )

        out_lines.append(json.dumps(rec_out))

    Path(args.out_jsonl).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nSaved JSONL to: {args.out_jsonl}")
    print(f"Saved reference overlay to: {args.overlay_out}")


if __name__ == "__main__":
    main()