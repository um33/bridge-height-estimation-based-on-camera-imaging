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


def underside_y(rec):
    uv = underside_midpoint(rec)
    if uv is None:
        return None
    return float(uv[1])


def bbox_h(rec):
    b = rec.get("bridge_bbox_used")
    if not b:
        return None
    return float(b["y2"]) - float(b["y1"])


def bbox_w(rec):
    b = rec.get("bridge_bbox_used")
    if not b:
        return None
    return float(b["x2"]) - float(b["x1"])


def moving_average(values, window=3):
    out = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        chunk = [v for v in values[left:i + 1] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def safe_ratio(ref_val, curr_val):
    if ref_val is None or curr_val is None:
        return None
    if abs(curr_val) <= 1e-9:
        return None
    return float(ref_val / curr_val)


def combine_depth_proxies(proxy_dict, weights):
    vals = []
    wsum = 0.0
    for k, v in proxy_dict.items():
        if v is None:
            continue
        w = float(weights.get(k, 0.0))
        if w <= 0:
            continue
        vals.append(w * float(v))
        wsum += w
    if wsum <= 1e-9:
        return None
    return float(sum(vals) / wsum)


def manual_ground_reference(K, dist, R, cam_height_m, u_ref, v_ref):
    ray_world, _ = pixel_to_world_ray(K, dist, R, u_ref, v_ref)
    Pg = intersect_ground(ray_world, cam_height_m=cam_height_m)
    return Pg


def auto_pick_ground_reference(rec, K, dist, R, cam_height_m, img_h, img_w):
    """
    General, bridge-aware ground reference selection.

    Strategy:
    - search under the detected bridge span, anchored at underside midpoint
    - use a moderate main search band
    - if needed, use a more permissive fallback band
    - build valid ground candidates
    - keep the depth-consistent cluster around the median
    - choose a representative candidate with a balanced score
    """

    under_uv = underside_midpoint(rec)
    bbox = rec.get("bridge_bbox_used")

    if under_uv is None or bbox is None:
        return None, None, None, {"reason": "missing_underside_or_bbox"}

    u_under, v_under = under_uv
    x1 = float(bbox["x1"])
    x2 = float(bbox["x2"])
    y2 = float(bbox["y2"])

    u_anchor = float(np.clip(u_under, x1 + 10.0, x2 - 10.0))

    u_cols = []
    for offset in (0, -20, 20, -40, 40):
        u = u_anchor + offset
        if x1 <= u <= x2 and 0 <= u < img_w:
            u_cols.append(float(u))

    if not u_cols:
        u_cols = [float(np.clip(u_anchor, 0, img_w - 1))]

    v_top = int(max(v_under + 25, y2 + 5))
    v_bot = int(min(v_top + 220, img_h * 0.78, img_h - 20))

    if v_bot <= v_top:
        return None, None, None, {
            "reason": "empty_search_band",
            "v_top": v_top,
            "v_bot": v_bot,
            "img_h": img_h,
            "u_anchor": float(u_anchor),
            "u_under": float(u_under),
            "v_under": float(v_under),
        }

    candidates = []

    for u_col in u_cols:
        for v_row in range(v_top, v_bot + 1, 5):
            ray_world, _ = pixel_to_world_ray(K, dist, R, u_col, float(v_row))
            Pg = intersect_ground(ray_world, cam_height_m=cam_height_m)
            if Pg is None:
                continue

            x = float(Pg[0])
            y = float(Pg[1])
            z = float(Pg[2])

            if z <= 0:
                continue
            if abs(y) > 0.20:
                continue
            if not (6.0 <= z <= 40.0):
                continue
            if abs(x) > 1.2:
                continue

            candidates.append({
                "u": float(u_col),
                "v": float(v_row),
                "Pg": Pg,
                "x": x,
                "y": y,
                "z": z,
            })

    v_bot_fallback = int(min(v_top + 320, img_h * 0.86, img_h - 20))

    if not candidates:
        for u_col in u_cols:
            for v_row in range(v_top, v_bot_fallback + 1, 5):
                ray_world, _ = pixel_to_world_ray(K, dist, R, u_col, float(v_row))
                Pg = intersect_ground(ray_world, cam_height_m=cam_height_m)
                if Pg is None:
                    continue

                x = float(Pg[0])
                y = float(Pg[1])
                z = float(Pg[2])

                if z <= 0:
                    continue
                if abs(y) > 0.35:
                    continue
                if not (4.0 <= z <= 45.0):
                    continue
                if abs(x) > 2.5:
                    continue

                candidates.append({
                    "u": float(u_col),
                    "v": float(v_row),
                    "Pg": Pg,
                    "x": x,
                    "y": y,
                    "z": z,
                })

    if not candidates:
        return None, None, None, {
            "reason": "no_candidate_after_fallback",
            "u_cols_used": [float(u) for u in u_cols],
            "v_top": int(v_top),
            "v_bot": int(v_bot),
            "v_bot_fallback": int(v_bot_fallback),
            "u_anchor": float(u_anchor),
            "u_under": float(u_under),
            "v_under": float(v_under),
        }

    z_values = np.array([c["z"] for c in candidates], dtype=np.float64)
    z_median = float(np.median(z_values))
    z_std = float(np.std(z_values))

    z_tol = max(1.0, 0.75 * z_std)
    filtered = [c for c in candidates if abs(c["z"] - z_median) <= z_tol]

    if not filtered:
        filtered = candidates

    def candidate_score(c):
        return (
            2.0 * abs(c["z"] - z_median)
            + 1.5 * abs(c["x"])
            + 0.03 * abs(c["u"] - u_under)
            + 0.01 * (c["v"] - v_top)
        )

    best = min(filtered, key=candidate_score)

    Pg_out = np.array(best["Pg"], dtype=np.float64).copy()
    Pg_out[2] = float(best["z"])

    return best["u"], best["v"], Pg_out, {
        "reason": "ok_balanced_clustered",
        "u_ref": float(best["u"]),
        "v_ref": float(best["v"]),
        "candidate_x": float(best["x"]),
        "candidate_y": float(best["y"]),
        "candidate_z": float(best["z"]),
        "median_z": float(z_median),
        "z_std": float(z_std),
        "z_tol": float(z_tol),
        "num_candidates": int(len(candidates)),
        "num_filtered": int(len(filtered)),
        "u_cols_used": [float(u) for u in u_cols],
        "v_top": int(v_top),
        "v_bot": int(v_bot),
        "v_bot_fallback": int(v_bot_fallback),
        "u_anchor": float(u_anchor),
        "u_under": float(u_under),
        "v_under": float(v_under),
    }


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
    ap.add_argument("--frame_min", type=int, default=None)
    ap.add_argument("--frame_max", type=int, default=None)
    ap.add_argument("--use_smoothing", action="store_true")
    ap.add_argument("--overlay_out", default="outputs/reference_groundpoint_overlay.jpg")
    ap.add_argument("--manual_ref_u", type=float, default=None)
    ap.add_argument("--manual_ref_v", type=float, default=None)

    args = ap.parse_args()

    K, dist, intr = load_intrinsics(args.intrinsics)
    img_h = int(intr["image_height"])
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
    ref_under_y_v = underside_y(ref_rec)
    ref_bbox_height = bbox_h(ref_rec)
    ref_bbox_width = bbox_w(ref_rec)

    if ref_under_uv is None or ref_under_len is None or ref_under_len <= 1e-9:
        raise RuntimeError("Reference frame missing underside midpoint or valid underside length.")

    if args.manual_ref_u is not None and args.manual_ref_v is not None:
        u_ref = float(args.manual_ref_u)
        v_ref = float(args.manual_ref_v)
        Pg = manual_ground_reference(K, dist, R, args.cam_height, u_ref, v_ref)
        ref_pick_debug = {
            "reason": "manual_reference_point",
            "manual_ref_u": u_ref,
            "manual_ref_v": v_ref,
        }
    else:
        u_ref, v_ref, Pg, ref_pick_debug = auto_pick_ground_reference(
            ref_rec, K, dist, R, args.cam_height, img_h, img_w
        )

    if Pg is None:
        raise RuntimeError(
            f"Reference ground selection failed. Debug: {ref_pick_debug}"
        )

    road_ray, (u_ref_ud, v_ref_ud) = pixel_to_world_ray(K, dist, R, u_ref, v_ref)
    z_ref_m = float(Pg[2])

    if not (4.0 <= z_ref_m <= 30.0):
        raise RuntimeError(
            f"Estimated z_ref_m is unrealistic: {z_ref_m:.3f} m. Debug: {ref_pick_debug}"
        )

    print("underside midpoint:", ref_under_uv)
    print(f"underside_y (ref v): {ref_under_y_v:.1f} px")
    print("selected road pixel:", (u_ref, v_ref))
    print("reference 3D point:", Pg.tolist())
    print("reference pick debug:", ref_pick_debug)
    print(
        f"selected candidate world x={ref_pick_debug.get('candidate_x', float('nan')):.3f}, "
        f"candidate z={ref_pick_debug.get('candidate_z', float('nan')):.3f}, "
        f"median z={ref_pick_debug.get('median_z', float('nan')):.3f}"
    )

    save_reference_overlay(
        video_path=args.video,
        ref_frame=args.ref_frame,
        u_ref=u_ref,
        v_ref=v_ref,
        underside_uv=ref_under_uv,
        out_path=args.overlay_out
    )

    WEIGHTS = {
    "under_y":   0.70,
    "under_len": 0.00,
    "bbox_h":    0.30,
    "bbox_w":    0.00,
    }

    raw_proxy_under_y = []
    raw_proxy_under_len = []
    raw_proxy_bbox_h = []
    raw_proxy_bbox_w = []
    raw_proxy_combined = []

    for rec in records:
        if not is_valid_record(rec):
            raw_proxy_under_y.append(None)
            raw_proxy_under_len.append(None)
            raw_proxy_bbox_h.append(None)
            raw_proxy_bbox_w.append(None)
            raw_proxy_combined.append(None)
            continue

        curr_under_y_v = underside_y(rec)
        curr_under_len = underside_length(rec)
        curr_bbox_h = bbox_h(rec)
        curr_bbox_w = bbox_w(rec)

        if ref_under_y_v is not None and curr_under_y_v is not None:
            ref_inv = img_h - ref_under_y_v
            curr_inv = img_h - curr_under_y_v
            p_under_y = safe_ratio(ref_inv, curr_inv)
        else:
            p_under_y = None

        p_under_len = safe_ratio(ref_under_len, curr_under_len)
        p_bbox_h = safe_ratio(ref_bbox_height, curr_bbox_h)
        p_bbox_w = safe_ratio(ref_bbox_width, curr_bbox_w)

        combined = combine_depth_proxies(
            {
                "under_y": p_under_y,
                "under_len": p_under_len,
                "bbox_h": p_bbox_h,
                "bbox_w": p_bbox_w,
            },
            weights=WEIGHTS,
        )

        raw_proxy_under_y.append(p_under_y)
        raw_proxy_under_len.append(p_under_len)
        raw_proxy_bbox_h.append(p_bbox_h)
        raw_proxy_bbox_w.append(p_bbox_w)
        raw_proxy_combined.append(combined)

    sm_under_y = moving_average(raw_proxy_under_y, window=3)
    sm_under_len = moving_average(raw_proxy_under_len, window=3)
    sm_bbox_h = moving_average(raw_proxy_bbox_h, window=3)
    sm_bbox_w = moving_average(raw_proxy_bbox_w, window=3)
    sm_combined = moving_average(raw_proxy_combined, window=3)

    out_lines = []

    print("\nCLEARANCE — MEDIAN GROUND REF + COMBINED DEPTH PROXY v2\n")
    print(
        f"Weights: under_y={WEIGHTS['under_y']}, under_len={WEIGHTS['under_len']}, "
        f"bbox_h={WEIGHTS['bbox_h']}, bbox_w={WEIGHTS['bbox_w']}"
    )
    print(f"z_ref_m (frame {args.ref_frame}): {z_ref_m:.3f} m")
    print(f"Reference road pixel: u={u_ref:.1f}, v={v_ref:.1f}")
    print(f"Ground ref selection: {ref_pick_debug.get('reason')}")
    print(
        f"\n{'frame':>5} | {'t':>4} | {'u_y_px':>7} | {'u_len':>6} | "
        f"{'p_uy':>6} | {'p_ul':>6} | {'p_bh':>6} | {'p_bw':>5} | "
        f"{'rel_d':>6} | {'Z_t':>6} | {'clear':>7} | status"
    )
    print("-" * 118)

    for i, rec in enumerate(records):
        rec_out = dict(rec)

        if args.use_smoothing:
            rel_depth = sm_combined[i]
            d_under_y = sm_under_y[i]
            d_under_len = sm_under_len[i]
            d_bbox_h = sm_bbox_h[i]
            d_bbox_w = sm_bbox_w[i]
        else:
            rel_depth = raw_proxy_combined[i]
            d_under_y = raw_proxy_under_y[i]
            d_under_len = raw_proxy_under_len[i]
            d_bbox_h = raw_proxy_bbox_h[i]
            d_bbox_w = raw_proxy_bbox_w[i]

        if not is_valid_record(rec):
            rec_out["clearance_m_estimate"] = None
            rec_out["clearance_status"] = "no_valid_underside_or_bbox"
            rec_out["clearance_debug"] = {
                "method": "median_groundref_combined_proxy_v2",
                "ref_frame": args.ref_frame,
                "estimated_z_ref_m": z_ref_m,
                "pitch_deg": args.pitch_deg,
                "cam_height_m": args.cam_height,
                "weights": WEIGHTS,
                "relative_depth_proxy": None,
                "z_t_m": None,
                "reference_pick_debug": ref_pick_debug,
            }
            out_lines.append(json.dumps(rec_out))
            continue

        uv = underside_midpoint(rec)
        if uv is None or rel_depth is None:
            rec_out["clearance_m_estimate"] = None
            rec_out["clearance_status"] = "no_relative_depth"
            rec_out["clearance_debug"] = {
                "method": "median_groundref_combined_proxy_v2",
                "ref_frame": args.ref_frame,
                "estimated_z_ref_m": z_ref_m,
                "pitch_deg": args.pitch_deg,
                "cam_height_m": args.cam_height,
                "weights": WEIGHTS,
                "relative_depth_proxy": rel_depth,
                "z_t_m": None,
                "reference_pick_debug": ref_pick_debug,
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
            "method": "median_groundref_combined_proxy_v2",
            "ref_frame": args.ref_frame,
            "estimated_z_ref_m": z_ref_m,
            "reference_ground_pixel": [u_ref, v_ref],
            "reference_ground_pixel_undistorted": [u_ref_ud, v_ref_ud],
            "reference_ground_point_3d": Pg.tolist(),
            "reference_pick_debug": ref_pick_debug,
            "pitch_deg": args.pitch_deg,
            "cam_height_m": args.cam_height,
            "used_smoothing": bool(args.use_smoothing),
            "weights": WEIGHTS,
            "underside_len": underside_length(rec),
            "underside_y_px": underside_y(rec),
            "bbox_h": bbox_h(rec),
            "bbox_w": bbox_w(rec),
            "ref_underside_len": ref_under_len,
            "ref_underside_y_px": ref_under_y_v,
            "ref_bbox_h": ref_bbox_height,
            "ref_bbox_w": ref_bbox_width,
            "proxy_under_y": d_under_y,
            "proxy_under_len": d_under_len,
            "proxy_bbox_h": d_bbox_h,
            "proxy_bbox_w": d_bbox_w,
            "relative_depth_proxy": rel_depth,
            "z_t_m": z_t,
            "underside_pixel": [u, v],
            "underside_pixel_undistorted": [u_ud, v_ud],
            "ray_world": ray_world.tolist(),
            "underside_point_estimated": P_under.tolist(),
        }

        def fmt(v, decimals=3):
            return f"{v:.{decimals}f}" if v is not None else "  N/A"

        print(
            f"{int(rec['frame_idx']):5d} | "
            f"{float(rec.get('time_sec', 0.0)):4.1f} | "
            f"{underside_y(rec):7.1f} | "
            f"{underside_length(rec):6.1f} | "
            f"{fmt(d_under_y):>6} | "
            f"{fmt(d_under_len):>6} | "
            f"{fmt(d_bbox_h):>6} | "
            f"{fmt(d_bbox_w):>5} | "
            f"{rel_depth:6.3f} | "
            f"{z_t:6.2f} | "
            f"{clearance:7.3f} | ok"
        )

        out_lines.append(json.dumps(rec_out))

    Path(args.out_jsonl).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nSaved JSONL to: {args.out_jsonl}")
    print(f"Saved reference overlay to: {args.overlay_out}")


if __name__ == "__main__":
    main()

#     python scripts/clearance_from_bridge_scale_with_ref_groundpoint.py \
#   --video video.MOV \
#   --jsonl outputs/extract_underside_lines.jsonl \
#   --out_jsonl outputs/clearance_output_V1_baseline.jsonl \
#   --intrinsics configs/intrinsics.yaml \
#   --cam_height 1.05 \
#   --pitch_deg 2.0 \
#   --ref_frame 180 \
#   --frame_min 0 \
#   --frame_max 330 \
#   --use_smoothing \
#   --overlay_out outputs/reference_overlay_V1_baseline.jpg