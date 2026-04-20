import json
from pathlib import Path


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


def bbox_h(rec):
    b = rec["bridge_bbox_used"]
    return float(b["y2"]) - float(b["y1"])


def underside_len(rec):
    u = rec["underside_line"]
    dx = float(u["x2"]) - float(u["x1"])
    dy = float(u["y2"]) - float(u["y1"])
    return (dx * dx + dy * dy) ** 0.5


def moving_average(values, window=3):
    out = []
    n = len(values)
    for i in range(n):
        left = max(0, i - window + 1)
        chunk = [v for v in values[left:i + 1] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--frame_min", type=int, default=None)
    ap.add_argument("--frame_max", type=int, default=None)
    ap.add_argument("--ref_frame", type=int, required=True)
    ap.add_argument("--out_csv", default="outputs/relative_depth_proxy.csv")
    args = ap.parse_args()

    records = load_records(args.jsonl)

    if args.frame_min is not None:
        records = [r for r in records if int(r["frame_idx"]) >= args.frame_min]
    if args.frame_max is not None:
        records = [r for r in records if int(r["frame_idx"]) <= args.frame_max]

    valid = [r for r in records if is_valid_record(r)]
    if not valid:
        raise RuntimeError("No valid records found.")

    ref = None
    for r in valid:
        if int(r["frame_idx"]) == args.ref_frame:
            ref = r
            break
    if ref is None:
        raise RuntimeError(f"Reference frame {args.ref_frame} not found among valid records.")

    ref_bbox_h = bbox_h(ref)
    ref_under_len = underside_len(ref)

    rows = []
    for r in valid:
        fh = bbox_h(r)
        fl = underside_len(r)

        scale_bbox_h = fh / ref_bbox_h if ref_bbox_h > 1e-9 else None
        scale_under = fl / ref_under_len if ref_under_len > 1e-9 else None

        depth_proxy_bbox_h = (1.0 / scale_bbox_h) if scale_bbox_h and scale_bbox_h > 1e-9 else None
        depth_proxy_under = (1.0 / scale_under) if scale_under and scale_under > 1e-9 else None

        rows.append({
            "frame_idx": int(r["frame_idx"]),
            "time_sec": float(r.get("time_sec", 0.0)),
            "bbox_h": fh,
            "underside_len": fl,
            "scale_bbox_h_vs_ref": scale_bbox_h,
            "scale_underside_len_vs_ref": scale_under,
            "depth_proxy_bbox_h": depth_proxy_bbox_h,
            "depth_proxy_underside_len": depth_proxy_under,
        })

    sm_bbox = moving_average([row["depth_proxy_bbox_h"] for row in rows], window=3)
    sm_under = moving_average([row["depth_proxy_underside_len"] for row in rows], window=3)

    for row, s1, s2 in zip(rows, sm_bbox, sm_under):
        row["depth_proxy_bbox_h_smooth3"] = s1
        row["depth_proxy_underside_len_smooth3"] = s2

    print("\nRELATIVE DEPTH PROXY\n")
    print(
        "frame | time | bbox_h | under_len | scale_h | scale_len | depth_h | depth_len | depth_h_sm | depth_len_sm"
    )
    print("-" * 120)

    for row in rows:
        print(
            f"{row['frame_idx']:5d} | "
            f"{row['time_sec']:.1f} | "
            f"{row['bbox_h']:.2f} | "
            f"{row['underside_len']:.2f} | "
            f"{row['scale_bbox_h_vs_ref']:.3f} | "
            f"{row['scale_underside_len_vs_ref']:.3f} | "
            f"{row['depth_proxy_bbox_h']:.3f} | "
            f"{row['depth_proxy_underside_len']:.3f} | "
            f"{row['depth_proxy_bbox_h_smooth3']:.3f} | "
            f"{row['depth_proxy_underside_len_smooth3']:.3f}"
        )

    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        vals = []
        for h in headers:
            v = row[h]
            vals.append("" if v is None else str(v))
        lines.append(",".join(vals))

    Path(args.out_csv).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()