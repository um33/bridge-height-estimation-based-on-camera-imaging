import json
import math
from pathlib import Path


def load_records(jsonl_path):
    records = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def bbox_metrics(bbox):
    if bbox is None:
        return None

    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    x2 = float(bbox["x2"])
    y2 = float(bbox["y2"])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h

    return {
        "bbox_w": w,
        "bbox_h": h,
        "bbox_area": area,
        "bbox_sqrt_area": math.sqrt(area) if area > 0 else 0.0,
    }


def underside_metrics(underside_line):
    if underside_line is None:
        return None

    x1 = float(underside_line["x1"])
    y1 = float(underside_line["y1"])
    x2 = float(underside_line["x2"])
    y2 = float(underside_line["y2"])

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    midpoint_x = 0.5 * (x1 + x2)
    midpoint_y = 0.5 * (y1 + y2)
    slope = dy / dx if abs(dx) > 1e-9 else None

    return {
        "underside_len": length,
        "underside_mid_x": midpoint_x,
        "underside_mid_y": midpoint_y,
        "underside_slope": slope,
    }


def safe_ratio(curr, prev):
    if curr is None or prev is None or abs(prev) < 1e-9:
        return None
    return curr / prev


def safe_delta(curr, prev):
    if curr is None or prev is None:
        return None
    return curr - prev


def is_valid_record(rec):
    return (
        rec.get("underside_status") == "ok"
        and rec.get("bridge_bbox_used") is not None
        and rec.get("underside_line") is not None
    )


def format_num(x, ndigits=2):
    if x is None:
        return "None"
    return f"{x:.{ndigits}f}"


def write_csv(rows, out_csv):
    if not rows:
        Path(out_csv).write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines = [",".join(headers)]

    for row in rows:
        vals = []
        for h in headers:
            v = row[h]
            if v is None:
                vals.append("")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))

    Path(out_csv).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_csv", default="outputs/bridge_multiframe_features.csv")
    ap.add_argument("--frame_min", type=int, default=None)
    ap.add_argument("--frame_max", type=int, default=None)
    args = ap.parse_args()

    records = load_records(args.jsonl)
    records = sorted(records, key=lambda r: int(r.get("frame_idx", 0)))

    if args.frame_min is not None:
        records = [r for r in records if int(r.get("frame_idx", 0)) >= args.frame_min]
    if args.frame_max is not None:
        records = [r for r in records if int(r.get("frame_idx", 0)) <= args.frame_max]

    valid_rows = []
    prev_row = None

    print("\nVALID FRAMES ANALYSIS\n")
    print(
        "frame | time | bbox_w | bbox_h | bbox_area | underside_len | underside_y | "
        "w_ratio | h_ratio | area_ratio | len_ratio | dy_underside"
    )
    print("-" * 120)

    for rec in records:
        if not is_valid_record(rec):
            continue

        frame_idx = int(rec["frame_idx"])
        time_sec = float(rec.get("time_sec", 0.0))

        bbox_m = bbox_metrics(rec["bridge_bbox_used"])
        under_m = underside_metrics(rec["underside_line"])

        row = {
            "frame_idx": frame_idx,
            "time_sec": time_sec,
            "bbox_w": bbox_m["bbox_w"],
            "bbox_h": bbox_m["bbox_h"],
            "bbox_area": bbox_m["bbox_area"],
            "bbox_sqrt_area": bbox_m["bbox_sqrt_area"],
            "underside_len": under_m["underside_len"],
            "underside_mid_x": under_m["underside_mid_x"],
            "underside_mid_y": under_m["underside_mid_y"],
            "underside_slope": under_m["underside_slope"],
            "bbox_w_ratio": safe_ratio(bbox_m["bbox_w"], prev_row["bbox_w"]) if prev_row else None,
            "bbox_h_ratio": safe_ratio(bbox_m["bbox_h"], prev_row["bbox_h"]) if prev_row else None,
            "bbox_area_ratio": safe_ratio(bbox_m["bbox_area"], prev_row["bbox_area"]) if prev_row else None,
            "bbox_sqrt_area_ratio": safe_ratio(bbox_m["bbox_sqrt_area"], prev_row["bbox_sqrt_area"]) if prev_row else None,
            "underside_len_ratio": safe_ratio(under_m["underside_len"], prev_row["underside_len"]) if prev_row else None,
            "underside_mid_y_delta": safe_delta(under_m["underside_mid_y"], prev_row["underside_mid_y"]) if prev_row else None,
            "underside_mid_x_delta": safe_delta(under_m["underside_mid_x"], prev_row["underside_mid_x"]) if prev_row else None,
        }

        print(
            f"{frame_idx:5d} | "
            f"{format_num(time_sec, 1):>4} | "
            f"{format_num(row['bbox_w']):>7} | "
            f"{format_num(row['bbox_h']):>7} | "
            f"{format_num(row['bbox_area']):>9} | "
            f"{format_num(row['underside_len']):>13} | "
            f"{format_num(row['underside_mid_y']):>10} | "
            f"{format_num(row['bbox_w_ratio'], 3):>7} | "
            f"{format_num(row['bbox_h_ratio'], 3):>7} | "
            f"{format_num(row['bbox_area_ratio'], 3):>10} | "
            f"{format_num(row['underside_len_ratio'], 3):>9} | "
            f"{format_num(row['underside_mid_y_delta'], 2):>11}"
        )

        valid_rows.append(row)
        prev_row = row

    write_csv(valid_rows, args.out_csv)
    print(f"\nSaved CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()