import json
import math
from pathlib import Path
from statistics import mean, median, pstdev


def load_records(jsonl_path):
    records = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return sorted(records, key=lambda r: int(r.get("frame_idx", 0)))


def is_valid_clearance_record(rec):
    clearance = rec.get("clearance_m_estimate")
    status = rec.get("clearance_status")
    return clearance is not None and status == "ok"


def to_float(x):
    if x is None:
        return None
    return float(x)


def summarize(values):
    if not values:
        return None

    vals = [float(v) for v in values]
    return {
        "count": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "min": min(vals),
        "max": max(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "range": max(vals) - min(vals),
    }


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
            vals.append("" if v is None else str(v))
        lines.append(",".join(vals))

    Path(out_csv).write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(x, nd=3):
    if x is None:
        return "None"
    return f"{x:.{nd}f}"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Input clearance JSONL")
    ap.add_argument("--frame_min", type=int, default=None)
    ap.add_argument("--frame_max", type=int, default=None)
    ap.add_argument("--frames", default=None,
                    help="Optional comma-separated exact frames, e.g. 180,210,240")
    ap.add_argument("--out_csv", default=None,
                    help="Optional CSV export of selected valid records")
    args = ap.parse_args()

    records = load_records(args.jsonl)

    if args.frame_min is not None:
        records = [r for r in records if int(r.get("frame_idx", 0)) >= args.frame_min]
    if args.frame_max is not None:
        records = [r for r in records if int(r.get("frame_idx", 0)) <= args.frame_max]

    if args.frames:
        chosen = {int(x.strip()) for x in args.frames.split(",") if x.strip()}
        records = [r for r in records if int(r.get("frame_idx", 0)) in chosen]

    total_records = len(records)
    valid_records = [r for r in records if is_valid_clearance_record(r)]

    rows = []
    for rec in valid_records:
        rows.append({
            "frame_idx": int(rec.get("frame_idx", 0)),
            "time_sec": float(rec.get("time_sec", 0.0)),
            "clearance_m_estimate": float(rec["clearance_m_estimate"]),
            "clearance_status": rec.get("clearance_status"),
            "underside_status": rec.get("underside_status"),
        })

    clearances = [row["clearance_m_estimate"] for row in rows]
    stats = summarize(clearances)

    print("\nCLEARANCE EVALUATION\n")
    print(f"Selected records: {total_records}")
    print(f"Valid clearance records: {len(valid_records)}")

    if not rows:
        print("No valid clearance records found in the selected range.")
        return

    print("\nValid records:")
    print("frame | time | clearance")
    print("-" * 32)
    for row in rows:
        print(
            f"{row['frame_idx']:5d} | "
            f"{row['time_sec']:.1f} | "
            f"{row['clearance_m_estimate']:.3f}"
        )

    print("\nSummary:")
    print(f"count   : {stats['count']}")
    print(f"mean    : {fmt(stats['mean'])}")
    print(f"median  : {fmt(stats['median'])}")
    print(f"min     : {fmt(stats['min'])}")
    print(f"max     : {fmt(stats['max'])}")
    print(f"std     : {fmt(stats['std'])}")
    print(f"range   : {fmt(stats['range'])}")

    if args.out_csv:
        write_csv(rows, args.out_csv)
        print(f"\nSaved CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()