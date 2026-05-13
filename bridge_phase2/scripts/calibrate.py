import glob
from pathlib import Path

import cv2
import numpy as np
import yaml


def find_corners(gray, pattern_size):
    """
    Tries the more robust OpenCV SB corner detector first if available,
    otherwise falls back to the classic method.
    Returns (ok, corners) where corners shape is (N,1,2).
    """
    # pattern_size is (cols, rows)
    if hasattr(cv2, "findChessboardCornersSB"):
        # SB is typically more robust to blur/lighting than the classic method.
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size)
        return ok, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    return ok, corners


def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--images", default="data/calib_images/*",
                    help="glob pattern for calibration images (jpg/png recommended)")
    ap.add_argument("--rows", type=int, required=True,
                    help="INNER corners rows (squares-1). Example: 6x6 squares -> rows=5")
    ap.add_argument("--cols", type=int, required=True,
                    help="INNER corners cols (squares-1). Example: 6x6 squares -> cols=5")
    ap.add_argument("--square_m", type=float, required=True,
                    help="square size in meters. Example: 2.7cm -> 0.027")
    ap.add_argument("--out", default="configs/intrinsics.yaml",
                    help="output yaml path")
    ap.add_argument("--max_images", type=int, default=0,
                    help="0 = use all images, else use only first N images (for quick tests)")
    ap.add_argument("--show", action="store_true",
                    help="show detected corners while running (press any key to advance, ESC to quit)")

    args = ap.parse_args()

    imgs = sorted(glob.glob(args.images))
    if args.max_images and args.max_images > 0:
        imgs = imgs[:args.max_images]

    if not imgs:
        raise SystemExit(f"No images found with pattern: {args.images}")

    pattern_size = (args.cols, args.rows)  # OpenCV expects (cols, rows)

    # Prepare known 3D points for the checkerboard corners in its own coordinate system (z=0 plane).
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_m  # scale grid spacing to meters

    objpoints = []  # list of (N,3) points in world coords (checkerboard plane)
    imgpoints = []  # list of (N,1,2) points in image pixel coords

    used = 0
    failed = 0
    last_gray = None

    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            failed += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        last_gray = gray

        ok, corners = find_corners(gray, pattern_size)
        if not ok or corners is None:
            failed += 1
            continue

        # Improve corner precision (sub-pixel refinement).
        # For SB detector, corners are often already good, but this doesn't hurt.
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        )

        objpoints.append(objp)
        imgpoints.append(corners)
        used += 1

        if args.show:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, ok)
            cv2.imshow("corners", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break

    if args.show:
        cv2.destroyAllWindows()

    if used < 8:
        raise SystemExit(
            f"Too few usable images: {used}. Failed/ignored: {failed}. "
            f"Aim for 10–20 good images. Also verify rows/cols are INNER corners."
        )

    if last_gray is None:
        raise SystemExit("Could not read any images successfully.")

    h, w = last_gray.shape[:2]

    # Calibrate camera. rms is the RMS reprojection error (lower is better).
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    Path("configs").mkdir(parents=True, exist_ok=True)

    data = {
        "image_width": int(w),
        "image_height": int(h),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "rms_reprojection_error": float(rms),
        "pattern": {"rows": args.rows, "cols": args.cols, "square_m": args.square_m},
        "used_images": int(used),
        "total_images_considered": int(len(imgs)),
        "failed_or_skipped": int(failed),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print("✅ Saved intrinsics:", args.out)
    print(f"Used images: {used}/{len(imgs)}  (failed/skipped: {failed})")
    print(f"RMS reprojection error: {rms:.4f} (lower is better; ~<1 is usually OK)")


if __name__ == "__main__":
    main()