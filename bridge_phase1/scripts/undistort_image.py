
import cv2
import yaml
import numpy as np
from pathlib import Path

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--intrinsics", default="configs/intrinsics.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs/undistorted.jpg")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="0=crop to remove black borders, 1=keep all pixels (more black borders)")
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.intrinsics).read_text(encoding="utf-8"))
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    h, w = img.shape[:2]

    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), args.alpha, (w, h))
    und = cv2.undistort(img, K, dist, None, newK)

    # Crop to ROI to remove black borders (especially when alpha=0)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        und = und[y:y+rh, x:x+rw]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, und)
    print("✅ Saved:", args.out)

if __name__ == "__main__":
    main()