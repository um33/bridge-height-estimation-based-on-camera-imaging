import os  # for reading environment variables (API key / model id)
import json  # for writing JSON lines (one JSON object per line)
from pathlib import Path  # for safe path handling across OSes
import cv2  # OpenCV: read video frames and save a temp image
from dotenv import load_dotenv  # loads variables from a .env file into the environment
from inference_sdk import InferenceHTTPClient  # Roboflow hosted inference client
from tqdm import tqdm  # progress bar while processing video


def ensure_dir(p: Path) -> None:  # helper to make sure output directory exists
    p.mkdir(parents=True, exist_ok=True)  # create folder(s) if missing; don't error if already exists


def main() -> None:  # main entrypoint of the script
    load_dotenv()  # read .env and put values into os.environ

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()  # fetch Roboflow API key from environment
    model_id = os.getenv("ROBOFLOW_MODEL_ID", "").strip()  # fetch Roboflow model id from environment

    if not api_key or not model_id:  # validate we have both values
        raise SystemExit("Missing ROBOFLOW_API_KEY or ROBOFLOW_MODEL_ID in .env")  # stop with clear message

    client = InferenceHTTPClient(  # create a client object to call Roboflow hosted inference
        api_url="https://detect.roboflow.com",  # Roboflow hosted detection endpoint base URL
        api_key=api_key  # your private API key (authorizes requests)
    )

    import argparse  # standard library for CLI flags/arguments

    ap = argparse.ArgumentParser()  # create a CLI argument parser
    ap.add_argument("--video", required=True, help="Path to input video")  # input video path
    ap.add_argument("--out", default="outputs", help="Output folder")  # folder where JSONL will be written
    ap.add_argument("--every", type=float, default=1.0, help="Sample one frame every N seconds")  # sampling interval
    ap.add_argument("--max_samples", type=int, default=0, help="0=all, else stop after N sampled frames")  # optional limit
    args = ap.parse_args()  # parse CLI args into an object

    video_path = Path(args.video)  # convert input string to a Path
    out_dir = Path(args.out)  # convert output folder string to a Path
    ensure_dir(out_dir)  # create the output folder if it doesn't exist

    cap = cv2.VideoCapture(str(video_path))  # open the video file for reading
    if not cap.isOpened():  # check video opened correctly
        raise SystemExit(f"Cannot open video: {video_path}")  # stop if video path is wrong/corrupt

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # read video FPS; fallback to 30 if missing
    step_frames = max(1, int(round(args.every * fps)))  # frames between samples: every_seconds * fps

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1  # try get total frames (may be -1 sometimes)
    total = n_frames if n_frames > 0 else None  # tqdm needs None if unknown total length

    out_jsonl = out_dir / f"{video_path.stem}_detections_every_{args.every:g}s.jsonl"  # JSONL output file name

    tmp_path = out_dir / "_tmp_frame.jpg"  # temporary file used because SDK accepts file path input well

    frame_idx = 0  # current frame index in the video
    samples_written = 0  # number of sampled frames we actually inferred & saved

    pbar = tqdm(total=total, desc=f"Sampling every {args.every}s (~{step_frames} frames)")  # progress bar

    with out_jsonl.open("w", encoding="utf-8") as f:  # open JSONL for writing (overwrites if exists)
        while True:  # loop until video ends
            ok, frame = cap.read()  # read next frame from the video
            if not ok:  # if no frame, we reached end (or read error)
                break  # exit loop

            pbar.update(1)  # update progress bar by 1 frame read

            if frame_idx % step_frames != 0:  # if this frame is NOT one of our sampled times
                frame_idx += 1  # advance frame index
                continue  # skip inference for this frame

            # Save sampled frame to a temp jpeg on disk (Roboflow SDK call uses path input reliably)
            cv2.imwrite(str(tmp_path), frame)  # write current sampled frame as a jpeg file

            # Run Roboflow inference on that saved image
            result = client.infer(str(tmp_path), model_id=model_id)  # send image to model and get JSON back
            preds = result.get("predictions", [])  # extract predictions list, default empty if none

            # Create one JSON record for this sampled time
            record = {  # build a dict matching the JSONL schema you want
                "frame_idx": frame_idx,  # the absolute frame index in the original video
                "time_sec": frame_idx / fps,  # time in seconds from video start (based on fps)
                "predictions": preds  # the model's predictions (boxes + class + confidence)
            }

            f.write(json.dumps(record) + "\n")  # write one JSON object per line to the JSONL file

            samples_written += 1  # increment number of sampled frames processed
            if args.max_samples and samples_written >= args.max_samples:  # if user requested a sampling limit
                break  # stop early after enough samples

            frame_idx += 1  # advance frame index for next loop iteration

        # Important: if we broke due to max_samples, frame_idx may not have been incremented for the last loop
        # This is fine since we are stopping anyway.

    pbar.close()  # close the progress bar cleanly
    cap.release()  # release the video file handle

    # Print summary
    print("\n✅ Done")  # friendly completion marker
    print(f"Video: {video_path}")  # show which video was processed
    print(f"FPS: {fps:.2f}")  # show FPS used for time conversion
    print(f"Sampling: every {args.every}s (~{step_frames} frames)")  # show sampling interval
    print(f"Samples written: {samples_written}")  # how many JSONL lines were produced
    print(f"JSONL output: {out_jsonl}")  # where the JSONL file is saved


if __name__ == "__main__":  # standard Python entrypoint guard
    main()  # call main() when run as a script



# python scripts/test_video.py --video video.MOV --every 1.0