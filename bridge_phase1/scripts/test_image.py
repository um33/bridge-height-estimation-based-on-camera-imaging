
import os, json
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

def main():
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    model_id = os.getenv("ROBOFLOW_MODEL_ID", "").strip()
    if not api_key or not model_id:
        raise SystemExit("Missing ROBOFLOW_API_KEY or ROBOFLOW_MODEL_ID in .env")

    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )

    # Pass a PATH (string) instead of bytes
    image_path = "image.jpg"  
    result = client.infer(image_path, model_id=model_id)

    # save the result in a jsonl file with in output folder
    with open("output/result.jsonl", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()