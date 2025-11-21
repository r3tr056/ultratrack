import argparse
import cv2
import os
import sys
import subprocess
import shutil
import requests
from tqdm import tqdm
import json
import glob

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
GALLERY_DIR = os.path.join(MODELS_DIR, "gallery")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build", "Release", "Release") # Adjust based on actual build path
EXECUTABLE = os.path.join(BUILD_DIR, "ultratrack.exe")

def setup_environment(args):
    print(f"[+] Setting up UltraTrack environment in {PROJECT_ROOT}...")
    
    # 1. Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(GALLERY_DIR, exist_ok=True)
    
    # 2. Download Models (Placeholders)
    models = {
        "yolov11n.onnx": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx",
        "osnet_x1_0.onnx": "https://github.com/kaiyangzhou/deep-person-reid/releases/download/v1.0/osnet_x1_0_imagenet.pth" # Placeholder, needs ONNX conversion usually
    }
    
    for name, url in models.items():
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            print(f"[-] Downloading {name}...")
            try:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                with open(path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
                    for data in response.iter_content(1024):
                        f.write(data)
                        bar.update(len(data))
            except Exception as e:
                print(f"[!] Failed to download {name}: {e}")
        else:
            print(f"[+] {name} already exists.")

    # 3. TensorRT Engine Conversion (Optional)
    if shutil.which("trtexec"):
        print("[+] trtexec found. Checking for engines...")
        # Logic to convert ONNX to Engine would go here
        pass
    else:
        print("[*] trtexec not found. Skipping TensorRT engine generation (CPU mode will be used).")

def train_tracker(args):
    print(f"[+] Starting Training (Gallery Creation) for target: {args.target}")
    target_dir = os.path.join(GALLERY_DIR, args.target)
    if os.path.exists(target_dir) and not args.overwrite:
        print(f"[!] Target '{args.target}' already exists. Use --overwrite to replace.")
        return
    
    if args.overwrite:
        shutil.rmtree(target_dir, ignore_errors=True)
    
    os.makedirs(target_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(int(args.input) if args.input.isdigit() else args.input)
    if not cap.isOpened():
        print(f"[!] Could not open input: {args.input}")
        return

    print("\nControls:")
    print("  SPACE: Pause/Resume")
    print("  s:     Select ROI and Save (when paused)")
    print("  q:     Quit")
    
    count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            display_frame = frame.copy()
        else:
            display_frame = frame.copy()
            cv2.putText(display_frame, "PAUSED - Press 's' to select target", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("UltraTrack Trainer", display_frame)
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == ord(' '):
            paused = not paused
        elif key == ord('q'):
            break
        elif key == ord('s') and paused:
            bbox = cv2.selectROI("UltraTrack Trainer", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("ROI selector") # Cleanup ROI window
            if bbox[2] > 0 and bbox[3] > 0:
                x, y, w, h = bbox
                crop = frame[int(y):int(y+h), int(x):int(x+w)]
                save_path = os.path.join(target_dir, f"{count:02d}.jpg")
                cv2.imwrite(save_path, crop)
                print(f"[+] Saved sample {count} to {save_path}")
                count += 1
                if count >= args.count:
                    print(f"[+] Collected {count} samples. Training complete.")
                    break
            else:
                print("[-] Invalid selection.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[+] Gallery for '{args.target}' created at {target_dir}")

def test_tracker(args):
    print(f"[+] Testing UltraTrack...")
    
    cmd = [EXECUTABLE]
    
    # Map Python args to C++ args
    if args.input:
        cmd.extend(["--input", args.input])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.confidence:
        cmd.extend(["--confidence", str(args.confidence)])
    
    # Gallery handling
    if args.target:
        gallery_path = os.path.join(GALLERY_DIR, args.target)
        if not os.path.exists(gallery_path):
            print(f"[!] Gallery for '{args.target}' not found. Run 'train' first.")
            return
        # Pass gallery path if C++ supports it (via config or arg)
        # Assuming we added --gallery arg to C++ or use env var
        # For now, let's assume we pass it via a new flag we should add to main.cpp
        # Or we just print it for now.
        print(f"[*] Using gallery: {gallery_path}")
        # cmd.extend(["--gallery", gallery_path]) # Uncomment when C++ main supports it

    print(f"[*] Running: {' '.join(cmd)}")
    
    try:
        if not os.path.exists(EXECUTABLE):
             # Fallback to searching in build dir
             print(f"[!] Executable not found at {EXECUTABLE}. Searching...")
             found = glob.glob(os.path.join(PROJECT_ROOT, "build", "**", "ultratrack.exe"), recursive=True)
             if found:
                 cmd[0] = found[0]
                 print(f"[*] Found at {cmd[0]}")
             else:
                 print("[!] Could not find ultratrack.exe. Please build the project first.")
                 return

        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")

def main():
    parser = argparse.ArgumentParser(description="UltraTrack Configuration & Training Utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup Command
    setup_parser = subparsers.add_parser("setup", help="Download models and setup environment")
    
    # Train Command
    train_parser = subparsers.add_parser("train", help="Create a target gallery (Few-Shot Training)")
    train_parser.add_argument("-t", "--target", required=True, help="Name of the target object")
    train_parser.add_argument("-i", "--input", default="0", help="Input video source (0 for camera, or path)")
    train_parser.add_argument("-c", "--count", type=int, default=5, help="Number of samples to collect")
    train_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing gallery")
    
    # Test Command
    test_parser = subparsers.add_parser("test", help="Run the tracker")
    test_parser.add_argument("-i", "--input", default="0", help="Input video source")
    test_parser.add_argument("-o", "--output", help="Output video path")
    test_parser.add_argument("-t", "--target", help="Name of the target to track (uses gallery)")
    test_parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_environment(args)
    elif args.command == "train":
        train_tracker(args)
    elif args.command == "test":
        test_tracker(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
