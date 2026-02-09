"""
SimUAV - Video Detection
Runs YOLOv8 detection on recorded video.
Air-to-Air UAV Detection & Tracking System
"""
import cv2
import time
from ultralytics import YOLO

# Configuration
MODEL_PATH = "best.pt"
INPUT_VIDEO = "flight_recording.mp4"
OUTPUT_VIDEO = "detection_result.mp4"
CONFIDENCE_THRESHOLD = 0.5


def main():
    print("=" * 60)
    print("SimUAV - Video Detection")
    print("=" * 60)
    
    # Load YOLO model
    print(f"Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Open input video
    print(f"Opening video: {INPUT_VIDEO}")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        print(f"Error: Cannot open {INPUT_VIDEO}")
        return
    
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {w}x{h} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))
    
    print(f"Output: {OUTPUT_VIDEO}")
    print("Processing... Press Q to stop")
    print("=" * 60)
    
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            
            # Draw detections
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"UAV: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw center
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    
                    detection_count += 1
            
            # Draw crosshair
            cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 0), 1)
            cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 0), 1)
            
            # Draw info
            cv2.rectangle(frame, (10, 10), (200, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Detections: {detection_count}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame
            out.write(frame)
            
            # Show progress
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            progress = frame_count / total_frames * 100
            print(f"\rProgress: {progress:.1f}% | FPS: {fps_actual:.1f} | Detections: {detection_count}", end="")
            
            # Display
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\n\nProcessing complete!")
        print(f"Processed {frame_count} frames in {elapsed:.1f}s")
        print(f"Average FPS: {frame_count/elapsed:.1f}")
        print(f"Total detections: {detection_count}")
        print(f"Output saved: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
