from detector import YoloDetector
from tracker import Tracker
import cv2
import time

confidence = 0.8
iou_threshold = 0.3
model_path = "best.pt"
video_path = "15sec_input_720p.mp4"

detector = YoloDetector(model_path=model_path, confidence=confidence)
tracker = Tracker()

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.perf_counter()

    detections = detector.detect(frame)
    tracks = tracker.track(detections, frame, iou_threshold, confidence)

    # Draw all tracks
    for track in tracks:
      x1, y1, x2, y2 = track["bbox"]
      track_id = track["track_id"]

      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    end_time = time.perf_counter()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    print(f"FPS: {fps:.2f}")
    print("---------------------------------")

    cv2.imshow("Detection + Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
      break

cap.release()
cv2.destroyAllWindows()
