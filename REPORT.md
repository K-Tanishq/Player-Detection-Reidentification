### Code:
[https://github.com/K-Tanishq/Player-Detection-Reidentification](https://github.com/K-Tanishq/Player-Detection-Reidentification)

# Objective

The goal of this project is to detect and track players in a 15-second football match video using object detection and re-identification techniques, ensuring consistent ID assignment to players throughout the video. This report details the development of a real-time player detection and re-identification system using computer vision and deep learning techniques.

# Approach and Methodology

## Object Detection

- **Model Used:** YOLOv11 (custom-trained model: `best.pt`)
- **Framework:** Ultralytics YOLO library
- **Classes of Interest:** person, player, referee

## Filtering & Preprocessing

- Only detections with class names matching the above and confidence scores above a threshold were retained.
- Bounding boxes were scaled to original frame dimensions to avoid resolution mismatches.

## Tracking and Re-Identification

- **Tracker Used:** DeepSort with MobileNet-based embedding extractor.
- DeepSort utilized the properties of Kalman Filter to store IDs and re-identify them.

# Techniques Explored

## Pretrained Embedding with Similarity Matching

- Using a pre-trained backbone model like ResNet18, ResNet50, VGG16, etc., to extract embeddings for detected player crops.
- Compared current frame embeddings with all stored ones using dot product or cosine similarity.
- Assigned and displayed ID, and updated the embedding database accordingly.

# Result

The final pipeline accurately detects and re-identifies players across the video timeline, ensuring consistent IDs with good bounding box quality and real-time performance.
![Players Detection and Re-Identification](https://github.com/K-Tanishq/Player-Detection-Reidentification/blob/98ae26d98b9ebfbedf229f206e1120b47e3725e7/output.gif)

# Challenges Faced

1. **Crowd Occlusion:** Towards the end of the video, crowding caused blank or false detections.
2. **Bounding Box Scaling Issues:** Discrepancies between detector’s and tracker’s image scales. Bounding boxes were coming too large that they couldn’t even be considered valid. This issue occurred because `[x1, y1, w, h]` was being used for bounding boxes, but DeepSort expects `[x1, y1, x2, y2]`.
3. **Confidence Mismatch:** Even after applying a confidence threshold, the tracker did not reflect the expected filtered detections. This was confirmed by testing the detector independently. The issue was resolved by applying an additional filter using Intersection Over Union (IOU) thresholds.

# Tools & Libraries

- Python
- OpenCV
- Ultralytics YOLOv11
- DeepSort Realtime
