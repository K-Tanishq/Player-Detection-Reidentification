from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        self.object_tracker = DeepSort(
            max_age=10,
            n_init=2,
            nms_max_overlap=0.3,
            max_cosine_distance=0.8,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True
        )

    def track(self, detections, frame, iou_threshold, confidence):

        formatted_detections = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_name = det["class_name"]
            
            w = x2 - x1
            h = y2 - y1
            formatted_detections.append([[x1, y1, w, h], conf, cls_name])

        tracks = self.object_tracker.update_tracks(formatted_detections, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            track_id = track.track_id
            
            bbox = list(map(int, ltrb))
            if any(self._iou(bbox, det["bbox"]) > iou_threshold and det["confidence"] >= confidence 
                  for det in detections):
                results.append({
                    "track_id": track_id,
                    "bbox": bbox
                })

        print(f"Number of tracks after filtering: {len(results)}")
        return results

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
