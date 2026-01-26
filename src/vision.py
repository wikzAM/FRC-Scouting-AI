from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        print(f"Loading YOLO model: {model_path}...")
        
        # TODO: Initialize the YOLO model using the model_path
        # Hint: self.model = YOLO(...)
        #self.model = YOLO(model_path)
        self.model = YOLO("runs/detect/train2/weights/best.pt")
        print("Model Classes:", self.model.names)
        if self.model:
            print("Model loaded successfully.")
        else:
            print("Error: Model failed to load.")
        # In the COCO dataset (what YOLO is trained on), Class 0 is 'person'.
        # We will track people for now as a proxy for robots.
        self.target_class_id = 0

    def detect(self, frame):
        """
        Input: An image (frame)
        Output: A list of lists: [[x1, y1, x2, y2, confidence], ...]
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 1. Run the model
        # verbose=False stops it from printing "Detected 1 person" to the console every frame
        results = self.model.track(rgb_frame, persist=True, tracker="botsort.yaml", conf = .35, verbose=False)[0] 
        
        detections = []
        
        # 2. Iterate through the results
        for box in results.boxes:

            # Check if we have an ID yet (sometimes new objects don't have one immediately)
            if box.id is not None:
                track_id = int(box.id.item()) # The Unique ID (e.g., Robot #1, Robot #2)
            else:
                track_id = -1 # Temporary ID for brand new detections
                
            # box.cls is a list of floats (e.g., [0.0] for person). We want an int.
            class_id = int(box.cls[0])
            
            # box.conf is the confidence (0.0 to 1.0)
            conf = float(box.conf[0])

            
            

        
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # This will ensure that things like heads or really small detections are ignored.
            area = (int(y2) - int(y1)) * (int(x2) - int(x1))
            if area < 500:
                continue

            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue

            h = int(y2) - int(y1)
            bumper_h = int(h * 0.35)  # 35% of height
            bumper_y1 = int(y2) - bumper_h
            bumper_y1 = max(bumper_y1, int(y1))  # Ensure it doesn't go above y1

            robot_crop = frame[int(bumper_y1):int(y2), int(x1):int(x2)]
            
            if robot_crop.size > 0:
                hsv_crop = cv2.cvtColor(robot_crop, cv2.COLOR_BGR2HSV)
                # We define what is "blue" in HSV space
                min_blue = np.array([100, 50, 50])
                max_blue = np.array([130, 255, 255])
                blue_mask = cv2.inRange(hsv_crop, min_blue, max_blue)
                blue_pixels = cv2.countNonZero(blue_mask)
                # Similarly for "red"
                min_red1 = np.array([0, 50, 50])
                max_red1 = np.array([10, 255, 255])
                min_red2 = np.array([170, 50, 50])
                max_red2 = np.array([180, 255, 255])
                red_mask1 = cv2.inRange(hsv_crop, min_red1, max_red1)
                red_mask2 = cv2.inRange(hsv_crop, min_red2, max_red2)
                red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
                # We compare if there are more blue or red pixels
                if blue_pixels > red_pixels:
                    class_id = 0 # Blue
                elif red_pixels > blue_pixels:
                    class_id = 1 # Red
                else:
                    class_id = int(box.cls[0]) # Fallback
            else:
                class_id = int(box.cls[0]) # Fallback

            detections.append([int(x1), int(y1), int(x2), int(y2), conf, class_id, track_id])

                
        return detections
    
    # Simple test code
if __name__ == "__main__":
    # 0 opens your default webcam. 
    # If you don't have one, put the path to a video file string here.
    cap = cv2.VideoCapture(0) 
    
    # Initialize your class
    detector = ObjectDetector()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # TODO: Call your detect method!
        # boxes = ...
        boxes = detector.detect(frame)
        
        # Loop through your boxes and draw them
        # for (x1, y1, x2, y2, conf) in boxes:
        for (x1, y1, x2, y2, conf) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Optionally, draw the confidence
            cv2.putText(frame, f"{conf:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Vision Test", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()