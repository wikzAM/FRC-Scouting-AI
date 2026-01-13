from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        print(f"Loading YOLO model: {model_path}...")
        
        # TODO: Initialize the YOLO model using the model_path
        # Hint: self.model = YOLO(...)
        self.model = YOLO(model_path)
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
        # 1. Run the model
        # verbose=False stops it from printing "Detected 1 person" to the console every frame
        results = self.model(frame, verbose=False)[0] 
        
        detections = []
        
        # 2. Iterate through the results
        for box in results.boxes:
            # box.cls is a list of floats (e.g., [0.0] for person). We want an int.
            class_id = int(box.cls[0])
            
            # box.conf is the confidence (0.0 to 1.0)
            conf = float(box.conf[0])
            
            # TODO: Write an IF statement.
            # We only want to keep this box IF:
            # 1. The class_id matches self.target_class_id
            # 2. The conf is greater than 0.5 (50% sure)
            
            # (Write your if statement here)
            if class_id == self.target_class_id and conf > .5:    
                # box.xyxy gets the coordinates [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Add it to our clean list
                detections.append([int(x1), int(y1), int(x2), int(y2), conf])
                
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