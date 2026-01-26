import cv2
import time
# Importing the classes we built in the other files
from capture import StreamLoader
from vision import ObjectDetector

def main():
    # --- CONFIG ---
    # Use a Twitch stream (e.g., 'firstinspires') or a file path
    #STREAM_URL = "https://www.twitch.tv/ishowspeed"
    #IS_LIVE = True
    # --- CONFIGURATION ---

    # Use local file for training data collection
    STREAM_URL = "data/raw/match_video.mp4" # Make sure this matches your file name
    IS_LIVE = False

    # --- INITIALIZATION ---
    print("1. Initializing Capture...")
    # TODO: Instantiate your StreamLoader (Pass STREAM_URL and IS_LIVE)
    # loader = ...
    loader = StreamLoader(STREAM_URL, IS_LIVE)

    print("2. Initializing Vision...")
    # TODO: Instantiate your ObjectDetector
    # detector = ...
    detector = ObjectDetector()

    print("3. Starting Loop. Click video window and press 'q' to quit.")
    cv2.namedWindow("FRC AI Scout", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("FRC AI Scout", 1280, 720)
    prev_time = 0 

    try:
        while True:
            # --- PHASE 1: INPUT ---
            # TODO: Get a frame from the loader
            # frame, ret = ...
            frame, ret = loader.get_frame()

            # Sanity check: If the frame is empty (stream buffering), skip this loop iteration
            if not ret:
                time.sleep(0.1) # Wait a tiny bit so we don't spam the CPU
                continue

            # --- PHASE 2: PROCESS ---
            # TODO: Pass the frame to the detector to get the list of boxes
            # detections = ...
            detections = detector.detect(frame)
            # --- PHASE 3: RENDER ---
            # Loop through the data and draw it
            for (x1, y1, x2, y2, conf, class_id, track_id) in detections:
                # Draw the rectangle
                if class_id == 0:
                    color = (255, 0, 0) #Blue in BGR
                    label = f"blue {track_id} {conf:.2f}"
                else:
                    color = (0, 0, 255) #Red in BGR
                    label = f"red {track_id} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                    
                cv2.putText(frame, label, (x1, max(0, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- FPS COUNTER (Pure Math) ---
            curr_time = time.time()
            # FPS = 1 second / time_taken_for_one_loop
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            # Draw FPS on top left
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show the result
            cv2.imshow("FRC AI Scout", frame)

            
            # Exit Logic (Press 'q')
            key = cv2.waitKey(1) & 0xFF
            if key ==ord('q'):
                break
            
            # Press 's' to save a Snapshot for training
            elif key == ord('s'):
                # Create a unique filename using the timestamp
                filename = f"data/raw/train_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Snapshot saved: {filename}")


    except KeyboardInterrupt:
        print("\nUser stopped the program.")
        
    finally:
        # --- CLEANUP ---
        # This block runs even if the app crashes. 
        # Crucial for releasing the camera/network socket so it doesn't get stuck open.
        print("Releasing resources...")
        if 'loader' in locals():
            loader.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()