import cv2
import streamlink
import time

class StreamLoader:
    def __enter__(self):
        # This runs when you enter the 'with' block
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # This runs when you leave the 'with' block (even if there is an error!)
        self.release()
        
    def __init__(self, source_url, is_live=False):
        """
        Args:
            source_url (str): Twitch URL (if live) or File Path (if not live)
            is_live (bool): Boolean to toggle Twitch logic
        """
        self.twitch_url = source_url
        self.stream_url = source_url
        self.is_live = is_live
        self.cap = None
        
        # Initialize the connection immediately
        self.connect()

    def connect(self):
        """
        Handles the logic to open the video capture.
        If live, it uses streamlink to get the .m3u8 URL first.
        """
        if self.is_live:
            print(f"Attempting to connect to Twitch stream: {self.twitch_url}")
            try:
                # TODO: Use streamlink.streams(self.twitch_url) to get available streams
                streams = streamlink.streams(self.twitch_url)
                # TODO: Select the 'best' or '720p' stream
                if 'best' in streams:
                    self.stream_url = streams.get('best').to_url()
                elif '720p' in streams:
                    self.stream_url = streams.get('720p').to_url()
                else:
                    first_key = next(iter(streams), None)
                    self.stream_url = streams.get(first_key).to_url() if first_key else None
                if self.stream_url is None:
                    print("Error: No suitable stream found.")
                    return
            except Exception as e:
                print(f"Error getting stream from Twitch/YouTube: {e}")
                return

        # Initialize OpenCV VideoCapture
        # TODO: Set self.cap = cv2.VideoCapture(...) with your (possibly updated) source_url
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            print("Error: Could not open the video source.")

    def get_frame(self):
        """
        Reads a single frame.
        Returns:
            frame: The image array
            ret: Boolean (True if frame is valid)
        """
        if self.cap is None:
            return None, False

        # TODO: call self.cap.read()
        ret, frame = self.cap.read()
        # TODO: If ret is False and self.is_live is True, maybe try calling self.connect() again? (Bonus points for reconnection logic)
        if not ret and self.is_live:
            print("Reconnecting to live stream...")
            self.connect()
            ret, frame = self.cap.read()
        return frame, ret

    def release(self):
        """Closes the stream."""
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    # TEST CODE: This block only runs if you run 'python src/capture.py' directly
    # Use this to verify your class works without running the whole app.
    
    # Test with a local file (download one first) or a twitch stream
    loader = StreamLoader("https://www.twitch.tv/firstinrobotics", is_live=True)
    
    while True:
        frame, ret = loader.get_frame()
        if not ret:
            break
        
        cv2.imshow("Test Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    loader.release()
    cv2.destroyAllWindows()