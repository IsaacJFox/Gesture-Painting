"""Webcam hand tracking. Press Q to quit."""

import argparse
import time

import cv2
import mediapipe as mp


class HandTracking:
    def __init__(self, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        # This demo uses the legacy API, so install the pinned requirements.
        self.hand_api = mp.solutions.hands
        self.drawing = mp.solutions.drawing_utils
        self.tracker = self.hand_api.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.results = None  # latest MediaPipe result, filled by detectHand()

    def detectHand(self, frame):
        """Run MediaPipe on a BGR frame and return the list of detected hands."""
        # OpenCV captures BGR; MediaPipe expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.tracker.process(rgb)
        return self.results.multi_hand_landmarks or []

    def trackFingers(self, frame, draw=True):
        """Detect hands, draw the skeleton, and return (frame, hands)."""
        hands = self.detectHand(frame)
        if draw:
            for hand in hands:
                self.drawing.draw_landmarks(frame, hand, self.hand_api.HAND_CONNECTIONS)
        return frame, hands

    def getFingerPostion(self, frame, hand_index=0,
                         finger=None):
        """Return the (x, y) pixel position of a fingertip, or None if no hand.

        Defaults to the index fingertip (landmark 8).
        """
        if finger is None:
            finger = self.hand_api.HandLandmark.INDEX_FINGER_TIP
        if not self.results or not self.results.multi_hand_landmarks:
            return None
        if hand_index >= len(self.results.multi_hand_landmarks):
            return None
        point = self.results.multi_hand_landmarks[hand_index].landmark[finger]
        height, width = frame.shape[:2]
        return int(point.x * width), int(point.y * height)

    def close(self):
        self.tracker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    window = "Gesture Painting - Hand Tracking Tutorial"
    camera = cv2.VideoCapture(args.camera)
    hand_tracker = HandTracking()
    try:
        if not camera.isOpened():
            raise RuntimeError("Cannot open webcam. Check camera permissions or try --camera 1.")

        previous_time = time.perf_counter()
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read webcam frame. Close other camera apps.")

            # Mirror the picture so moving right also looks like moving right.
            frame = cv2.flip(frame, 1)
            frame, hands = hand_tracker.trackFingers(frame)

            # Index fingertip as a pixel position for the future painting cursor.
            for i in range(len(hands)):
                position = hand_tracker.getFingerPostion(frame, hand_index=i)
                if position:
                    cv2.circle(frame, position, 12, (0, 255, 255), -1)

            now = time.perf_counter()
            fps = 1 / max(now - previous_time, 0.000001)
            previous_time = now
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 85), (30, 30, 30), -1)
            cv2.putText(frame, f"Hands: {len(hands)} | FPS: {fps:.0f}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Move your hand | Yellow dot = index tip | Q: quit",
                        (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Release resources even if an error occurs.
        hand_tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
