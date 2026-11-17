"""MediaPipe hand tracking for Gesture Painting (MediaPipe 0.10.21)."""

import argparse
import time

import cv2
import mediapipe as mp


class HandTracking:
    def __init__(self, max_hands=2, detection_confidence=0.5,
                 tracking_confidence=0.5, model_complexity=1,
                 processing_scale=1.0):
        """Track hands, optionally using smaller frames to improve speed.

        Leave processing_scale at 1.0 for the most detail. Try 0.75 if
        tracking is slow; smaller frames can reduce fingertip accuracy.
        """
        if not 0.5 <= processing_scale <= 1.0:
            raise ValueError("processing_scale must be between 0.5 and 1.0")

        self.processing_scale = processing_scale
        self.processing_ms = 0.0
        self.hand_api = mp.solutions.hands
        self.drawing = mp.solutions.drawing_utils
        self.tracker = self.hand_api.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.results = None

    def detectHand(self, frame):
        """Detect hands and store landmarks for getFingerPosition()."""
        if frame is None or frame.size == 0:
            self.results = None
            return []

        # MediaPipe uses normalized coordinates, so detections still map
        # directly onto the ORIGINAL frame when processing is downscaled.
        processing_frame = frame
        if self.processing_scale < 1.0:
            processing_frame = cv2.resize(
                frame, None,
                fx=self.processing_scale,
                fy=self.processing_scale,
                interpolation=cv2.INTER_AREA,
            )

        rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        start = time.perf_counter()
        self.results = self.tracker.process(rgb)
        self.processing_ms = (time.perf_counter() - start) * 1000
        return self.results.multi_hand_landmarks or []

    def trackFingers(self, frame, draw=True):
        """Preserve the (frame, hands) return format used by main.py."""
        hands = self.detectHand(frame)
        if draw and frame is not None:
            for hand in hands:
                self.drawing.draw_landmarks(
                    frame, hand, self.hand_api.HAND_CONNECTIONS
                )
        return frame, hands

    def getFingerPosition(self, frame, hand_index=0, finger=None):
        """Return the requested landmark as an (x, y) pixel position."""
        if finger is None:
            finger = self.hand_api.HandLandmark.INDEX_FINGER_TIP

        if (frame is None or self.results is None
                or not self.results.multi_hand_landmarks
                or not 0 <= hand_index < len(self.results.multi_hand_landmarks)):
            return None

        point = self.results.multi_hand_landmarks[hand_index].landmark[finger]
        height, width = frame.shape[:2]
        return (
            max(0, min(width - 1, int(point.x * width))),
            max(0, min(height - 1, int(point.y * height))),
        )

    # Keep the original project's misspelled method working too.
    def getFingerPostion(self, frame, hand_index=0, finger=None):
        return self.getFingerPosition(frame, hand_index, finger)

    def isHandLowQuality(self, hand_index=0):
        """Flag hands appearing very small in the frame (a rough warning).

        This is not MediaPipe's confidence score. It only checks the size of
        the visible hand, so it doesn't block drawing or change gestures.
        """
        if (self.results is None or not self.results.multi_hand_landmarks
                or not 0 <= hand_index < len(self.results.multi_hand_landmarks)):
            return False

        landmarks = self.results.multi_hand_landmarks[hand_index].landmark
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return max(width, height) < 0.12

    def close(self):
        self.tracker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Tracking resolution: 1.0 for detail, 0.75 for speed')
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    tracker = None
    try:
        if not camera.isOpened():
            raise RuntimeError('Cannot open webcam. Check permissions or --camera index.')

        tracker = HandTracking(processing_scale=args.scale)
        last = time.perf_counter()
        fps = 0.0

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('Could not read a webcam frame.')

            frame = cv2.flip(frame, 1)
            frame, hands = tracker.trackFingers(frame, draw=True)

            for i in range(len(hands)):
                point = tracker.getFingerPosition(frame, hand_index=i)
                if point is not None:
                    cv2.circle(frame, point, 11, (0, 255, 255), -1)

            if any(tracker.isHandLowQuality(i) for i in range(len(hands))):
                cv2.putText(frame, 'Move hand closer to camera',
                            (12, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 220, 255), 2)

            now = time.perf_counter()
            current_fps = 1.0 / max(now - last, 1e-6)
            fps = current_fps if fps == 0.0 else 0.9 * fps + 0.1 * current_fps
            last = now

            label = (f'Hands: {len(hands)} | FPS: {fps:.0f} | '
                     f'Tracking: {tracker.processing_ms:.0f} ms')
            cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.imshow('Hand Tracking - Q to quit', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        if tracker is not None:
            tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
