"""Gesture Painting blackboard. Run: python main.py"""
import argparse
import time
import cv2
from handTracking import HandTracking
from gestures import GestureRecognition, ClearHold
from drawing import Drawing

WIDTH, HEIGHT = 1280, 720
TOP, BOTTOM = 85, 670
WINDOW = 'Gesture Painting'
COLORS = [('White', (255, 255, 255)), ('Green', (80, 255, 80)),
          ('Blue', (255, 170, 60)), ('Pink', (200, 80, 255))]


class GesturePainting:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.board = Drawing(WIDTH, HEIGHT)
        self.recognizer = GestureRecognition()
        self.clear_hold = ClearHold()
        self.camera = None
        self.tracker = None
        self.webcam_background = False
        self.message = ''
        self.message_until = 0
        self.color_index = 0
        self.hover_target = None
        self.hover_since = 0
        self.hover_fired = False

    def notify(self, message):
        self.message, self.message_until = message, time.monotonic() + 3

    def start_webcam(self):
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            raise RuntimeError('Cannot open webcam. Allow camera access or try --camera 1.')
        self.tracker = HandTracking(max_hands=2)

    def select_color(self, point, mode, now):
        x, y = point
        target = (x - 650) // 130 if 650 <= x < 1170 and 10 <= y <= 65 and mode == 'draw' else None
        if target != self.hover_target:
            self.hover_target, self.hover_since, self.hover_fired = target, now, False
        if target is not None and now - self.hover_since >= 0.7 and not self.hover_fired:
            self.color_index = int(target)
            self.board.color = COLORS[self.color_index][1]
            self.hover_fired = True
        return target

    def run(self):
        try:
            self.start_webcam()
            while True:
                ok, frame = self.camera.read()
                if not ok:
                    raise RuntimeError('Webcam frame unavailable. Close other camera apps.')
                frame = cv2.resize(cv2.flip(frame, 1), (WIDTH, HEIGHT))
                _, hands = self.tracker.trackFingers(frame, draw=False)
                mode = self.recognizer.recognizeGestures(hands)
                now = time.monotonic()
                clear, progress = self.clear_hold.update(mode == 'clear', now)
                if clear:
                    self.board.clearCanvas()
                    self.notify('Board cleared - lower hands to continue')
                point = None
                if hands:
                    p = hands[0].landmark
                    if mode == 'erase':
                        x = sum(p[i].x for i in (0, 5, 9, 13, 17)) / 5
                        y = sum(p[i].y for i in (0, 5, 9, 13, 17)) / 5
                    else:
                        x, y = p[8].x, p[8].y
                    point = (max(0, min(WIDTH-1, int(x * WIDTH))),
                             max(0, min(HEIGHT-1, int(y * HEIGHT))))
                    self.select_color(point, mode, now)
                else:
                    self.hover_target = None
                if point and TOP + 50 <= point[1] < BOTTOM - 50 and mode in ('draw', 'highlight', 'erase'):
                    self.board.stroke(point, mode)
                else:
                    self.board.stop()
                background = frame if self.webcam_background else None
                display = self.board.render(background)
                cv2.rectangle(display, (0, 0), (WIDTH, TOP), (32, 32, 32), -1)
                cv2.rectangle(display, (0, BOTTOM), (WIDTH, HEIGHT), (32, 32, 32), -1)
                label = f'Gesture Painting | {mode.upper()}'
                cv2.putText(display, label, (15, 30), 0, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f'Brush: {self.board.brush_size} | {COLORS[self.color_index][0]}',
                            (15, 60), 0, 0.5, (200, 200, 200), 1)
                for i, (name, color) in enumerate(COLORS):
                    left = 650 + i * 130
                    cv2.rectangle(display, (left, 10), (left+120, 65), color, 2 if i != self.color_index else 4)
                    cv2.putText(display, name, (left+10, 43), 0, 0.55, color, 1)
                # Small live preview helps position hands even on a black board.
                preview = cv2.resize(frame, (240, 135))
                display[TOP:TOP+135, WIDTH-240:WIDTH] = preview
                if point:
                    cv2.circle(display, point, 50 if mode == 'erase' else 9, (160, 160, 160), 2)
                if mode == 'clear':
                    cv2.putText(display, f'Hold BOTH palms to clear: {progress:.0%}',
                                (300, 120), 0, 0.7, (0, 220, 255), 2)
                    cv2.rectangle(display, (300, 130), (300+int(500*progress), 140), (0, 220, 255), -1)
                cv2.putText(display, 'Index: draw | 2 fingers: highlight | Palm: wipe | Both palms 2s: clear',
                            (15, 692), 0, 0.53, (230, 230, 230), 1)
                cv2.putText(display, 'B: background | S: save | 1-4: color | +/-: brush | Q: quit',
                            (15, 713), 0, 0.48, (180, 180, 180), 1)
                if now < self.message_until:
                    cv2.putText(display, self.message, (20, 640), 0, 0.65, (0, 220, 255), 2)
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('b'):
                    self.webcam_background = not self.webcam_background
                    self.board.stop()
                elif key == ord('s'):
                    path = self.board.saveDrawing(background)
                    self.notify('Saved: ' + path.name)
                elif key in map(ord, '1234'):
                    self.color_index = key - ord('1')
                    self.board.color = COLORS[self.color_index][1]
                    self.board.stop()
                elif key in (ord('+'), ord('='), ord('-')):
                    self.board.brush_size = max(2, min(25, self.board.brush_size + (-1 if key == ord('-') else 1)))
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.close_program()

    def close_program(self):
        if self.tracker:
            self.tracker.close()
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    GesturePainting(args.camera).run()
