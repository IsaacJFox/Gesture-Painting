"""Gesture Painting blackboard. Run: python main.py"""
import argparse
import math
import time
from collections import deque
import cv2
from hand_tracking import HandTracking
from gestures import GestureRecognition, ClearHold
from drawing import Drawing

WIDTH, HEIGHT = 1280, 720
TOP, BOTTOM = 85, 670
WINDOW = 'Gesture Painting'
COLORS = [('White', (255, 255, 255)), ('Green', (80, 255, 80)),
          ('Blue', (255, 170, 60)), ('Pink', (200, 80, 255))]

# Live smoothing dial ([ and ] keys): (MIN_CUTOFF, BETA, STRING_LENGTH).
# Left = more responsive, right = steadier but the pen trails your finger more.
SMOOTHING_LEVELS = [
    (3.0, 0.010, 0.0),
    (2.0, 0.008, 0.0),
    (1.5, 0.006, 0.0),
    (1.0, 0.005, 0.0),
    (0.8, 0.004, 0.0),
    (0.6, 0.004, 2.0),   # default
    (0.5, 0.003, 3.0),
    (0.4, 0.003, 4.0),
    (0.3, 0.002, 5.0),
]
# One notch steadier than before; press [ if the brush feels too slow.
DEFAULT_SMOOTHING = 5


def wobble(points):
    """Average zig-zag of the last few points, in pixels (lower = steadier)."""
    pts = list(points)[-30:]
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        total += math.hypot(a[0] - 2 * b[0] + c[0], a[1] - 2 * b[1] + c[1])
    return total / (len(pts) - 2)


def draw_trail(display, trail, color, thickness):
    pts = [(int(x), int(y)) for x, y in trail]
    for a, b in zip(pts, pts[1:]):
        cv2.line(display, a, b, color, thickness, cv2.LINE_AA)


class GesturePainting:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.board = Drawing(WIDTH, HEIGHT)
        # pause_votes: frames needed to end a stroke (higher = fewer accidental
        # stroke breaks, lower = strokes end faster when you lift your finger).
        self.recognizer = GestureRecognition(pause_votes=4, aspect=WIDTH / HEIGHT)
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
        self.fps = 0.0
        self.last_frame = time.monotonic()
        # Debug view (press D): raw fingertip vs smoothed pen, and timings.
        self.debug = False
        self.raw_trail = deque(maxlen=90)
        self.pen_trail = deque(maxlen=90)
        self.times = [0.0, 0.0, 0.0]  # ms: camera wait, tracking, drawing
        self.smoothing = DEFAULT_SMOOTHING
        self.cursor_position = None
        self.apply_smoothing()

    def apply_smoothing(self):
        min_cutoff, beta, string = SMOOTHING_LEVELS[self.smoothing]
        self.board.MIN_CUTOFF = min_cutoff
        self.board.BETA = beta
        self.board.STRING_LENGTH = string

    def notify(self, message):
        self.message, self.message_until = message, time.monotonic() + 3

    def start_webcam(self):
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            raise RuntimeError('Cannot open webcam. Allow camera access or try --camera 1.')
        # MJPG lets most webcams reach 30 fps at 720p. The default format is
        # often slower, and some cameras drop their frame rate in dim light,
        # which makes strokes look jerky.
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
                t0 = time.perf_counter()
                ok, frame = self.camera.read()
                if not ok:
                    raise RuntimeError('Webcam frame unavailable. Close other camera apps.')
                frame = cv2.resize(cv2.flip(frame, 1), (WIDTH, HEIGHT))
                t1 = time.perf_counter()
                _, hands = self.tracker.trackFingers(frame, draw=False)
                t2 = time.perf_counter()
                mode = self.recognizer.recognizeGestures(hands)
                now = time.monotonic()
                self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_frame, 1e-6)
                self.last_frame = now
                clear, progress = self.clear_hold.update(mode == 'clear', now)
                if clear:
                    self.board.clearCanvas()
                    self.notify('Board cleared - lower hands to continue')
                # The pen follows the same hand that decided the gesture.
                primary = self.recognizer.primary
                point = None
                if primary is not None:
                    p = primary.landmark
                    if mode == 'erase':
                        x = sum(p[i].x for i in (0, 5, 9, 13, 17)) / 5
                        y = sum(p[i].y for i in (0, 5, 9, 13, 17)) / 5
                    else:
                        x, y = p[8].x, p[8].y
                    # Keep float precision: the smoothing filter works better
                    # without whole-pixel rounding.
                    point = (max(0.0, min(WIDTH - 1.0, x * WIDTH)),
                             max(0.0, min(HEIGHT - 1.0, y * HEIGHT)))
                    self.select_color((int(point[0]), int(point[1])), mode, now)
                else:
                    self.hover_target = None
                if point and TOP + 50 <= point[1] < BOTTOM - 50 and mode in ('draw', 'highlight', 'erase'):
                    self.board.stroke(point, mode)
                    if mode in ('draw', 'highlight') and self.board.previous:
                        self.raw_trail.append(point)
                        self.pen_trail.append(self.board.previous)
                else:
                    self.board.stop()
                    self.raw_trail.clear()
                    self.pen_trail.clear()
                background = frame if self.webcam_background else None
                display = self.board.render(background)
                cv2.rectangle(display, (0, 0), (WIDTH, TOP), (32, 32, 32), -1)
                cv2.rectangle(display, (0, BOTTOM), (WIDTH, HEIGHT), (32, 32, 32), -1)
                shapes = 'ON' if self.board.auto_shapes else 'OFF'
                label = f'Gesture Painting | {mode.upper()} | Auto-shapes: {shapes} | {self.fps:.0f} fps'
                cv2.putText(display, label, (15, 30), 0, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f'Brush: {self.board.brush_size} | {COLORS[self.color_index][0]} | Smoothing {self.smoothing}/{len(SMOOTHING_LEVELS) - 1}',
                            (15, 60), 0, 0.5, (200, 200, 200), 1)
                for i, (name, color) in enumerate(COLORS):
                    left = 650 + i * 130
                    cv2.rectangle(display, (left, 10), (left+120, 65), color, 2 if i != self.color_index else 4)
                    cv2.putText(display, name, (left+10, 43), 0, 0.55, color, 1)
                # Small live preview helps position hands even on a black board.
                preview = cv2.resize(frame, (240, 135))
                display[TOP:TOP+135, WIDTH-240:WIDTH] = preview
                if point:
                    if self.board.previous is not None:
                        # Show the actual smoothed brush/eraser position, not
                        # the noisy raw landmark used to calculate it.
                        self.cursor_position = self.board.previous
                    elif self.cursor_position is None:
                        self.cursor_position = point
                    else:
                        # In PAUSE the brush is inactive. Smooth just the
                        # visible cursor so it doesn't tremble while idle.
                        alpha = 0.30
                        self.cursor_position = (
                            self.cursor_position[0] + alpha * (point[0] - self.cursor_position[0]),
                            self.cursor_position[1] + alpha * (point[1] - self.cursor_position[1]),
                        )
                    cursor = (int(round(self.cursor_position[0])),
                              int(round(self.cursor_position[1])))
                    cv2.circle(display, cursor, 50 if mode == 'erase' else 9, (160, 160, 160), 2)
                else:
                    self.cursor_position = None
                if self.debug:
                    # Red = raw fingertip from the tracker, green = smoothed pen.
                    draw_trail(display, self.raw_trail, (60, 60, 255), 1)
                    draw_trail(display, self.pen_trail, (80, 255, 80), 2)
                    cam_ms, track_ms, draw_ms = self.times
                    cv2.putText(display, f'cam wait {cam_ms:.0f} ms | tracking {track_ms:.0f} ms | draw {draw_ms:.0f} ms',
                                (15, 110), 0, 0.55, (0, 200, 255), 1)
                    cv2.putText(display, f'wobble raw {wobble(self.raw_trail):.1f} px | pen {wobble(self.pen_trail):.1f} px | glitches dropped {self.board.spikes}',
                                (15, 135), 0, 0.55, (0, 200, 255), 1)
                if mode == 'clear':
                    cv2.putText(display, f'Hold BOTH palms to clear: {progress:.0%}',
                                (300, 120), 0, 0.7, (0, 220, 255), 2)
                    cv2.rectangle(display, (300, 130), (300+int(500*progress), 140), (0, 220, 255), -1)
                cv2.putText(display, 'Index: draw | 2 fingers: highlight | Palm: wipe | Both palms 2s: clear',
                            (15, 692), 0, 0.53, (230, 230, 230), 1)
                cv2.putText(display, 'B: camera | S: save | 1-4: color | +/-: brush | [ ]: smooth | T: shapes | D: debug | Q: quit',
                            (15, 713), 0, 0.48, (180, 180, 180), 1)
                if now < self.message_until:
                    cv2.putText(display, self.message, (20, 640), 0, 0.65, (0, 220, 255), 2)
                cv2.imshow(WINDOW, display)
                t3 = time.perf_counter()
                sample = ((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000)
                self.times = [0.9 * old + 0.1 * new for old, new in zip(self.times, sample)]
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('d'):
                    self.debug = not self.debug
                elif key == ord('t'):
                    # Finish the current stroke before switching modes.
                    self.board.stop()
                    self.board.auto_shapes = not self.board.auto_shapes
                    self.notify('Auto-shapes ' + ('ON' if self.board.auto_shapes else 'OFF'))
                elif key in (ord('['), ord(']')):
                    step = -1 if key == ord('[') else 1
                    self.smoothing = max(0, min(len(SMOOTHING_LEVELS) - 1, self.smoothing + step))
                    self.apply_smoothing()
                    self.notify(f'Smoothing {self.smoothing}/{len(SMOOTHING_LEVELS) - 1}')
                elif key == ord('b'):
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
    parser.add_argument('--camera', type=int, default=1)
    args = parser.parse_args()
    GesturePainting(args.camera).run()

