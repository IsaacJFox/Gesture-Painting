"""Gesture recognition with filtering to prevent accidental switching."""

import math
import time
from collections import deque


# MediaPipe landmarks are normalized separately along x and y. Correct for
# the camera aspect ratio so fingertip distances have the right proportions.
DEFAULT_ASPECT = 1280 / 720


def distance(a, b, aspect=DEFAULT_ASPECT):
    return math.hypot((a.x - b.x) * aspect, a.y - b.y)


def fingers_up(hand, aspect=DEFAULT_ASPECT):
    """Return (index, middle, ring, pinky) as booleans."""
    p = hand.landmark

    def d(i, j):
        return distance(p[i], p[j], aspect)

    index_up = d(8, 5) > 1.5 * d(6, 5)
    middle_up = d(12, 0) > 1.30 * d(10, 0)
    ring_up = d(16, 0) > 1.15 * d(14, 0)
    pinky_up = d(20, 0) > 1.15 * d(18, 0)

    return index_up, middle_up, ring_up, pinky_up


def classify(fingers):
    """Recognize drawing, highlighting, erasing, and pausing."""
    index, middle, ring, pinky = fingers

    if index and middle and ring and pinky:
        return 'erase'
    if index and middle and not ring and not pinky:
        return 'highlight'
    # Back to the original pointing gesture: index raised, middle curled.
    # The ring and pinky can be relaxed, but not both fully extended.
    if index and not middle and not (ring and pinky):
        return 'draw'
    return 'pause'


class GestureRecognition:
    def __init__(
        self,
        history_size=7,
        votes_required=5,
        missing_grace_frames=2,
        pause_votes=3,
        clear_grace_frames=3,
        aspect=DEFAULT_ASPECT,
    ):
        self.mode = 'pause'
        self.primary = None
        self.aspect = aspect

        self._recent = deque(maxlen=history_size)
        self._votes = {
            'pause': pause_votes,
            'draw': votes_required,
            'clear': votes_required,
            'highlight': votes_required + 1,
            'erase': votes_required + 1,
        }
        self._missing_grace_frames = missing_grace_frames
        self._missing_frames = 0
        self._clear_grace_frames = clear_grace_frames
        self._clear_lost = 0
        self._primary_wrist = None

    def isHandOpen(self, hand):
        return all(fingers_up(hand, self.aspect))

    def _pick_primary(self, hands):
        """Continue tracking the same hand if MediaPipe reorders hands."""
        if len(hands) == 1 or self._primary_wrist is None:
            chosen = hands[0]
        else:
            px, py = self._primary_wrist

            def gap(hand):
                wrist = hand.landmark[0]
                return math.hypot(
                    (wrist.x - px) * self.aspect, wrist.y - py
                )

            chosen = min(hands, key=gap)

        wrist = chosen.landmark[0]
        self._primary_wrist = wrist.x, wrist.y
        return chosen

    def _detect_gesture(self, hands, primary):
        if len(hands) >= 2 and all(self.isHandOpen(h) for h in hands[:2]):
            return 'clear'
        return classify(fingers_up(primary, self.aspect))

    def _drop_to_pause(self):
        self.mode = 'pause'
        self._recent.clear()
        self._clear_lost = 0

    def recognizeGestures(self, hands):
        if not hands:
            self._missing_frames += 1
            # Allow a few missing camera frames without changing modes.
            if self.mode == 'clear':
                self._clear_lost += 1
                if self._clear_lost > self._clear_grace_frames:
                    self._drop_to_pause()
            elif self._missing_frames > self._missing_grace_frames:
                self._drop_to_pause()

            if self._missing_frames > self._missing_grace_frames:
                self.primary = None
                self._primary_wrist = None
            return self.mode

        self._missing_frames = 0
        self.primary = self._pick_primary(hands)
        observed = self._detect_gesture(hands, self.primary)

        if self.mode == 'clear':
            if observed == 'clear':
                self._clear_lost = 0
                return 'clear'
            self._clear_lost += 1
            if self._clear_lost <= self._clear_grace_frames:
                return 'clear'
            self._drop_to_pause()

        # Confirm a gesture over multiple frames to avoid mode flicker.
        self._recent.append(observed)
        needed = min(self._recent.maxlen, self._votes[observed])
        if self._recent.count(observed) >= needed:
            if observed == 'clear':
                self._clear_lost = 0
            self.mode = observed
        return self.mode


class ClearHold:
    def __init__(self, seconds=2.0):
        self.seconds = seconds
        self.started = None
        self.fired = False

    def update(self, active, now=None):
        if now is None:
            now = time.monotonic()

        if not active:
            self.started = None
            self.fired = False
            return False, 0.0

        if self.started is None:
            self.started = now

        progress = min(1.0, (now - self.started) / self.seconds)
        trigger = progress >= 1 and not self.fired

        if trigger:
            self.fired = True

        return trigger, progress
