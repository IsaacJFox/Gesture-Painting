"""Simple geometric gestures; keep your fingers upright and palm facing camera."""
import math


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(hand):
    p = hand.landmark
    # Compare fingertip distance from wrist with the middle joint distance.
    return tuple(distance(p[t], p[0]) > 1.15 * distance(p[j], p[0])
                 for t, j in ((8, 6), (12, 10), (16, 14), (20, 18)))


class GestureRecognition:
    def isHandOpen(self, hand):
        return all(fingers_up(hand))

    def recognizeGestures(self, hands):
        if len(hands) >= 2 and all(self.isHandOpen(h) for h in hands[:2]):
            return 'clear'
        if not hands:
            return 'pause'
        fingers = fingers_up(hands[0])
        if all(fingers):
            return 'erase'
        if fingers == (True, True, False, False):
            return 'highlight'
        if fingers == (True, False, False, False):
            return 'draw'
        return 'pause'


class ClearHold:
    """Fire once per continuous hold; release both palms before rearming."""
    def __init__(self, seconds=2.0):
        self.seconds = seconds
        self.started = None
        self.fired = False

    def update(self, active, now):
        if not active:
            self.started, self.fired = None, False
            return False, 0.0
        if self.started is None:
            self.started = now
        progress = min(1.0, (now - self.started) / self.seconds)
        trigger = progress >= 1 and not self.fired
        if trigger:
            self.fired = True
        return trigger, progress
