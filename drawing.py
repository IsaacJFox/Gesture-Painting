"""Persistent ink and transparent highlights, independent of webcam frames."""
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np


class Drawing:
    def __init__(self, width=1280, height=720):
        self.ink = np.zeros((height, width, 3), dtype=np.uint8)
        self.ink_mask = np.zeros((height, width), dtype=np.uint8)
        self.highlight_mask = np.zeros((height, width), dtype=np.uint8)
        self.color = (255, 255, 255)
        self.brush_size = 5
        self.previous = None
        self.previous_mode = None

    def stop(self):
        self.previous = None
        self.previous_mode = None

    def stroke(self, point, mode):
        start = self.previous if self.previous_mode == mode else point
        # Break a line if tracking suddenly jumps across the board.
        if start is None or np.linalg.norm(np.array(start) - point) > 180:
            start = point
        if mode == 'draw':
            cv2.line(self.ink, start, point, self.color, self.brush_size, cv2.LINE_AA)
            cv2.line(self.ink_mask, start, point, 255, self.brush_size, cv2.LINE_AA)
        elif mode == 'highlight':
            # A mask prevents each webcam frame from making the same mark darker.
            cv2.line(self.highlight_mask, start, point, 255, 30, cv2.LINE_AA)
        elif mode == 'erase':
            for layer in (self.ink, self.ink_mask, self.highlight_mask):
                cv2.line(layer, start, point, 0, 100, cv2.LINE_8)
        else:
            self.stop()
            return
        self.previous, self.previous_mode = point, mode

    def clearCanvas(self):
        self.ink.fill(0)
        self.ink_mask.fill(0)
        self.highlight_mask.fill(0)
        self.stop()

    def render(self, background=None):
        base = np.zeros_like(self.ink) if background is None else background.copy()
        alpha = (self.ink_mask.astype(np.float32) / 255)[..., None]
        base = base * (1 - alpha) + self.ink.astype(np.float32) * alpha
        highlight_alpha = (self.highlight_mask.astype(np.float32) / 255 * 0.35)[..., None]
        base = base * (1 - highlight_alpha) + np.array([0, 255, 255]) * highlight_alpha
        return base.clip(0, 255).astype(np.uint8)

    def saveDrawing(self, background=None):
        folder = Path(__file__).resolve().parent / 'saved_drawings'
        folder.mkdir(exist_ok=True)
        path = folder / (datetime.now().strftime('drawing_%Y%m%d_%H%M%S_%f') + '.png')
        if not cv2.imwrite(str(path), self.render(background)):
            raise OSError('Could not save drawing')
        return path
