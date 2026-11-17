"""Drawing functionality for Gesture Painting."""

import math
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


class Drawing:

    # Intentionally forgiving: a sketch can be uneven or partly open.
    # Press T in main.py to turn shape correction off for pure freehand work.
    SHAPE_MIN_SIZE = 24
    SHAPE_CLOSURE_RATIO = 0.78

    # Smoothing tuning. Two stages: a One Euro style filter, then an optional
    # "string" stabilizer on top of it (off by default).
    # More smoothing always means more lag, and a pen that trails your finger
    # invites wobble of its own, so change one knob at a time.
    #   Line is shaky/jittery   -> lower MIN_CUTOFF a little (0.8 -> 0.6)
    #   Pen lags behind finger  -> raise MIN_CUTOFF or BETA
    MIN_CUTOFF = 0.8
    BETA = 0.004
    JUMP_LIMIT = 180

    # The pen only moves once your (filtered) fingertip is more than this
    # many pixels away from it, then it is pulled along at that distance.
    # Wobble smaller than this never reaches the canvas.
    # 0 = off. Try 3 to 5 for a stronger dead zone (adds a little lag).
    STRING_LENGTH = 0.0

    # Spike gate: a one-frame tracker glitch (the point leaps away and comes
    # straight back) is dropped instead of being drawn. A move is suspicious
    # when it is farther than GATE_BASE + GATE_SPEED * (expected travel).
    # GATE_BASE = 0 turns the gate off.
    GATE_BASE = 40.0
    GATE_SPEED = 2.5

    # Sub-pixel precision bits used when drawing.
    SHIFT = 4

    # Approximate pixel spacing between samples along a curve.
    CURVE_STEP = 3.0

    def __init__(self, width=1280, height=720):

        self.ink = np.zeros((height, width, 3), dtype=np.uint8)
        self.ink_mask = np.zeros((height, width), dtype=np.uint8)
        self.highlight_mask = np.zeros((height, width), dtype=np.uint8)

        # Solid highlight color (BGR), blended in by render().
        self._tint = np.full((height, width, 3), (0, 255, 255), dtype=np.uint8)

        self.color = (255, 255, 255)
        self.brush_size = 8

        # Draw a roughly closed shape and lift your finger to straighten it.
        # Press T in main.py to turn this on or off.
        self.auto_shapes = True
        self.last_shape = None
        self.shape_feedback = None
        self._shape_points = []
        self._shape_before = None
        self._shape_style = None

        # Keep the original 50px eraser by default. The main program can
        # change this later (for example, from the distance between fingers).
        self.eraser_radius = 50
        self._previous_eraser_radius = None

        self.previous = None
        self.previous_mode = None

        self._filtered_position = None
        self._filtered_velocity = (0.0, 0.0)
        self._previous_raw = None
        self._previous_time = 0.0
        self._pen = None

        # Spike gate state: a suspicious point waiting for confirmation, and
        # a count of glitches dropped so far (shown in the debug view).
        self._pending = None
        self.spikes = 0

        # (last midpoint, last point) of the stroke being drawn.
        self._curve = None

    def stop(self):

        self._finish_stroke()
        self._finish_shape()

        self.previous = None
        self.previous_mode = None
        self._filtered_position = None
        self._pen = None
        self._pending = None
        self._curve = None
        self._previous_eraser_radius = None

    def smooth_point(self, point):
        """Return a smoothed (x, y) as floats."""

        now = perf_counter()

        x = float(point[0])
        y = float(point[1])

        if self._filtered_position is None:
            self._pending = None

        elif self.GATE_BASE > 0:

            last_x, last_y = self._previous_raw
            far = math.hypot(x - last_x, y - last_y) > self._gate_limit(now)

            if self._pending is not None:

                pending_x, pending_y, pending_time = self._pending
                self._pending = None

                if far:
                    # Still far away, so the earlier point was real movement.
                    self._filter_step(pending_x, pending_y, pending_time)

                else:
                    # It leapt away and came straight back: a glitch. Drop it.
                    self.spikes += 1

            elif far:
                # Suspicious jump: hold the pen still and wait one frame to
                # see whether the next point confirms it.
                self._pending = (x, y, now)
                return self._pen

        self._filter_step(x, y, now)

        return self._pen

    def _gate_limit(self, now):
        """Farthest a point may move before it counts as suspicious."""

        dt = min(0.1, max(0.01, now - self._previous_time))
        speed = math.hypot(*self._filtered_velocity)

        return self.GATE_BASE + self.GATE_SPEED * speed * dt

    def _filter_step(self, x, y, now):
        """One step of the One Euro filter plus the string stabilizer."""

        if self._filtered_position is None:

            self._filtered_position = (x, y)
            self._pen = (x, y)
            self._previous_raw = (x, y)
            self._filtered_velocity = (0.0, 0.0)
            self._previous_time = now

        else:

            px, py = self._filtered_position

            # Calculate time between frames.
            dt = min(0.1, max(0.01, now - self._previous_time))

            old_x, old_y = self._previous_raw

            vx = (x - old_x) / dt
            vy = (y - old_y) / dt

            # Smooth the movement speed.
            speed_alpha = dt / (dt + 1.0 / (2 * np.pi * 1.0))

            prev_vx, prev_vy = self._filtered_velocity

            vx = prev_vx + speed_alpha * (vx - prev_vx)
            vy = prev_vy + speed_alpha * (vy - prev_vy)

            self._filtered_velocity = (vx, vy)

            # Adjust smoothing based on movement speed.
            cutoff = self.MIN_CUTOFF + self.BETA * np.hypot(vx, vy)

            alpha = dt / (dt + 1.0 / (2 * np.pi * cutoff))

            # Handle sudden tracking jumps.
            if np.hypot(x - px, y - py) > self.JUMP_LIMIT:

                self._filtered_position = (x, y)
                self._filtered_velocity = (0.0, 0.0)
                self._pen = (x, y)

            else:

                self._filtered_position = (
                    px + alpha * (x - px),
                    py + alpha * (y - py)
                )

            self._previous_raw = (x, y)
            self._previous_time = now

        # String stabilizer: the pen trails the filtered fingertip on a
        # short string, so small wobbles are ignored completely.
        fx, fy = self._filtered_position
        pen_x, pen_y = self._pen

        dx = fx - pen_x
        dy = fy - pen_y
        dist = math.hypot(dx, dy)

        if dist > self.STRING_LENGTH:

            pull = (dist - self.STRING_LENGTH) / dist

            self._pen = (pen_x + dx * pull, pen_y + dy * pull)

    def _stroke_line(self, layer, a, b, color, thickness):
        """Anti-aliased sub-pixel line with a round cap at the end."""

        f = 1 << self.SHIFT

        pa = (int(round(a[0] * f)), int(round(a[1] * f)))
        pb = (int(round(b[0] * f)), int(round(b[1] * f)))

        cv2.line(layer, pa, pb, color, thickness, cv2.LINE_AA, self.SHIFT)

        # Round join so corners between segments stay smooth.
        cv2.circle(
            layer,
            pb,
            max(1, thickness * f // 2),
            color,
            -1,
            cv2.LINE_AA,
            self.SHIFT
        )

    def _targets(self, mode):
        """Layers (and their color/thickness) that a mode paints on."""

        if mode == 'draw':
            return [
                (self.ink, self.color, self.brush_size),
                (self.ink_mask, 255, self.brush_size),
            ]

        if mode == 'highlight':
            return [(self.highlight_mask, 255, 30)]

        return []

    def _draw_curve(self, mode, p0, control, p1):
        """Draw a quadratic Bezier curve from p0 to p1."""

        length = (
            math.hypot(control[0] - p0[0], control[1] - p0[1])
            + math.hypot(p1[0] - control[0], p1[1] - control[1])
        )

        steps = int(min(24, max(2, length / self.CURVE_STEP)))

        targets = self._targets(mode)
        previous = p0

        for i in range(1, steps + 1):

            t = i / steps
            u = 1.0 - t

            q = (
                u * u * p0[0] + 2 * u * t * control[0] + t * t * p1[0],
                u * u * p0[1] + 2 * u * t * control[1] + t * t * p1[1],
            )

            for layer, color, thickness in targets:
                self._stroke_line(layer, previous, q, color, thickness)

            previous = q

    def _finish_stroke(self):
        """Draw the small tail between the last midpoint and last point."""

        if self._curve is not None:

            midpoint, last = self._curve

            for layer, color, thickness in self._targets(self.previous_mode):
                self._stroke_line(layer, midpoint, last, color, thickness)

        self._curve = None

    @staticmethod
    def _point_gap(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @classmethod
    def _simplify_path(cls, points, tolerance):
        """Ramer-Douglas-Peucker: keep corners, remove small hand wobble."""
        if len(points) <= 2:
            return list(points)

        a, b = points[0], points[-1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length_squared = dx * dx + dy * dy
        farthest, farthest_index = 0.0, 0

        for i in range(1, len(points) - 1):
            p = points[i]
            if length_squared < 1e-8:
                distance = cls._point_gap(p, a)
            else:
                t = max(0.0, min(1.0,
                    ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_squared))
                nearest = (a[0] + t * dx, a[1] + t * dy)
                distance = cls._point_gap(p, nearest)
            if distance > farthest:
                farthest, farthest_index = distance, i

        if farthest <= tolerance:
            return [a, b]

        left = cls._simplify_path(points[:farthest_index + 1], tolerance)
        right = cls._simplify_path(points[farthest_index:], tolerance)
        return left[:-1] + right

    @classmethod
    def _closed_corners(cls, points, tolerance):
        """Simplify a loop by splitting it into two open arcs first."""
        farthest_index = max(range(1, len(points)),
                             key=lambda i: cls._point_gap(points[0], points[i]))
        first = cls._simplify_path(points[:farthest_index + 1], tolerance)
        second = cls._simplify_path(points[farthest_index:] + points[:1], tolerance)
        corners = first[:-1] + second[:-1]

        unique = []
        for corner in corners:
            if not unique or cls._point_gap(corner, unique[-1]) > tolerance * 0.5:
                unique.append(corner)
        if len(unique) > 1 and cls._point_gap(unique[0], unique[-1]) <= tolerance:
            unique.pop()
        return unique

    @staticmethod
    def _polygon_area(points):
        return abs(sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(points, points[1:] + points[:1])
        )) / 2.0

    @classmethod
    def _denoise_loop(cls, points):
        """Reduce noisy landmark positions without adding pen lag."""
        ring = list(points)
        if cls._point_gap(ring[0], ring[-1]) < 30:
            ring.pop()  # Don't double-count a repeated start/end sample.
        count = len(ring)
        weights = (1, 2, 3, 2, 1)
        smoothed = []
        for i in range(count):
            neighbors = [ring[(i + offset) % count]
                         for offset in (-2, -1, 0, 1, 2)]
            smoothed.append((
                sum(w * p[0] for w, p in zip(weights, neighbors)) / 9,
                sum(w * p[1] for w, p in zip(weights, neighbors)) / 9,
            ))
        return smoothed + smoothed[:1]

    @staticmethod
    def _minimum_rectangle(corners):
        """Find a neatly aligned rectangle, including rotated rectangles."""
        best = None
        for a, b in zip(corners, corners[1:] + corners[:1]):
            theta = math.atan2(b[1] - a[1], b[0] - a[0])
            cos_a, sin_a = math.cos(theta), math.sin(theta)
            u = [p[0] * cos_a + p[1] * sin_a for p in corners]
            v = [-p[0] * sin_a + p[1] * cos_a for p in corners]
            left, right, top, bottom = min(u), max(u), min(v), max(v)
            area = (right - left) * (bottom - top)
            if best is None or area < best[0]:
                best = (area, theta, left, right, top, bottom)
        return best

    @classmethod
    def _loose_circle(cls, points):
        """Fit a full circle even from a rough, incomplete circular arc."""
        if len(points) < 8:
            return None

        # A least-squares circle fit is much less sensitive to where the user
        # started/stopped than requiring the first and last points to meet.
        coords = np.asarray(points, dtype=np.float64)
        xs, ys = coords[:, 0], coords[:, 1]
        if min(np.ptp(xs), np.ptp(ys)) < cls.SHAPE_MIN_SIZE:
            return None

        matrix = np.column_stack((2 * xs, 2 * ys, np.ones(len(xs))))
        cx, cy, constant = np.linalg.lstsq(matrix, xs * xs + ys * ys, rcond=None)[0]
        radius_squared = cx * cx + cy * cy + constant
        if radius_squared < 400:
            return None
        radius = math.sqrt(radius_squared)
        radii = np.hypot(xs - cx, ys - cy)
        relative_error = float(np.sqrt(np.mean((radii - radius) ** 2)) / radius)

        angles = sorted((math.atan2(y - cy, x - cx) + 2 * math.pi)
                        % (2 * math.pi) for x, y in points)
        gaps = [b - a for a, b in zip(angles, angles[1:])]
        gaps.append(angles[0] + 2 * math.pi - angles[-1])
        covered_degrees = math.degrees(2 * math.pi - max(gaps))

        # Just over half of a clear circle is enough; the fitting error
        # stops ordinary bends from turning into circles.
        if covered_degrees >= 190 and relative_error <= 0.095:
            return 'circle', (float(cx), float(cy), float(radius))
        return None

    @staticmethod
    def _convex_hull(points):
        """Get the outer corners of a drawn shape (no extra dependency)."""
        sorted_points = sorted(set((float(x), float(y)) for x, y in points))
        if len(sorted_points) <= 2:
            return sorted_points

        def cross(a, b, c):
            return ((b[0] - a[0]) * (c[1] - a[1])
                    - (b[1] - a[1]) * (c[0] - a[0]))

        lower, upper = [], []
        for p in sorted_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        for p in reversed(sorted_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    @staticmethod
    def _smooth_shape_points(points):
        """Remove webcam jitter from shape recognition only, not the pen."""
        coords = np.asarray(points, dtype=float)
        if len(coords) < 5:
            return list(points)
        padded = np.pad(coords, ((2, 2), (0, 0)), mode='edge')
        weights = np.asarray((1, 2, 3, 2, 1), dtype=float) / 9
        smoothed = sum(weights[i] * padded[i:i + len(coords)]
                       for i in range(5))
        return [tuple(row) for row in smoothed]

    @classmethod
    def _loose_rectangle(cls, points):
        """Fit a rectangle from as few as three reasonably straight sides."""
        if len(points) < 8:
            return None
        coords = np.asarray(points, dtype=float)
        xs, ys = coords[:, 0], coords[:, 1]
        span = math.hypot(float(np.ptp(xs)), float(np.ptp(ys)))
        path_length = sum(cls._point_gap(a, b)
                          for a, b in zip(points, points[1:]))
        if path_length < span * 1.28:
            return None

        # Try orientations in 5-degree steps. This also handles rotated boxes.
        best = None
        for degrees in range(0, 90, 5):
            angle = math.radians(degrees)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            u = xs * cos_a + ys * sin_a
            v = -xs * sin_a + ys * cos_a
            left, right = float(np.min(u)), float(np.max(u))
            top, bottom = float(np.min(v)), float(np.max(v))
            side_a, side_b = right - left, bottom - top
            if min(side_a, side_b) < cls.SHAPE_MIN_SIZE:
                continue
            if max(side_a, side_b) / min(side_a, side_b) > 4:
                continue

            deviations = np.stack((abs(u - left), abs(u - right),
                                   abs(v - top), abs(v - bottom)))
            score = float(np.mean(np.min(deviations, axis=0))) / min(side_a, side_b)
            sides_seen = len(set(np.argmin(deviations, axis=0).tolist()))
            if sides_seen < 3:
                continue
            if best is None or score < best[0]:
                best = (score, angle, left, right, top, bottom)

        if best is None or best[0] > 0.065:
            return None

        _, theta, left, right, top, bottom = best
        size_a, size_b = right - left, bottom - top
        is_square = 0.72 <= size_a / max(size_b, 1) <= 1.38
        if is_square:
            side = (size_a + size_b) / 2
            mid_a, mid_b = (left + right) / 2, (top + bottom) / 2
            left, right = mid_a - side / 2, mid_a + side / 2
            top, bottom = mid_b - side / 2, mid_b + side / 2

        cos_a, sin_a = math.cos(theta), math.sin(theta)
        corners = [(u * cos_a - v * sin_a, u * sin_a + v * cos_a)
                   for u, v in ((left, top), (right, top),
                                (right, bottom), (left, bottom))]
        return ('square' if is_square else 'rectangle'), corners

    @classmethod
    def _loose_triangle(cls, points):
        """Use the outer three corners to infer an unfinished triangle."""
        if len(points) < 8:
            return None
        hull = cls._convex_hull(points)
        if len(hull) < 3:
            return None
        xs, ys = [p[0] for p in hull], [p[1] for p in hull]
        diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        corners = cls._closed_corners(hull, max(8, diagonal * 0.09))
        if len(corners) != 3:
            return None
        area = cls._polygon_area(corners)
        if area < 0.16 * (max(xs) - min(xs)) * (max(ys) - min(ys)):
            return None

        # Reject arbitrary scribbles inside a triangular bounding outline.
        def segment_gap(p, a, b):
            dx, dy = b[0] - a[0], b[1] - a[1]
            denom = dx * dx + dy * dy
            t = 0 if denom < 1 else max(0, min(1,
                ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom))
            return cls._point_gap(p, (a[0] + t * dx, a[1] + t * dy))

        average_gap = sum(min(segment_gap(p, a, b)
                              for a, b in zip(corners, corners[1:] + corners[:1]))
                          for p in points) / len(points)
        if average_gap > diagonal * 0.055:
            return None

        cx = sum(p[0] for p in corners) / 3
        cy = sum(p[1] for p in corners) / 3
        radius = sum(math.hypot(p[0] - cx, p[1] - cy) for p in corners) / 3
        angle = math.atan2(corners[0][1] - cy, corners[0][0] - cx)
        vertices = [(cx + radius * math.cos(angle + 2 * math.pi * i / 3),
                     cy + radius * math.sin(angle + 2 * math.pi * i / 3))
                    for i in range(3)]
        return 'triangle', vertices

    @classmethod
    def _recognize_shape(cls, points):
        """Return (shape name, geometry) or None for ordinary freehand ink."""
        if len(points) < 6:
            return None

        xs, ys = [p[0] for p in points], [p[1] for p in points]
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        if min(width, height) < cls.SHAPE_MIN_SIZE:
            return None

        diagonal = math.hypot(width, height)
        travel = sum(cls._point_gap(a, b)
                     for a, b in zip(points, points[1:]))
        # A repeated back-and-forth scribble can simplify into three corners.
        # Reject those without requiring a neatly drawn shape.
        if travel > diagonal * 8:
            return None

        # Recognize approximate circular arcs without needing closure first.
        # Smooth only the outline used for deciding its shape. The original
        # drawing still uses the existing pen-stabilizer unchanged.
        sample_points = cls._smooth_shape_points(points)

        circle = cls._loose_circle(sample_points)
        if circle is not None:
            return circle

        box = cls._loose_rectangle(sample_points)
        if box is not None:
            return box

        triangle = cls._loose_triangle(sample_points)
        if triangle is not None:
            return triangle

        # Try possible finishing points from newest to oldest. If the hand
        # curls inward after completing a circle, the full path may fail, but
        # the outline just before the curl can still be recognized.
        limit = max(55, min(240, diagonal * cls.SHAPE_CLOSURE_RATIO))
        stride = max(1, len(points) // 120)
        for end in range(len(points) - 1, max(7, len(points) // 2), -stride):
            if cls._point_gap(points[0], points[end]) > limit:
                continue
            match = cls._recognize_closed_shape(points[:end + 1])
            if match is not None:
                return match
        return None

    @classmethod
    def _recognize_closed_shape(cls, points):
        """Fit one completed outline to a geometric shape."""

        if len(points) < 6:
            return None

        points = cls._denoise_loop(points)
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        diagonal = math.hypot(width, height)
        travel = sum(cls._point_gap(a, b) for a, b in zip(points, points[1:]))

        # Open lines and unrelated scribbles should never turn into shapes.
        if (travel < diagonal * 1.30
                or travel > diagonal * 8.0):
            return None

        corners = cls._closed_corners(points, max(7.0, diagonal * 0.070))
        count = len(corners)
        if count < 3:
            return None

        area = cls._polygon_area(corners)
        if area < width * height * 0.18:
            return None

        if count == 3:
            # Make the sides of a rough triangle equal, maintaining rotation.
            cx = sum(p[0] for p in corners) / 3
            cy = sum(p[1] for p in corners) / 3
            radius = sum(math.hypot(p[0] - cx, p[1] - cy) for p in corners) / 3
            theta = math.atan2(corners[0][1] - cy, corners[0][0] - cx)
            vertices = [
                (cx + radius * math.cos(theta + 2 * math.pi * i / 3),
                 cy + radius * math.sin(theta + 2 * math.pi * i / 3))
                for i in range(3)
            ]
            return 'triangle', vertices

        if count == 4:
            # Only straighten a quadrilateral if its corners resemble 90 deg.
            for i, p in enumerate(corners):
                prev, nxt = corners[i - 1], corners[(i + 1) % 4]
                u = (prev[0] - p[0], prev[1] - p[1])
                v = (nxt[0] - p[0], nxt[1] - p[1])
                denom = math.hypot(*u) * math.hypot(*v)
                if denom < 1 or abs((u[0] * v[0] + u[1] * v[1]) / denom) > 0.60:
                    return None

            rect_area, theta, left, right, top, bottom = cls._minimum_rectangle(corners)
            if rect_area <= 0 or area / rect_area < 0.50:
                return None
            size_u, size_v = right - left, bottom - top
            is_square = 0.72 <= size_u / max(size_v, 1) <= 1.38
            if is_square:
                side = (size_u + size_v) / 2
                mid_u, mid_v = (left + right) / 2, (top + bottom) / 2
                left, right = mid_u - side / 2, mid_u + side / 2
                top, bottom = mid_v - side / 2, mid_v + side / 2

            cos_a, sin_a = math.cos(theta), math.sin(theta)
            vertices = [
                (u * cos_a - v * sin_a, u * sin_a + v * cos_a)
                for u, v in ((left, top), (right, top),
                             (right, bottom), (left, bottom))
            ]
            return ('square' if is_square else 'rectangle'), vertices

        # A roughly round closed stroke becomes a perfect circle.
        perimeter = sum(cls._point_gap(a, b)
                        for a, b in zip(corners, corners[1:] + corners[:1]))
        circularity = 4 * math.pi * area / max(perimeter * perimeter, 1)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        radii = [math.hypot(p[0] - cx, p[1] - cy) for p in points[:-1]]
        mean_radius = sum(radii) / len(radii)
        radius_variation = (
            math.sqrt(sum((r - mean_radius) ** 2 for r in radii) / len(radii))
            / max(mean_radius, 1)
        )
        # A wobbly square can have many apparent corners, but the varying
        # distance to its center distinguishes it from a round circle.
        if (0.78 <= width / height <= 1.28
                and circularity >= 0.75
                and radius_variation < 0.085):
            radius = (width + height) / 4
            return 'circle', (cx, cy, radius)

        return None

    def _finish_shape(self):
        """Replace a recognized rough shape, keeping older strokes intact."""
        points, before, style = self._shape_points, self._shape_before, self._shape_style
        self._shape_points, self._shape_before, self._shape_style = [], None, None

        if not self.auto_shapes or before is None or style is None:
            return
        match = self._recognize_shape(points)
        if match is None:
            # Give visible feedback when a sizable loop was attempted. Short
            # handwriting strokes and normal lines remain silent.
            if len(points) >= 12:
                width = max(p[0] for p in points) - min(p[0] for p in points)
                height = max(p[1] for p in points) - min(p[1] for p in points)
                diagonal = math.hypot(width, height)
                travel = sum(self._point_gap(a, b)
                             for a, b in zip(points, points[1:]))
                near_start = min(self._point_gap(points[0], p)
                                 for p in points[len(points) // 2:])
                attempted_loop = near_start < max(35, min(140, diagonal * 0.45))
                if (min(width, height) > 35 and travel > diagonal * 1.6
                        and attempted_loop):
                    self.shape_feedback = 'Shape not recognized - try a larger, clearer outline'
            return

        name, geometry = match
        color, brush = style
        self.ink[:] = before[0]
        self.ink_mask[:] = before[1]

        for layer, paint in ((self.ink, color), (self.ink_mask, 255)):
            if name == 'circle':
                cx, cy, radius = geometry
                cv2.circle(layer, (int(round(cx)), int(round(cy))),
                           int(round(radius)), paint, brush, cv2.LINE_AA)
            else:
                for a, b in zip(geometry, geometry[1:] + geometry[:1]):
                    self._stroke_line(layer, a, b, paint, brush)

        self.last_shape = name
        self.shape_feedback = None

    def stroke(self, point, mode):

        # Shape detection uses real fingertip positions. The smoothed pen can
        # trail behind your hand, leaving a visually closed circle looking
        # open to the recognizer when you lift your finger.
        raw_point = (float(point[0]), float(point[1]))

        if mode != self.previous_mode:
            self._finish_stroke()
            self._finish_shape()
            self._filtered_position = None
            self._previous_eraser_radius = None

        if mode in ('draw', 'highlight'):
            point = self.smooth_point(point)

        else:
            self._filtered_position = None
            point = (float(point[0]), float(point[1]))

        start = self.previous if self.previous_mode == mode else None

        # Prevent sudden jumps from creating unwanted lines.
        new_stroke = (
            start is None
            or math.hypot(start[0] - point[0], start[1] - point[1])
            > self.JUMP_LIMIT
        )

        if mode in ('draw', 'highlight'):

            if new_stroke or self._curve is None:

                self._finish_stroke()
                self._finish_shape()

                if mode == 'draw' and self.auto_shapes:
                    # Keep a snapshot of only the ink layers so the rough
                    # stroke can be replaced without erasing earlier work.
                    self._shape_before = (self.ink.copy(), self.ink_mask.copy())
                    self._shape_style = (tuple(self.color), int(self.brush_size))
                    self._shape_points = [raw_point]

                for layer, color, thickness in self._targets(mode):
                    self._stroke_line(layer, point, point, color, thickness)

                self._curve = (point, point)

            else:

                if (mode == 'draw' and self._shape_before is not None
                        and self._pending is None):
                    # Ignore raw one-frame jumps rejected by the spike gate.
                    self._shape_points.append(raw_point)

                midpoint, last = self._curve

                new_midpoint = (
                    (last[0] + point[0]) / 2,
                    (last[1] + point[1]) / 2,
                )

                # Curve through the midpoints, using the raw points as
                # control points. This rounds off corners between frames.
                self._draw_curve(mode, midpoint, last, new_midpoint)

                self._curve = (new_midpoint, point)

        elif mode == 'erase':

            s = point if new_stroke else start

            a = (int(round(s[0])), int(round(s[1])))
            b = (int(round(point[0])), int(round(point[1])))

            radius = max(3, int(self.eraser_radius))
            previous_radius = (
                radius if new_stroke or self._previous_eraser_radius is None
                else self._previous_eraser_radius
            )
            # Follow the reference's continuous eraser idea: bridge the old
            # and new eraser locations even if the radius changes mid-stroke.
            bridge_width = 2 * max(previous_radius, radius)

            for layer in (
                self.ink,
                self.ink_mask,
                self.highlight_mask
            ):

                cv2.line(layer, a, b, 0, bridge_width, cv2.LINE_8)
                cv2.circle(layer, b, radius, 0, -1, cv2.LINE_8)

            self._previous_eraser_radius = radius

        else:
            self.stop()
            return

        self.previous = point
        self.previous_mode = mode

    def clearCanvas(self):

        self._curve = None

        # Clearing must discard the in-progress shape snapshot, otherwise
        # stop() could restore ink that was just cleared.
        self._shape_points = []
        self._shape_before = None
        self._shape_style = None
        self.last_shape = None
        self.shape_feedback = None

        self.ink.fill(0)
        self.ink_mask.fill(0)
        self.highlight_mask.fill(0)

        self.stop()

    def render(self, background=None):
        """Composite ink and highlight over the background (fast uint8 math).

        Always returns a new image, so the caller can draw on it freely.
        """

        if background is None:
            # Ink is already stored pre-blended against black.
            base = self.ink.copy()

        else:
            keep = cv2.cvtColor(
                cv2.bitwise_not(self.ink_mask), cv2.COLOR_GRAY2BGR
            )

            base = cv2.multiply(background, keep, scale=1 / 255)
            base = cv2.add(base, self.ink)

        # Highlight: blend a translucent tint where the highlight mask is set.
        if cv2.countNonZero(self.highlight_mask):

            amount = cv2.convertScaleAbs(self.highlight_mask, alpha=0.35)

            amount3 = cv2.cvtColor(amount, cv2.COLOR_GRAY2BGR)
            keep3 = cv2.cvtColor(cv2.bitwise_not(amount), cv2.COLOR_GRAY2BGR)

            base = cv2.add(
                cv2.multiply(base, keep3, scale=1 / 255),
                cv2.multiply(self._tint, amount3, scale=1 / 255),
            )

        return base

    def saveDrawing(self, background=None):

        folder = Path(__file__).resolve().parent / 'saved_drawings'

        folder.mkdir(exist_ok=True)

        path = folder / (
            datetime.now().strftime('drawing_%Y%m%d_%H%M%S_%f')
            + '.png'
        )

        if not cv2.imwrite(str(path), self.render(background)):
            raise OSError('Could not save drawing')

        return path


