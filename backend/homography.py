"""
Homography computation and player tracking pipeline.

Court coordinate system (NBA half-court, metres):
  Origin (0,0) = top-left baseline corner
  X axis = along baseline (0 → 15.24 m)
  Y axis = towards half-court (0 → 14.63 m)

Standard point labels that the user can click:
  "tl"  = top-left corner  (0,      0     )
  "tr"  = top-right corner (15.24,  0     )
  "bl"  = bottom-left      (0,      14.63 )
  "br"  = bottom-right     (15.24,  14.63 )
  "ftl" = free-throw left  (0,      5.79  )
  "ftr" = free-throw right (15.24,  5.79  )
  "c"   = center circle    (7.62,   14.63 )
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Court geometry constants
# ---------------------------------------------------------------------------

# Metres
COURT_W = 15.24
COURT_H = 14.63  # half-court length

# Render scale: pixels per metre
COURT_SCALE = 40  # 40 px/m → canvas 610 × 585 px (nice TV-friendly size)

CANVAS_W = int(COURT_W * COURT_SCALE)
CANVAS_H = int(COURT_H * COURT_SCALE)

# Named court reference points (metres)
COURT_REFERENCE_POINTS: dict[str, tuple[float, float]] = {
    "tl":  (0.0,    0.0),
    "tr":  (15.24,  0.0),
    "bl":  (0.0,    14.63),
    "br":  (15.24,  14.63),
    "ftl": (0.0,    5.79),
    "ftr": (15.24,  5.79),
    "c":   (7.62,   14.63),
}

# Team colour ranges in HSV (hue only, 0-179 in OpenCV)
# Values are approximate; tweak per use-case
TEAM_A_HUE_RANGE = (100, 135)   # blue-ish
TEAM_B_HUE_RANGE = (0,  15)     # red-ish (also check 165-179)

# Player detection parameters
MIN_PLAYER_AREA = 600
MAX_PLAYER_AREA = 8000
TRAIL_LENGTH = 45   # frames (~1.5 s at 30 fps)

# Dot colours (BGR)
TEAM_A_COLOR = (220, 80,  40)    # blue
TEAM_B_COLOR = (40,  80,  220)   # red
UNKNOWN_COLOR = (180, 180, 180)  # grey


# ---------------------------------------------------------------------------
# Homography computation
# ---------------------------------------------------------------------------

def compute_homography(points: list[dict]) -> np.ndarray:
    """
    Compute the homography matrix from clicked image points to court metres.

    Each point dict must have 'x', 'y' (image pixels) and 'label' (string key
    matching COURT_REFERENCE_POINTS, e.g. 'tl', 'tr', 'bl', 'br').

    Falls back to ordering by position if labels are missing/unknown:
      top-left, top-right, bottom-left, bottom-right.
    """
    src_pts = []
    dst_pts = []

    labelled = [p for p in points if p.get("label") in COURT_REFERENCE_POINTS]
    if len(labelled) >= 4:
        for p in labelled:
            src_pts.append([p["x"], p["y"]])
            mx, my = COURT_REFERENCE_POINTS[p["label"]]
            dst_pts.append([mx * COURT_SCALE, my * COURT_SCALE])
    else:
        # Sort by position: top-left, top-right, bottom-left, bottom-right
        sorted_pts = sorted(points, key=lambda p: (p["y"], p["x"]))
        top = sorted(sorted_pts[:2], key=lambda p: p["x"])
        bottom = sorted(sorted_pts[2:4], key=lambda p: p["x"])
        ordered = [top[0], top[1], bottom[0], bottom[1]]
        corners = [
            COURT_REFERENCE_POINTS["tl"],
            COURT_REFERENCE_POINTS["tr"],
            COURT_REFERENCE_POINTS["bl"],
            COURT_REFERENCE_POINTS["br"],
        ]
        for p, (mx, my) in zip(ordered, corners):
            src_pts.append([p["x"], p["y"]])
            dst_pts.append([mx * COURT_SCALE, my * COURT_SCALE])

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Could not compute homography — points may be collinear")
    return H


# ---------------------------------------------------------------------------
# Court diagram rendering
# ---------------------------------------------------------------------------

def _draw_court(canvas: np.ndarray) -> None:
    """Draw an NBA half-court diagram onto `canvas` (BGR, already filled)."""
    s = COURT_SCALE
    c = (255, 255, 255)  # white lines
    t = 2                # line thickness

    def m(x, y):
        return (int(x * s), int(y * s))

    # Boundary
    cv2.rectangle(canvas, m(0, 0), m(COURT_W, COURT_H), c, t)

    # Paint / key (NBA: 4.88m wide, 5.79m deep)
    paint_w = 4.88
    paint_x = (COURT_W - paint_w) / 2
    cv2.rectangle(canvas, m(paint_x, 0), m(paint_x + paint_w, 5.79), c, t)

    # Free-throw line
    cv2.line(canvas, m(paint_x, 5.79), m(paint_x + paint_w, 5.79), c, t)

    # Free-throw arc (radius 1.83m from FT line center)
    ft_center = (int(COURT_W / 2 * s), int(5.79 * s))
    ft_radius = int(1.83 * s)
    cv2.ellipse(canvas, ft_center, (ft_radius, ft_radius), 0, 0, 180, c, t)

    # Basket (0.46m from baseline, centre at COURT_W/2)
    basket_pt = m(COURT_W / 2, 1.575)
    cv2.circle(canvas, basket_pt, int(0.23 * s), c, t)

    # Restricted area arc (radius 1.22m from basket)
    cv2.ellipse(canvas, basket_pt, (int(1.22 * s), int(1.22 * s)), 0, 0, 180, c, t)

    # 3-point arc (radius 7.24m from basket, cut to sidelines at 0.9m from baseline)
    three_pt_straight_h = 0.9  # metres from baseline to where arc meets sideline
    cv2.line(canvas, m(0, three_pt_straight_h), m(0.9, three_pt_straight_h), c, t)
    cv2.line(canvas, m(COURT_W - 0.9, three_pt_straight_h),
             m(COURT_W, three_pt_straight_h), c, t)
    cv2.ellipse(canvas, basket_pt, (int(7.24 * s), int(7.24 * s)), 0, 4, 176, c, t)

    # Half-court line
    cv2.line(canvas, m(0, COURT_H), m(COURT_W, COURT_H), c, t)

    # Centre circle (radius 1.83m)
    cv2.circle(canvas, m(COURT_W / 2, COURT_H), int(1.83 * s), c, t)


def _make_blank_court() -> np.ndarray:
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = (34, 85, 34)  # dark green floor
    _draw_court(canvas)
    return canvas


BLANK_COURT = _make_blank_court()


# ---------------------------------------------------------------------------
# Player tracker
# ---------------------------------------------------------------------------

class PlayerTracker:
    """
    Simple centroid-based tracker.
    Matches detections frame-to-frame by nearest-neighbour.
    """

    def __init__(self):
        self.next_id = 0
        self.players: dict[int, dict] = {}   # id → {trail, team, last_pos}
        self.max_dist = 80  # px — max movement between frames

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections: list of {cx, cy, team}
        Returns list of {id, cx, cy, team, trail}
        """
        if not self.players:
            for d in detections:
                self._register(d)
            return self._get_active()

        # Build cost matrix
        player_ids = list(self.players.keys())
        if not player_ids or not detections:
            if not detections:
                # Age out all
                for pid in player_ids:
                    self.players[pid]["missing"] = self.players[pid].get("missing", 0) + 1
                self.players = {k: v for k, v in self.players.items() if v.get("missing", 0) < 10}
                return self._get_active()
            for d in detections:
                self._register(d)
            return self._get_active()

        prev_pos = np.array([[self.players[pid]["last_pos"]] for pid in player_ids],
                            dtype=np.float32).squeeze(1)  # (N, 2)
        curr_pos = np.array([[d["cx"], d["cy"]] for d in detections],
                            dtype=np.float32)              # (M, 2)

        # Euclidean distances (N×M)
        dists = np.linalg.norm(prev_pos[:, None] - curr_pos[None, :], axis=2)

        matched_prev = set()
        matched_curr = set()

        # Greedy matching: smallest distance first
        pairs = sorted(
            [(dists[i, j], i, j) for i in range(len(player_ids)) for j in range(len(detections))],
            key=lambda x: x[0]
        )
        for dist, i, j in pairs:
            if i in matched_prev or j in matched_curr:
                continue
            if dist > self.max_dist:
                break
            pid = player_ids[i]
            d = detections[j]
            p = self.players[pid]
            p["last_pos"] = (d["cx"], d["cy"])
            p["trail"].append((d["cx"], d["cy"]))
            if len(p["trail"]) > TRAIL_LENGTH:
                p["trail"].pop(0)
            p["team"] = d["team"]
            p["missing"] = 0
            matched_prev.add(i)
            matched_curr.add(j)

        # Age out unmatched existing players
        for i, pid in enumerate(player_ids):
            if i not in matched_prev:
                self.players[pid]["missing"] = self.players[pid].get("missing", 0) + 1

        # Register new detections
        for j, d in enumerate(detections):
            if j not in matched_curr:
                self._register(d)

        # Remove long-missing players
        self.players = {k: v for k, v in self.players.items() if v.get("missing", 0) < 10}

        return self._get_active()

    def _register(self, d: dict):
        self.players[self.next_id] = {
            "last_pos": (d["cx"], d["cy"]),
            "trail": [(d["cx"], d["cy"])],
            "team": d["team"],
            "missing": 0,
        }
        self.next_id += 1

    def _get_active(self):
        out = []
        for pid, p in self.players.items():
            if p.get("missing", 0) == 0:
                cx, cy = p["last_pos"]
                out.append({
                    "id": pid,
                    "cx": cx,
                    "cy": cy,
                    "team": p["team"],
                    "trail": list(p["trail"]),
                })
        return out


# ---------------------------------------------------------------------------
# Team classification
# ---------------------------------------------------------------------------

def _classify_team(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
    """Classify a player's team based on the dominant jersey hue."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop_h = y2 - y1
    # Upper 60% = jersey area
    jersey = frame[y1: y1 + int(crop_h * 0.6), x1:x2]
    if jersey.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
    hue_vals = hsv[:, :, 0].flatten().astype(np.float32)
    sat_vals = hsv[:, :, 1].flatten().astype(np.float32)

    # Only consider saturated pixels (ignore white/black/grey)
    saturated = hue_vals[sat_vals > 60]
    if len(saturated) < 10:
        return "unknown"

    mean_hue = float(np.mean(saturated))

    lo_a, hi_a = TEAM_A_HUE_RANGE
    lo_b, hi_b = TEAM_B_HUE_RANGE

    score_a = np.sum((saturated >= lo_a) & (saturated <= hi_a))
    score_b = np.sum(((saturated >= lo_b) & (saturated <= hi_b)) |
                     (saturated >= 165))   # wrap-around red

    if score_a > score_b and score_a > len(saturated) * 0.2:
        return "A"
    if score_b > score_a and score_b > len(saturated) * 0.2:
        return "B"
    return "unknown"


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _render_court_frame(players: list[dict]) -> np.ndarray:
    """Render one frame of the 2D court with player dots and trails."""
    canvas = BLANK_COURT.copy()

    for player in players:
        trail = player["trail"]
        team = player["team"]

        if team == "A":
            color = TEAM_A_COLOR
        elif team == "B":
            color = TEAM_B_COLOR
        else:
            color = UNKNOWN_COLOR

        # Draw fading trail
        n = len(trail)
        for k in range(1, n):
            alpha = k / n
            pt1 = (int(trail[k - 1][0]), int(trail[k - 1][1]))
            pt2 = (int(trail[k][0]), int(trail[k][1]))
            # Blend colour with background based on alpha
            line_color = tuple(int(c * alpha) for c in color)
            cv2.line(canvas, pt1, pt2, line_color, 2, cv2.LINE_AA)

        # Draw dot at current position
        cx, cy = int(player["cx"]), int(player["cy"])
        cv2.circle(canvas, (cx, cy), 8, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_video_with_homography(
    clip_path: str,
    homography_matrix: np.ndarray,
    output_dir: str,
    job_id: str,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> dict:
    """
    Process the clipped video:
    1. Detect players per frame via background subtraction
    2. Project positions to 2D court via homography
    3. Render court animation frames
    4. Write court animation video

    Returns dict with 'court_anim_path'.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    court_anim_path = str(Path(output_dir) / f"court_{job_id}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(court_anim_path, fourcc, fps, (CANVAS_W, CANVAS_H))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=40, detectShadows=False
    )

    tracker = PlayerTracker()
    frame_idx = 0

    # Warm up background model on first few frames without writing output
    warmup = min(30, total_frames // 5)
    for _ in range(warmup):
        ret, frame = cap.read()
        if not ret:
            break
        bg_subtractor.apply(frame)
        frame_idx += 1

    # Reset to beginning for actual processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_PLAYER_AREA <= area <= MAX_PLAYER_AREA):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Foot point = bottom-centre of bounding box
            foot_x = x + w / 2
            foot_y = float(y + h)
            img_pt = np.array([[[foot_x, foot_y]]], dtype=np.float32)
            court_pt = cv2.perspectiveTransform(img_pt, homography_matrix)[0][0]

            cx, cy = float(court_pt[0]), float(court_pt[1])
            # Clip to canvas bounds
            if not (0 <= cx < CANVAS_W and 0 <= cy < CANVAS_H):
                continue

            team = _classify_team(frame, x, y, x + w, y + h)
            detections.append({"cx": cx, "cy": cy, "team": team})

        tracked = tracker.update(detections)
        court_frame = _render_court_frame(tracked)
        out.write(court_frame)

        frame_idx += 1
        if progress_callback and total_frames > 0:
            pct = int(frame_idx / total_frames * 90)   # 90% for tracking, 10% for combine
            progress_callback(pct)

    cap.release()
    out.release()

    # Re-encode with ffmpeg for browser compatibility
    import subprocess
    compat_path = str(Path(output_dir) / f"court_compat_{job_id}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", court_anim_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        compat_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        os.replace(compat_path, court_anim_path)

    return {"court_anim_path": court_anim_path}
