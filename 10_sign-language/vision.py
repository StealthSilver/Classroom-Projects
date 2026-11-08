#!/usr/bin/env python3
"""संकेतवाणी — American Sign Language fingerspelling, in the browser.

Run:
    python3 vision.py

Then open the link it prints. The first launch installs the camera libraries
and downloads a hand-tracking model. index.html is the page; this file is the
detector.

Check the letter rules without a camera:
    python3 vision.py --self-test
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache"))
INDEX_PATH = ROOT / "index.html"
MODEL_PATH = ROOT / "hand_landmarker.task"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
MODEL_URLS = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
)

# A finger counts as extended when the joint at the middle knuckle is this
# straight, in degrees. 180 is a straight line.
EXTENDED_ANGLE = 150
# Palm area below this means the palm is turned sideways (G, H, P, Q).
EDGE_PALM = 0.40
# Index finger aimed this far from upright is pointing downward.
DOWN_ANGLE = 145

# MediaPipe hand connections, drawn by the page.
# 0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky.


class Feat(NamedTuple):
    ext: tuple  # index, middle, ring, pinky
    angles: tuple
    pinch: float
    palm_area: float
    thumb_tip: tuple
    index_tip: tuple
    middle_tip: tuple
    index_mcp: tuple
    middle_mcp: tuple
    ring_mcp: tuple
    index_dip: tuple
    spread: float
    crossed: bool
    hooked: bool
    index_angle_img: float
    mean_angle: float
    thumb_out: bool
    tip_gap: float


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _norm(v):
    length = math.hypot(v[0], v[1])
    if length < 1e-8:
        return (1.0, 0.0)
    return (v[0] / length, v[1] / length)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a, b, c):
    """Interior angle at point b, in degrees."""
    v1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) or 1e-8
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2) or 1e-8
    cos = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def as_points(landmarks):
    points = []
    for landmark in landmarks:
        if isinstance(landmark, (tuple, list)):
            x, y = float(landmark[0]), float(landmark[1])
            z = float(landmark[2]) if len(landmark) > 2 else 0.0
        else:
            x, y, z = float(landmark.x), float(landmark.y), float(landmark.z)
        points.append((x, y, z))
    return points


def extract(points):
    """Describe a 21-point hand in a palm-sized coordinate frame.

    Local +y runs from the wrist toward the middle knuckle.
    Local +x runs from the pinky side toward the index side.
    Distances are in units of the palm (wrist to middle knuckle).
    """
    if len(points) != 21:
        return None
    wrist = points[0]
    index_mcp = points[5]
    middle_mcp = points[9]
    ring_mcp = points[13]
    pinky_mcp = points[17]
    scale = math.sqrt(
        (middle_mcp[0] - wrist[0]) ** 2
        + (middle_mcp[1] - wrist[1]) ** 2
        + (middle_mcp[2] - wrist[2]) ** 2
    )
    if scale < 1e-4:
        return None

    up = _norm((middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]))
    across = (index_mcp[0] - pinky_mcp[0], index_mcp[1] - pinky_mcp[1])
    along = _dot(across, up)
    across = _norm((across[0] - along * up[0], across[1] - along * up[1]))

    def loc(point):
        dx, dy = point[0] - wrist[0], point[1] - wrist[1]
        return (
            _dot((dx, dy), across) / scale,
            _dot((dx, dy), up) / scale,
            point[2] / scale,
        )

    local = [loc(point) for point in points]
    chains = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    angles = tuple(_angle(local[mcp], local[pip], local[dip]) for mcp, pip, dip, _tip in chains)
    ext = tuple(angle >= EXTENDED_ANGLE for angle in angles)
    ax, ay = index_mcp[0] - wrist[0], index_mcp[1] - wrist[1]
    bx, by = pinky_mcp[0] - wrist[0], pinky_mcp[1] - wrist[1]
    palm_area = abs(ax * by - ay * bx) / (scale * scale)
    index_tip, middle_tip = local[8], local[12]
    index_dip = local[7]
    thumb_tip = local[4]
    fingertips = (local[8], local[12], local[16], local[20])
    tip_gap = sum(_dist(thumb_tip, tip) for tip in fingertips) / 4
    dx, dy = points[8][0] - points[5][0], points[8][1] - points[5][1]
    index_angle_img = math.degrees(math.atan2(dx, -dy)) if (dx or dy) else 0.0
    hooked = (
        112 <= angles[0] <= 152
        and index_tip[1] + 0.05 < index_dip[1]
        and angles[1] < 110
        and angles[2] < 110
        and angles[3] < 110
    )
    return Feat(
        ext=ext,
        angles=angles,
        pinch=_dist(thumb_tip, index_tip),
        palm_area=palm_area,
        thumb_tip=thumb_tip,
        index_tip=index_tip,
        middle_tip=middle_tip,
        index_mcp=local[5],
        middle_mcp=local[9],
        ring_mcp=local[13],
        index_dip=index_dip,
        spread=abs(index_tip[0] - middle_tip[0]),
        crossed=index_tip[0] < middle_tip[0] - 0.02 and local[5][0] > local[9][0] + 0.02,
        hooked=hooked,
        index_angle_img=index_angle_img,
        mean_angle=sum(angles) / 4.0,
        thumb_out=thumb_tip[0] > local[5][0] + 0.08,
        tip_gap=tip_gap,
    )


def _closed_letter(feat: Feat):
    """A, E, M, N, O, S, T, and C — the shapes with every finger folded."""
    tx, ty, _tz = feat.thumb_tip
    # C is the open curve. O and E bring the thumb against the fingertips;
    # E is the tighter fist, O stays rounder.
    if feat.mean_angle >= 132:
        return "C", 0.84
    if feat.tip_gap <= 0.22 and feat.mean_angle >= 108:
        return "O", 0.86
    if feat.tip_gap <= 0.22:
        return "E", 0.84
    if tx > feat.index_mcp[0] + 0.12 and ty < 1.25:
        return "A", 0.88
    if ty > 1.05 and tx < feat.index_mcp[0]:
        return "S", 0.82
    if tx < feat.ring_mcp[0] + 0.08:
        return "M", 0.78
    if tx < feat.middle_mcp[0] + 0.08:
        return "N", 0.76
    return "T", 0.76


def _two_fingers(feat: Feat):
    """H, K, P, R, U, and V — index and middle up, the others folded."""
    edge = feat.palm_area < EDGE_PALM
    down = abs(feat.index_angle_img) >= DOWN_ANGLE
    if edge and down:
        return "P", 0.82
    if edge:
        return "H", 0.86
    if feat.crossed:
        return "R", 0.88
    tx, ty, _tz = feat.thumb_tip
    lo = min(feat.index_tip[0], feat.middle_tip[0]) - 0.1
    hi = max(feat.index_tip[0], feat.middle_tip[0]) + 0.1
    if ty > 1.05 and lo <= tx <= hi:
        return "K", 0.86
    if feat.spread >= 0.24:
        return "V", 0.9
    return "U", 0.88


def _index_only(feat: Feat):
    """D, G, L, Q, and X — only the index finger is doing the work."""
    edge = feat.palm_area < EDGE_PALM
    down = abs(feat.index_angle_img) >= DOWN_ANGLE
    if feat.hooked:
        return "X", 0.84
    if edge and down:
        return "Q", 0.82
    if edge:
        return "G", 0.84
    if feat.thumb_out:
        return "L", 0.9
    return "D", 0.88


def _unclear(feat: Feat) -> bool:
    """True when a finger is neither straight nor folded, so the pose is not a letter yet.

    C is the exception: every finger shares the same open curve, with a gap at the thumb.
    """
    mid = [angle for angle in feat.angles if 130 <= angle < 158]
    if not mid:
        return False
    if len(mid) == 4 and max(feat.angles) - min(feat.angles) <= 18 and feat.pinch >= 0.45 and feat.palm_area >= EDGE_PALM:
        return False
    return True


def classify(feat: Feat):
    """Map a hand description to one ASL fingerspelled letter."""
    if feat.hooked:
        return "X", 0.84
    if _unclear(feat):
        return "", 0.0
    index_up, middle_up, ring_up, pinky_up = feat.ext
    if pinky_up and not index_up and not middle_up and not ring_up:
        if feat.thumb_tip[0] > 0.70:
            return "Y", 0.9
        return "I", 0.9
    if feat.pinch < 0.32 and middle_up and ring_up and pinky_up and not index_up:
        return "F", 0.9
    if index_up and middle_up and ring_up and pinky_up:
        if feat.thumb_tip[0] < feat.index_mcp[0] + 0.05:
            return "B", 0.9
        return "", 0.0
    if index_up and middle_up and ring_up and not pinky_up:
        return "W", 0.88
    if index_up and middle_up and not ring_up and not pinky_up:
        return _two_fingers(feat)
    if index_up and not middle_up and not ring_up and not pinky_up:
        return _index_only(feat)
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return _closed_letter(feat)
    return "", 0.0


def _path_length(points):
    return sum(_dist(a, b) for a, b in zip(points, points[1:]))


def _reversals(values):
    signs = []
    for left, right in zip(values, values[1:]):
        delta = right - left
        if abs(delta) < 0.008:
            continue
        sign = 1 if delta > 0 else -1
        if not signs or signs[-1] != sign:
            signs.append(sign)
    return max(0, len(signs) - 1)


class SignReader:
    """Turns landmarks into a letter, including the moving letters J and Z."""

    def __init__(self):
        self.history = deque(maxlen=24)

    def miss(self):
        self.history.clear()

    def read(self, landmarks):
        points = as_points(landmarks)
        blank = {
            "letter": "",
            "confidence": 0.0,
            "fingers": [False, False, False, False, False],
            "landmarks": [[round(p[0], 4), round(p[1], 4)] for p in points],
        }
        feat = extract(points)
        if feat is None:
            return blank
        letter, confidence = classify(feat)
        letter, confidence = self._apply_motion(letter, confidence, points)
        return {
            "letter": letter,
            "confidence": round(confidence, 2),
            "fingers": [feat.thumb_out, *feat.ext],
            "landmarks": blank["landmarks"],
        }

    def _apply_motion(self, letter, confidence, points):
        now = time.monotonic()
        if self.history and now - self.history[-1][0] > 0.75:
            self.history.clear()
        self.history.append(
            (now, (points[8][0], points[8][1]), (points[20][0], points[20][1]), letter)
        )
        if letter == "I" and self._draws_j():
            return "J", min(confidence, 0.84)
        if letter in {"D", "G", "X"} and self._draws_z():
            return "Z", min(confidence, 0.8)
        return letter, confidence

    def _trail(self, letter, slot):
        return [item[slot] for item in self.history if item[3] == letter]

    def _draws_j(self):
        trail = self._trail("I", 2)
        if len(trail) < 8:
            return False
        length = _path_length(trail)
        net = _dist(trail[0], trail[-1])
        if length < 0.18 or net < 0.05:
            return False
        curved = length / net >= 1.25
        moved_down = trail[-1][1] - trail[0][1] > 0.05
        return curved and moved_down

    def _draws_z(self):
        trail = []
        for item in self.history:
            if item[3] in {"D", "G", "X"}:
                trail.append(item[1])
        if len(trail) < 8:
            return False
        return _path_length(trail) >= 0.18 and _reversals([point[0] for point in trail]) >= 2


class HandTracker:
    def __init__(self, model_path: Path):
        import mediapipe as mp

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._mp = mp
        self._reader = SignReader()
        self._lock = threading.Lock()

    def detect_jpeg(self, payload: bytes) -> dict:
        import cv2
        import numpy as np

        array = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not read that camera frame.")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        if width > 640:
            resized_height = max(1, int(height * (640 / width)))
            rgb = cv2.resize(rgb, (640, resized_height), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._landmarker.detect(mp_image)
            if not result.hand_landmarks:
                self._reader.miss()
                return {
                    "ok": True,
                    "hand": False,
                    "letter": "",
                    "confidence": 0,
                    "fingers": [False, False, False, False, False],
                    "landmarks": [],
                }
            reading = self._reader.read(result.hand_landmarks[0])
        reading["ok"] = True
        reading["hand"] = True
        return reading


TRACKER: HandTracker | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            if not INDEX_PATH.exists():
                self._send(500, b"index.html is missing beside vision.py.", "text/plain; charset=utf-8")
                return
            self._send(200, INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/health":
            ready = TRACKER is not None
            self._send_json(200 if ready else 503, {"ok": ready})
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self._send(404, b"Not found.", "text/plain; charset=utf-8")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/detect":
            self._send(404, b"Not found.", "text/plain; charset=utf-8")
            return
        if TRACKER is None:
            self._send_json(503, {"ok": False, "error": "The detector is still starting."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 4_000_000:
            self._send_json(400, {"ok": False, "error": "Expected a camera frame."})
            return
        payload = self.rfile.read(length)
        try:
            result = TRACKER.detect_jpeg(payload)
        except Exception as exc:  # noqa: BLE001 - report one frame, keep serving
            traceback.print_exc()
            self._send_json(400, {"ok": False, "error": str(exc)})
            return
        self._send_json(200, result)

    def _send_json(self, code, payload):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _send(self, code, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, fmt, *args):
        if len(args) >= 2 and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("%s\n" % (fmt % args))


def _imports_ok() -> bool:
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        return False
    return True


def ensure_runtime():
    """Install MediaPipe and OpenCV into a local environment on first launch."""
    if _imports_ok():
        return
    # Compare prefixes. Resolving the binary is wrong here: a virtualenv's
    # python is a symlink to the same interpreter that created it.
    try:
        in_venv = Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    except OSError:
        in_venv = False
    if in_venv or os.environ.get("SIGNHAND_BOOTSTRAPPED") == "1":
        sys.exit(
            "The camera libraries did not import.\n"
            "Install them, then run this again:\n"
            "  python3 -m pip install mediapipe opencv-python-headless numpy"
        )
    print("First launch: setting up the camera libraries. This can take a minute.", flush=True)
    if not VENV_PYTHON.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(ROOT / ".venv")])
    subprocess.check_call(
        [str(VENV_PYTHON), "-m", "pip", "install", "-q", "mediapipe", "opencv-python-headless", "numpy"]
    )
    env = os.environ.copy()
    env["SIGNHAND_BOOTSTRAPPED"] = "1"
    script = str(Path(__file__).resolve())
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), script, *sys.argv[1:]], env)


def ensure_model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return MODEL_PATH
    print("Downloading the hand-tracking model...", flush=True)
    temporary = MODEL_PATH.with_suffix(".task.part")
    last_error: Exception | None = None
    for url in MODEL_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "letterhand/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if temporary.stat().st_size < 1_000_000:
                raise RuntimeError("The downloaded model file was incomplete.")
            temporary.replace(MODEL_PATH)
            return MODEL_PATH
        except Exception as exc:  # noqa: BLE001 - try the next official URL
            last_error = exc
            temporary.unlink(missing_ok=True)
    sys.exit(
        "Could not download the hand-tracking model.\n"
        f"{last_error}\n"
        "Check the internet connection and run python3 vision.py again."
    )


def serve(model_path: Path):
    global TRACKER
    print("Loading the hand tracker...", flush=True)
    TRACKER = HandTracker(model_path)
    server = None
    port = 8000
    for candidate in range(8000, 8020):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        sys.exit("No free local port was found between 8000 and 8019.")
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}"
    print(flush=True)
    print("संकेतवाणी is running.", flush=True)
    print(f"Open this page in your browser:  {url}", flush=True)
    print("Allow the camera, then hold one ASL letter in front of it.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    print(flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


# --- Self-test poses -------------------------------------------------------
# Landmarks are built in the same palm coordinate frame the detector uses,
# then placed in an image so the rules can be checked without a camera.

MCP = {
    "index": (0.42, 0.94, 0.0),
    "middle": (0.00, 1.00, 0.0),
    "ring": (-0.30, 0.94, 0.0),
    "pinky": (-0.54, 0.94, 0.0),
}
FINGER_ORDER = ("index", "middle", "ring", "pinky")


def _finger_chain(mcp, curl, spread_sign, lengths):
    angle = math.radians(90 + spread_sign * 8)
    points = []
    position = mcp
    # Curl toward the pinky side so the fingertips come back across the palm
    # and a thumb held out to the side stays clear of them.
    for length, full_bend in zip(lengths, (75, 95, 70)):
        angle += math.radians(curl * full_bend)
        position = (
            position[0] + math.cos(angle) * length,
            position[1] + math.sin(angle) * length,
            0.0,
        )
        points.append(position)
    return points


def _cluster(local):
    """Draw the four fingertips into one point, as in E and O."""
    tips = (8, 12, 16, 20)
    center_x = sum(local[index][0] for index in tips) / 4
    center_y = sum(local[index][1] for index in tips) / 4
    for index, weight in ((7, 0.35), (11, 0.35), (15, 0.35), (19, 0.35), (8, 0.88), (12, 0.88), (16, 0.88), (20, 0.88)):
        x, y, z = local[index]
        local[index] = (x + (center_x - x) * weight, y + (center_y - y) * weight, z)


def _adduct(local, strength):
    weights = (0.2, 0.55, 0.92)
    for (left, right), weight in zip(((6, 10), (7, 11), (8, 12)), weights):
        amount = weight * strength
        ax, ay, az = local[left]
        bx, by, bz = local[right]
        mid = (ax + bx) / 2
        local[left] = (ax + (mid - ax) * amount, ay, az)
        local[right] = (bx + (mid - bx) * amount, by, bz)


def build_landmarks(spec) -> list:
    local = [(0.0, 0.0, 0.0)] * 21
    local[0] = (0.0, 0.0, 0.0)
    curls = spec.get("curls", (0, 0, 0, 0))
    spread = spec.get("spread", 1.0)
    signs = (1.0, 0.15, -0.7, -1.05)
    lengths = ((0.40, 0.26, 0.22), (0.44, 0.28, 0.24), (0.40, 0.26, 0.22), (0.30, 0.20, 0.16))
    for name, curl, sign, finger_lengths, mcp_index in zip(
        FINGER_ORDER, curls, signs, lengths, (5, 9, 13, 17)
    ):
        mcp = MCP[name]
        local[mcp_index] = mcp
        pip, dip, tip = _finger_chain(mcp, curl, sign * spread, finger_lengths)
        local[mcp_index + 1] = pip
        local[mcp_index + 2] = dip
        local[mcp_index + 3] = tip
    if spec.get("together"):
        _adduct(local, 1.0)
    if spec.get("cluster"):
        _cluster(local)
    if spec.get("cross"):
        index_tip, middle_tip = local[8], local[12]
        local[8] = (middle_tip[0] - 0.1, index_tip[1], 0.0)
        local[12] = (index_tip[0] + 0.1, middle_tip[1], 0.0)
    if spec.get("hook"):
        dip = local[7]
        local[8] = (dip[0] + 0.02, dip[1] - 0.2, 0.0)
    thumb_tip = spec.get("thumb", (0.2, 0.5, 0.0))
    if spec.get("pinch") or spec.get("on_tips"):
        tip = local[8]
        thumb_tip = (tip[0] + 0.03, tip[1] - 0.02, 0.0)
    thumb_mcp = (0.38, 0.40, 0.0)
    local[1] = (0.16, 0.22, 0.0)
    local[2] = thumb_mcp
    local[3] = ((thumb_mcp[0] + thumb_tip[0]) / 2, (thumb_mcp[1] + thumb_tip[1]) / 2, 0.0)
    local[4] = thumb_tip

    angle = math.radians(spec.get("angle", 0.0))
    finger_axis = (math.sin(angle), -math.cos(angle))
    thumb_axis = (-finger_axis[1], finger_axis[0])
    flatten = spec.get("flatten", 1.0)
    scale = spec.get("scale", 0.18)
    wrist = spec.get("wrist", (0.50, 0.78))
    image = []
    for x, y, z in local:
        ix = wrist[0] + (x * flatten * thumb_axis[0] + y * finger_axis[0]) * scale
        iy = wrist[1] + (x * flatten * thumb_axis[1] + y * finger_axis[1]) * scale
        image.append((ix, iy, z))
    if spec.get("mirror"):
        image = [(1.0 - x, y, z) for x, y, z in image]
    return image


SELF_TEST_CASES = [
    ("A", dict(curls=(1, 1, 1, 1), thumb=(0.86, 0.78, 0)), "A"),
    ("B", dict(curls=(0, 0, 0, 0), thumb=(0.08, 0.82, 0)), "B"),
    ("C", dict(curls=(0.40, 0.40, 0.40, 0.40), thumb=(1.05, 1.35, 0)), "C"),
    ("D", dict(curls=(0, 1, 1, 1), thumb=(0.10, 1.15, 0)), "D"),
    ("E", dict(curls=(1, 1, 1, 1), on_tips=True, cluster=True), "E"),
    ("F", dict(curls=(0.9, 0, 0, 0), pinch=True), "F"),
    ("G", dict(curls=(0, 1, 1, 1), thumb=(0.36, 1.45, 0), flatten=0.22, angle=80), "G"),
    ("H", dict(curls=(0, 0, 1, 1), thumb=(0.2, 0.6, 0), flatten=0.22, angle=80, together=True), "H"),
    ("I", dict(curls=(1, 1, 1, 0), thumb=(0.32, 0.55, 0)), "I"),
    ("K", dict(curls=(0, 0, 1, 1), thumb=(0.16, 1.32, 0)), "K"),
    ("L", dict(curls=(0, 1, 1, 1), thumb=(1.15, 0.48, 0)), "L"),
    ("M", dict(curls=(1, 1, 1, 1), thumb=(-0.42, 0.58, 0)), "M"),
    ("N", dict(curls=(1, 1, 1, 1), thumb=(-0.08, 0.58, 0)), "N"),
    ("O", dict(curls=(0.58, 0.58, 0.58, 0.58), pinch=True, cluster=True), "O"),
    ("P", dict(curls=(0, 0, 1, 1), thumb=(0.16, 1.2, 0), flatten=0.28, angle=180), "P"),
    ("Q", dict(curls=(0, 1, 1, 1), thumb=(0.34, 1.35, 0), flatten=0.22, angle=180), "Q"),
    ("R", dict(curls=(0, 0, 1, 1), thumb=(0.1, 0.5, 0), together=True, cross=True), "R"),
    ("S", dict(curls=(1, 1, 1, 1), thumb=(0.12, 1.28, 0)), "S"),
    ("T", dict(curls=(1, 1, 1, 1), thumb=(0.22, 0.68, 0)), "T"),
    ("U", dict(curls=(0, 0, 1, 1), thumb=(0.1, 0.48, 0), together=True), "U"),
    ("V", dict(curls=(0, 0, 1, 1), thumb=(0.02, 0.42, 0)), "V"),
    ("W", dict(curls=(0, 0, 0, 1), thumb=(0.08, 0.7, 0)), "W"),
    ("X", dict(curls=(0.45, 1, 1, 1), thumb=(0.45, 0.6, 0), hook=True), "X"),
    ("Y", dict(curls=(1, 1, 1, 0), thumb=(1.02, 0.46, 0)), "Y"),
    ("open", dict(curls=(0, 0, 0, 0), thumb=(1.05, 0.7, 0)), ""),
    ("L-tilt", dict(curls=(0, 1, 1, 1), thumb=(1.15, 0.48, 0), angle=24), "L"),
    ("B-tilt", dict(curls=(0, 0, 0, 0), thumb=(0.08, 0.82, 0), angle=-18), "B"),
    ("A-mirror", dict(curls=(1, 1, 1, 1), thumb=(0.86, 0.78, 0), mirror=True), "A"),
    ("R-mirror", dict(curls=(0, 0, 1, 1), thumb=(0.1, 0.5, 0), together=True, cross=True, mirror=True), "R"),
]


def _shift(landmarks, dx, dy):
    return [(x + dx, y + dy, z) for x, y, z in landmarks]


def _check_motion():
    failures = []
    still = SignReader()
    pose_i = build_landmarks(dict(curls=(1, 1, 1, 0), thumb=(0.32, 0.55, 0)))
    still_letters = [still.read(pose_i)["letter"] for _ in range(10)]
    if still_letters[-1] != "I":
        failures.append(f"a still pinky should stay I, got {still_letters[-1]}")

    moving = SignReader()
    path = [(0.0, 0.035 * i) for i in range(7)]
    path += [(-0.04 * i, 0.21 - 0.01 * i) for i in range(1, 5)]
    seen = [moving.read(_shift(pose_i, dx, dy))["letter"] for dx, dy in path]
    if seen[-1] != "J":
        failures.append(f"a pinky hook should read J, got {seen}")

    drawing = SignReader()
    pose_d = build_landmarks(dict(curls=(0, 1, 1, 1), thumb=(0.10, 1.15, 0)))
    z_path = [(0.03 * i, 0.0) for i in range(5)]
    z_path += [(0.12 - 0.03 * i, 0.025 * i) for i in range(1, 5)]
    z_path += [(0.03 * i, 0.10) for i in range(1, 5)]
    seen_z = [drawing.read(_shift(pose_d, dx, dy))["letter"] for dx, dy in z_path]
    if seen_z[-1] != "Z":
        failures.append(f"an index zigzag should read Z, got {seen_z}")
    return failures


def run_self_test() -> int:
    failures = 0
    for name, spec, expected in SELF_TEST_CASES:
        landmarks = build_landmarks(spec)
        feat = extract(landmarks)
        if feat is None:
            print(f"FAIL {name:<8} landmarks could not be read")
            failures += 1
            continue
        letter, confidence = classify(feat)
        ok = letter == expected and (expected == "" or confidence >= 0.7)
        mark = "ok  " if ok else "FAIL"
        shown = letter or "-"
        fingers = "".join(str(int(flag)) for flag in feat.ext)
        print(
            f"{mark} {name:<8} {shown:<1}  palm={feat.palm_area:.2f} pinch={feat.pinch:.2f} "
            f"ext={fingers} thumb=({feat.thumb_tip[0]:.2f},{feat.thumb_tip[1]:.2f}) "
            f"spread={feat.spread:.2f} aim={feat.index_angle_img:.0f} mean={feat.mean_angle:.0f} "
            f"gap={feat.tip_gap:.2f} hook={int(feat.hooked)} cross={int(feat.crossed)}"
        )
        if not ok:
            failures += 1
            print(f"     expected {expected or '-'} with confidence >= 0.7, got {shown} ({confidence:.2f})")
    for message in _check_motion():
        print(f"FAIL {message}")
        failures += 1
    if failures:
        print(f"\n{failures} self-test check(s) failed.")
        return 1
    print("\nSelf-test passed. Static letters, J, and Z match the hand shapes.")
    return 0


def main(argv) -> int:
    if "--self-test" in argv:
        return run_self_test()
    ensure_runtime()
    model_path = ensure_model()
    serve(model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
