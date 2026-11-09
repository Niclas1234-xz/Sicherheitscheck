"""
Erkennungslogik für den Sicherheitscheck (ohne Benutzeroberfläche).

- Gesichtserkennung mit MediaPipe Face Landmarker (bis zu 5 Gesichter, 478 Punkte inkl. Iris)
- Kopfhaltung über solvePnP (OpenCV) mit einem generischen 3D-Kopfmodell
- Blickrichtung über die Lage der Iris zwischen den Augenwinkeln
- Zustandslogik: Wann erscheint eine Warnung, wann wird sie wieder "scharf" geschaltet?
"""

from __future__ import annotations

import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# Einstellungen
# --------------------------------------------------------------------------- #

PROCESS_WIDTH = 1280         # Analysebreite (volle 720p-Auflösung: schärfere Augen, auch weiter weg)
PREVIEW_WIDTH = 640          # Breite der Live-Vorschau im Fenster
MAX_FPS = 15                # maximale Analysen pro Sekunde (schont die CPU)
MAX_FACES = 5                # maximal gleichzeitig erkannte Gesichter
CAMERA_HFOV_DEG = 70.0       # typischer horizontaler Bildwinkel einer Webcam (für Entfernung und Winkel)

CONFIRM_PRESENT_S = 0.5      # so lange muss eine zweite Person sichtbar sein, bevor gewarnt wird
GAP_TOLERANCE_S = 0.35       # kurze Erkennungsaussetzer werden toleriert
CONFIRM_ABSENT_S = 2.0       # so lange ohne zweite Person => gilt als "verschwunden"

# Blick auf den Bildschirm: Abweichung der Blickrichtung von der Richtung "zur Kamera".
# Der Bildschirm liegt direkt unter der Webcam, darum ist nach unten mehr Spielraum erlaubt.
LOOK_YAW_TOL_DEG = 25.0      # links/rechts, wenn die Augen gut sichtbar sind
LOOK_YAW_TOL_HEAD_DEG = 32.0 # links/rechts, wenn nur die Kopfhaltung bekannt ist (Person weiter weg):
                             # die Augen können den Kopf um einige Grad «korrigieren»
LOOK_UP_TOL_DEG = 15.0       # über die Kamera hinaus nach oben
LOOK_DOWN_TOL_DEG = 45.0     # unter die Kamera (= auf den Bildschirm)

# Strenger Modus für Personen weiter weg: Nur wer wirklich auf den Laptop starrt, löst die Warnung aus.
# Bis STRICT_FROM_M bleibt alles wie oben, bis STRICT_FULL_M wird gleitend auf die strengen Werte umgestellt.
STRICT_FROM_M = 1.3          # ab dieser Entfernung beginnt der strenge Modus
STRICT_FULL_M = 2.2          # ab hier gelten die strengen Werte voll
FAR_YAW_TOL_DEG = 15.0       # links/rechts (statt 25–32°)
FAR_UP_TOL_DEG = 8.0         # nach oben (statt 15°)
FAR_DOWN_TOL_DEG = 30.0      # nach unten (statt 45°)
FAR_DWELL_S = 1.5            # so lange muss der Blick ununterbrochen auf dem Bildschirm bleiben

# Augenmodell: Der Blick wird aus der Lage der Iris relativ zum (aus der Kopfhaltung berechneten)
# Augapfel bestimmt. So zählt, wohin die Augen schauen, auch wenn der Kopf weggedreht ist.
EYE_RADIUS_MM = 11.5         # wirksamer Augapfelradius bis zur Iris
OUTER_EYE_DIST_MM = 92.0     # typischer Abstand der äusseren Augenwinkel (Massstab)
EYE_VERTICAL_WEIGHT = 0.6    # vertikale Irislage ist wegen der Lider unsicherer -> schwächer gewichten
MAX_HEAD_YAW_FOR_EYES = 75.0 # bei stärker weggedrehtem Kopf sind die Augen nicht mehr messbar

# Mindestgrössen (bezogen auf ein 640 px breites Bild, werden automatisch skaliert)
MIN_EYE_PX = 7.0             # kleinere Augen: nur Kopfhaltung verwenden
EYE_FULL_FACTOR = 2.0        # ab MIN_EYE_PX * Faktor zählen die Augen voll, dazwischen gleitender Übergang
MIN_FACE_PX = 14.0           # kleinere Gesichter: Blickrichtung wird nicht geschätzt

# Fern-Erkennung: OpenCV YuNet findet auch kleine Gesichter weit hinten im Raum. Gefundene Gesichter
# werden ausgeschnitten, vergrössert und mit MediaPipe genau vermessen.
FAR_DETECTION = True
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = [
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx",
]
YUNET_WIDTH = 1280           # Analysebreite der Fern-Erkennung
YUNET_SCORE = 0.75           # Mindest-Sicherheit für ein Gesicht
YUNET_SCORE_POINTS_ONLY = 0.88  # strengere Schwelle, wenn nur die 5 Grobpunkte verfügbar sind
YUNET_EVERY_N = 3            # nur jedes n-te Bild komplett absuchen (spart Energie, ~5x pro Sekunde)
MAX_FAR_FACES = 3            # höchstens so viele ferne Gesichter pro Bild genau vermessen
FAR_CROP_FACTOR = 2.2        # Ausschnitt = Gesichtsgrösse x Faktor
FAR_CROP_SIZE = 256          # Ausschnitt wird auf diese Kantenlänge vergrössert

LOOK_SMOOTHING = 0.35        # Glättung der Blickentscheidung (0..1, höher = schneller)
LOOK_ON = 0.6                # Schwelle, ab der "schaut" gilt
LOOK_OFF = 0.35              # Schwelle, unter der "schaut nicht" gilt

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_FILE = "face_landmarker.task"


# --------------------------------------------------------------------------- #
# Modell laden
# --------------------------------------------------------------------------- #

class ModelError(RuntimeError):
    pass


def app_data_dir() -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "Sicherheitscheck"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Sicherheitscheck"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "sicherheitscheck"


def model_candidates(name: str = MODEL_FILE) -> List[Path]:
    here = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return [here / name, app_data_dir() / name]


def _ssl_contexts():
    import ssl
    out = []
    try:
        import certifi  # sorgt v. a. unter macOS für gültige Zertifikate
        out.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    out.append(ssl.create_default_context())  # System-Zertifikate (z. B. Firmennetzwerke)
    return out


def _download(url: str, target: Path, progress=None, label: str = "Modell") -> None:
    last_exc = None
    for ctx in _ssl_contexts():
        try:
            _download_with(url, target, ctx, progress, label)
            return
        except Exception as exc:  # nächsten Zertifikatsspeicher probieren
            last_exc = exc
    raise last_exc


def _download_with(url: str, target: Path, ctx, progress=None, label: str = "Modell") -> None:
    tmp = target.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=30, context=ctx) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(f"{label} wird heruntergeladen … {done * 100 // total} %")
        tmp.replace(target)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def load_far_model_bytes(progress=None) -> Optional[bytes]:
    """Lädt das Modell der Fern-Erkennung (~230 KB). Fehlt es, läuft das Programm ohne Fern-Erkennung."""
    if not FAR_DETECTION or not hasattr(cv2, "FaceDetectorYN"):
        return None
    for p in model_candidates(YUNET_FILE):
        if p.is_file() and p.stat().st_size > 50_000:
            return p.read_bytes()
    target = app_data_dir() / YUNET_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Modell für die Fern-Erkennung wird einmalig heruntergeladen …")
    for url in YUNET_URLS:
        try:
            _download(url, target, progress, "Fern-Erkennung")
            if target.is_file() and target.stat().st_size > 50_000:
                return target.read_bytes()
        except Exception:
            continue
    return None


def load_model_bytes(progress: Optional[Callable[[str], None]] = None) -> bytes:
    """Lädt das Erkennungsmodell. Beim ersten Start wird es einmalig (~4 MB) heruntergeladen."""
    for p in model_candidates():
        if p.is_file() and p.stat().st_size > 100_000:
            return p.read_bytes()

    target = app_data_dir() / MODEL_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Erkennungsmodell wird einmalig heruntergeladen …")
    try:
        _download(MODEL_URL, target, progress, "Erkennungsmodell")
    except Exception as exc:
        raise ModelError(
            "Das Erkennungsmodell konnte nicht heruntergeladen werden.\n"
            f"Lade die Datei «{MODEL_FILE}» manuell von\n{MODEL_URL}\n"
            f"herunter und lege sie in den Programmordner.\n(Fehler: {exc})"
        ) from exc
    return target.read_bytes()


def create_landmarker(model_bytes: bytes, single_image: bool = False):
    """single_image=True: Landmarker für einzelne, vergrösserte Ausschnitte (ein Gesicht)."""
    import mediapipe as mp  # noqa: F401  (stellt sicher, dass MediaPipe installiert ist)
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.IMAGE if single_image else vision.RunningMode.VIDEO,
        num_faces=1 if single_image else MAX_FACES,
        output_face_blendshapes=True,   # Mimikwerte (Lidschlag, Lächeln, Kiefer)
        min_face_detection_confidence=0.3 if single_image else 0.5,
        min_face_presence_confidence=0.3 if single_image else 0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------------------------------- #
# Kamera
# --------------------------------------------------------------------------- #

def open_camera(index: int = 0):
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _ in range(10):  # manche Kameras liefern erst nach ein paar Bildern
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
                time.sleep(0.05)
        if cap is not None:
            cap.release()
    return None


# --------------------------------------------------------------------------- #
# Geometrie: Kopfhaltung und Blickrichtung
# --------------------------------------------------------------------------- #

# Generisches 3D-Kopfmodell (mm). x nach rechts (im Bild), y nach oben, z aus dem Gesicht heraus.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),            # Nasenspitze
    (0.0, -330.0, -65.0),       # Kinn
    (-225.0, 170.0, -135.0),    # äusserer Augenwinkel, im Bild links
    (225.0, 170.0, -135.0),     # äusserer Augenwinkel, im Bild rechts
    (-150.0, -150.0, -125.0),   # Mundwinkel, im Bild links
    (150.0, -150.0, -125.0),    # Mundwinkel, im Bild rechts
], dtype=np.float64)

IDX_NOSE, IDX_CHIN = 1, 152
IDX_EYE_OUTER = (33, 263)
IDX_MOUTH = (61, 291)
EYES = ((33, 133), (362, 263))  # Augenwinkel je Auge
IRIS_CENTERS = (468, 473)


def _pose_points(lm: np.ndarray) -> np.ndarray:
    """Wählt die 6 Modellpunkte aus; Paare werden nach Bildposition sortiert (robust gegen Spiegelung)."""
    e1, e2 = lm[IDX_EYE_OUTER[0]], lm[IDX_EYE_OUTER[1]]
    m1, m2 = lm[IDX_MOUTH[0]], lm[IDX_MOUTH[1]]
    if e1[0] > e2[0]:
        e1, e2 = e2, e1
    if m1[0] > m2[0]:
        m1, m2 = m2, m1
    return np.array([lm[IDX_NOSE], lm[IDX_CHIN], e1, e2, m1, m2], dtype=np.float64)


def focal_px(frame_w: float) -> float:
    """Brennweite in Pixeln aus dem angenommenen Bildwinkel der Webcam."""
    return float(frame_w) / (2.0 * math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0))


def solve_head_pose(lm: np.ndarray, frame_w: int, frame_h: int):
    """Kopfhaltung per solvePnP. Liefert (R, t) in Kamerakoordinaten oder None."""
    pts = _pose_points(lm)
    f = focal_px(frame_w)
    cam = np.array([[f, 0, frame_w / 2.0], [0, f, frame_h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(4)

    ok, rvec, tvec = False, None, None
    flag = getattr(cv2, "SOLVEPNP_SQPNP", None)
    if flag is not None:
        try:
            ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, pts, cam, dist, flags=flag)
        except cv2.error:
            ok = False
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    if t[2] <= 0:
        return None
    return R, t


def direction_deviation(vec: np.ndarray, t: np.ndarray):
    """
    Winkel (horizontal, vertikal) in Grad zwischen einer Richtung und der Richtung "vom Gesicht zur
    Kamera". Positiv horizontal = zur rechten Bildseite, positiv vertikal = unterhalb der Kamera.
    """
    z = -t / np.linalg.norm(t)
    x = np.array([1.0, 0.0, 0.0]) - z * z[0]
    x /= np.linalg.norm(x)
    y = np.cross(x, z)  # zeigt nach unten
    fx, fy, fz = float(vec @ x), float(vec @ y), float(vec @ z)
    return math.degrees(math.atan2(fx, fz)), math.degrees(math.atan2(fy, math.hypot(fx, fz)))


def head_deviation(lm: np.ndarray, frame_w: int, frame_h: int):
    """Abweichung der Kopfrichtung von der Richtung zur Kamera (Grad) oder None."""
    pose = solve_head_pose(lm, frame_w, frame_h)
    if pose is None:
        return None
    R, t = pose
    return direction_deviation(R @ np.array([0.0, 0.0, 1.0]), t)


def eye_gaze_vector(lm: np.ndarray, R: np.ndarray, min_eye_px: float = MIN_EYE_PX):
    """
    Blickrichtung der Augen als Einheitsvektor in Kamerakoordinaten (oder None).

    Modell: Die Iris liegt auf einer Kugel (Augapfel) mit Radius r. Schaut ein Auge geradeaus,
    liegt die Iris vor der Mitte zwischen den Augenwinkeln. Verschiebt sich die Iris im Bild
    gegenüber dieser Mitte um delta, dreht sich der Blick entsprechend. Zusammen mit der Kopfrichtung
    ergibt das den echten Blickvektor im Raum, unabhängig davon, wie weit der Kopf weggedreht ist:
        blick_xy = kopf_vorwärts_xy + delta / (r * massstab)
    """
    if lm.shape[0] < 478:
        return None
    ax = R @ np.array([1.0, 0.0, 0.0])        # Kopf-Querachse
    fwd = R @ np.array([0.0, 0.0, 1.0])       # Kopf-Vorwärtsrichtung
    proj = math.hypot(ax[0], ax[1])            # Verkürzung der Querachse im Bild
    if proj < 0.25:
        return None
    outer_px = float(np.linalg.norm(lm[IDX_EYE_OUTER[0]] - lm[IDX_EYE_OUTER[1]]))
    scale = outer_px / (OUTER_EYE_DIST_MM * proj)  # Pixel pro Millimeter auf Kopfhöhe
    if scale * EYE_RADIUS_MM < 1.5:
        return None

    irises = [lm[IRIS_CENTERS[0]], lm[IRIS_CENTERS[1]]]
    acc = np.zeros(2)
    wsum = 0.0
    for a, b in EYES:
        pa, pb = lm[a], lm[b]
        width = float(np.linalg.norm(pb - pa))
        if width < min_eye_px:
            continue  # Auge zu klein oder durch die Kopfdrehung verdeckt
        mid = (pa + pb) / 2.0
        iris = min(irises, key=lambda p: float(np.linalg.norm(p - mid)))
        delta = (iris - mid) / (EYE_RADIUS_MM * scale)
        delta[1] *= EYE_VERTICAL_WEIGHT
        weight = width * width  # das besser sichtbare (der Kamera zugewandte) Auge zählt mehr
        acc += weight * delta
        wsum += weight
    if wsum == 0:
        return None

    gxy = fwd[:2] + acc / wsum
    n = float(np.linalg.norm(gxy))
    if n > 0.985:
        gxy *= 0.985 / n
    gz = math.sqrt(max(0.0, 1.0 - float(gxy @ gxy)))
    gz = -gz if fwd[2] <= 0 else gz
    return np.array([gxy[0], gxy[1], gz])


def is_looking(dev_h: float, dev_v: float, yaw_tol: float = LOOK_YAW_TOL_DEG,
               up_tol: float = LOOK_UP_TOL_DEG, down_tol: float = LOOK_DOWN_TOL_DEG) -> bool:
    return abs(dev_h) <= yaw_tol and -up_tol <= dev_v <= down_tol


def strictness(distance_m: Optional[float], far: bool = False) -> float:
    """0 = nah (normale Toleranzen), 1 = weit weg (streng). Gleitender Übergang dazwischen."""
    if distance_m is None:
        return 1.0 if far else 0.0
    return max(0.0, min(1.0, (distance_m - STRICT_FROM_M) / (STRICT_FULL_M - STRICT_FROM_M)))


def judge_looking(face, yaw_tol_near: float) -> None:
    """Setzt face.looking mit Toleranzen je nach Entfernung und die nötige Blickdauer (face.dwell_s)."""
    s = strictness(face.distance_m, face.far)
    yaw_tol = yaw_tol_near + (FAR_YAW_TOL_DEG - yaw_tol_near) * s
    up_tol = LOOK_UP_TOL_DEG + (FAR_UP_TOL_DEG - LOOK_UP_TOL_DEG) * s
    down_tol = LOOK_DOWN_TOL_DEG + (FAR_DOWN_TOL_DEG - LOOK_DOWN_TOL_DEG) * s
    face.strict = s
    face.dwell_s = FAR_DWELL_S * s
    face.looking = is_looking(face.gaze_h, face.gaze_v, yaw_tol, up_tol, down_tol)


# --------------------------------------------------------------------------- #
# Analyse eines Bildes
# --------------------------------------------------------------------------- #

# Landmark-Gruppen für Messungen und die HUD-Darstellung
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EAR_POINTS = (((159, 145), (158, 153), (33, 133)), ((386, 374), (385, 380), (362, 263)))
MODEL_UNITS_PER_MM = 450.0 / 92.0   # Massstab des 3D-Kopfmodells


@dataclass
class Face:
    bbox: tuple                      # (x0, y0, x1, y1) normiert 0..1
    center: tuple                    # normiert
    area: float                      # normiert
    dev_h: Optional[float] = None    # Kopfrichtung relativ zur Kamera
    dev_v: Optional[float] = None
    gaze_h: Optional[float] = None   # tatsächliche Blickrichtung (Augen + Kopf)
    gaze_v: Optional[float] = None
    eyes_used: bool = False          # wurde die Blickrichtung aus den Augen bestimmt?
    looking: Optional[bool] = None   # None = nicht schätzbar
    role: str = "other"              # "owner" (du) oder "other"
    far: bool = False                # über die Fern-Erkennung gefunden
    strict: float = 0.0              # 0 = nah, 1 = weit weg (strenge Blickprüfung)
    dwell_s: float = 0.0             # nötige ununterbrochene Blickdauer, bis es als «starrt» zählt
    staring: Optional[bool] = None   # schaut lange genug hin (vom Tracker gesetzt)
    eye_weight: float = 0.0          # Anteil der Augen an der Blickschätzung (0 = nur Kopf, 1 = nur Augen)
    pts5: Optional[np.ndarray] = field(default=None, repr=False)  # 5 Grobpunkte (nur Fern-Erkennung)
    # zusätzliche Messwerte
    lm: Optional[np.ndarray] = field(default=None, repr=False)  # Gesichtspunkte in Pixeln
    frame_w: int = 0
    frame_h: int = 0
    gaze_vec: Optional[np.ndarray] = field(default=None, repr=False)
    distance_m: Optional[float] = None
    roll_deg: Optional[float] = None
    ear: Optional[float] = None      # Augenöffnung (Eye Aspect Ratio)
    mouth: Optional[float] = None    # Mundöffnung relativ zur Mundbreite
    blend: dict = field(default_factory=dict, repr=False)  # MediaPipe-Mimikwerte
    # vom Tracker ergänzt
    track_id: int = 0
    eyes_open: Optional[float] = None
    blink_rate: Optional[float] = None
    blinks: int = 0
    talking: bool = False
    smiling: Optional[bool] = None
    looking_since: Optional[float] = None
    attention: Optional[float] = None


def eye_aspect_ratio(lm: np.ndarray) -> Optional[float]:
    vals = []
    for (a1, b1), (a2, b2), (c1, c2) in EAR_POINTS:
        horiz = float(np.linalg.norm(lm[c1] - lm[c2]))
        if horiz < 1e-6:
            continue
        vals.append((float(np.linalg.norm(lm[a1] - lm[b1])) + float(np.linalg.norm(lm[a2] - lm[b2]))) / (2 * horiz))
    return sum(vals) / len(vals) if vals else None


def mouth_open_ratio(lm: np.ndarray) -> Optional[float]:
    width = float(np.linalg.norm(lm[61] - lm[291]))
    if width < 1e-6:
        return None
    return float(np.linalg.norm(lm[13] - lm[14])) / width


def analyze_landmarks(lm_px: np.ndarray, w: int, h: int) -> Face:
    x0, y0 = lm_px[:, 0].min(), lm_px[:, 1].min()
    x1, y1 = lm_px[:, 0].max(), lm_px[:, 1].max()
    face = Face(
        bbox=(max(0.0, x0 / w), max(0.0, y0 / h), min(1.0, x1 / w), min(1.0, y1 / h)),
        center=(((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h),
        area=((x1 - x0) / w) * ((y1 - y0) / h),
        lm=lm_px, frame_w=w, frame_h=h,
    )
    if lm_px.shape[0] >= 468:
        face.ear = eye_aspect_ratio(lm_px)
        face.mouth = mouth_open_ratio(lm_px)
        e1, e2 = sorted((lm_px[IDX_EYE_OUTER[0]], lm_px[IDX_EYE_OUTER[1]]), key=lambda p: p[0])
        face.roll_deg = math.degrees(math.atan2(e2[1] - e1[1], e2[0] - e1[0]))

    k = w / 640.0  # Mindestgrössen an die Analyseauflösung anpassen
    if (x1 - x0) < MIN_FACE_PX * k:
        return face
    pose = solve_head_pose(lm_px, w, h)
    if pose is None:
        return face
    R, t = pose
    fwd = R @ np.array([0.0, 0.0, 1.0])
    face.dev_h, face.dev_v = direction_deviation(fwd, t)
    face.distance_m = float(np.linalg.norm(t)) / MODEL_UNITS_PER_MM / 1000.0

    # Je grösser die Augen im Bild, desto mehr zählen sie. Bei weit entfernten Personen (kleine Augen)
    # entscheidet zunehmend die Kopfhaltung.
    gaze = None
    if abs(face.dev_h) <= MAX_HEAD_YAW_FOR_EYES:
        gaze = eye_gaze_vector(lm_px, R, MIN_EYE_PX * k)
    weight = 0.0
    if gaze is not None:
        eye_px = max(float(np.linalg.norm(lm_px[a] - lm_px[b])) for a, b in EYES)
        lo = MIN_EYE_PX * k
        weight = max(0.0, min(1.0, (eye_px - lo) / (lo * (EYE_FULL_FACTOR - 1.0))))
    if weight > 0:
        v = weight * gaze + (1.0 - weight) * fwd
        v = v / max(1e-9, float(np.linalg.norm(v)))
    else:
        v = fwd
    face.eye_weight = weight
    face.eyes_used = weight >= 0.5
    face.gaze_vec = v
    face.gaze_h, face.gaze_v = direction_deviation(v, t)
    tol = LOOK_YAW_TOL_HEAD_DEG + (LOOK_YAW_TOL_DEG - LOOK_YAW_TOL_HEAD_DEG) * weight
    judge_looking(face, tol)
    return face


# Grobes 3D-Modell für die 5 Punkte der Fern-Erkennung (Augenmitten, Nase, Mundwinkel)
FIVE_POINT_MODEL = np.array([
    (-151.0, 170.0, -130.0), (151.0, 170.0, -130.0), (0.0, 0.0, 0.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)


def face_from_points5(box, pts5: np.ndarray, w: int, h: int) -> Face:
    """Gesicht nur aus den 5 Grobpunkten (sehr weit weg): Blickrichtung allein aus der Kopfhaltung."""
    x, y, bw, bh = [float(v) for v in box]
    face = Face(
        bbox=(max(0.0, x / w), max(0.0, y / h), min(1.0, (x + bw) / w), min(1.0, (y + bh) / h)),
        center=((x + bw / 2) / w, (y + bh / 2) / h), area=(bw / w) * (bh / h),
        far=True, frame_w=w, frame_h=h, pts5=pts5,
    )
    e1, e2 = sorted((pts5[0], pts5[1]), key=lambda p: p[0])
    m1, m2 = sorted((pts5[3], pts5[4]), key=lambda p: p[0])
    img = np.array([e1, e2, pts5[2], m1, m2], dtype=np.float64)
    face.roll_deg = math.degrees(math.atan2(e2[1] - e1[1], e2[0] - e1[0]))
    if bw < 10:
        return face
    f = focal_px(w)
    cam = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok = False
    flag = getattr(cv2, "SOLVEPNP_SQPNP", None)
    try:
        if flag is not None:
            ok, rvec, tvec = cv2.solvePnP(FIVE_POINT_MODEL, img, cam, np.zeros(4), flags=flag)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(FIVE_POINT_MODEL, img, cam, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        ok = False
    if not ok:
        return face
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    if t[2] <= 0:
        return face
    fwd = R @ np.array([0.0, 0.0, 1.0])
    face.dev_h, face.dev_v = direction_deviation(fwd, t)
    face.gaze_h, face.gaze_v = face.dev_h, face.dev_v
    face.gaze_vec = fwd
    face.distance_m = float(np.linalg.norm(t)) / MODEL_UNITS_PER_MM / 1000.0
    judge_looking(face, LOOK_YAW_TOL_HEAD_DEG)
    return face


class FarFaceDetector:
    """OpenCV YuNet: findet auch kleine Gesichter weit hinten im Bild."""

    def __init__(self, model_bytes: bytes):
        self._size = (320, 320)
        self._det = None
        try:  # direkt aus dem Speicher (funktioniert auch bei Umlauten im Benutzerpfad)
            self._det = cv2.FaceDetectorYN.create("onnx", np.frombuffer(model_bytes, np.uint8),
                                                  np.array([], np.uint8), self._size, YUNET_SCORE, 0.3, 50)
        except Exception:
            self._det = None
        if self._det is None:
            for p in model_candidates(YUNET_FILE):
                if p.is_file():
                    self._det = cv2.FaceDetectorYN.create(str(p), "", self._size, YUNET_SCORE, 0.3, 50)
                    break
        if self._det is None:
            raise RuntimeError("Fern-Erkennung konnte nicht gestartet werden")
        # Selbsttest: manche OpenCV-Versionen laden das Modell, scheitern aber beim Erkennen
        self._det.detect(np.zeros((self._size[1], self._size[0], 3), np.uint8))

    def detect(self, frame_bgr: np.ndarray):
        """Liefert [(box(x, y, w, h), punkte5 (5x2), sicherheit)] in Koordinaten von frame_bgr."""
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, YUNET_WIDTH / float(w))
        img = frame_bgr if scale >= 1.0 else cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                                                         interpolation=cv2.INTER_AREA)
        size = (img.shape[1], img.shape[0])
        if size != self._size:
            self._det.setInputSize(size)
            self._size = size
        _, found = self._det.detect(img)
        out = []
        if found is None:
            return out
        for row in found:
            box = row[0:4] / scale
            pts = row[4:14].reshape(5, 2) / scale
            out.append((box, pts, float(row[14])))
        return out


class TileFarDetector:
    """Ersatz, falls YuNet nicht verfügbar ist: sucht mit MediaPipe in vergrösserten Bildvierteln."""

    TILE = 0.6  # Kantenlänge einer Kachel relativ zum Bild (überlappend)

    def __init__(self, landmarker, mp_module):
        self._lmk = landmarker
        self._mp = mp_module

    def detect(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        tw, th = int(w * self.TILE), int(h * self.TILE)
        out = []
        for x0 in (0, w - tw):
            for y0 in (0, h - th):
                tile = frame_bgr[y0:y0 + th, x0:x0 + tw]
                rgb = np.ascontiguousarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
                try:
                    res = self._lmk.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
                except Exception:
                    continue
                for lms in (res.face_landmarks or []):
                    lm = np.array([[x0 + p.x * tw, y0 + p.y * th] for p in lms], dtype=np.float64)
                    bx0, by0 = lm.min(axis=0)
                    bx1, by1 = lm.max(axis=0)
                    box = np.array([bx0, by0, bx1 - bx0, by1 - by0])
                    if any(abs(box[0] - b[0][0]) < box[2] * 0.5 and abs(box[1] - b[0][1]) < box[3] * 0.5
                           for b in out):
                        continue  # in überlappenden Kacheln doppelt gefunden
                    pts5 = np.array([(lm[33] + lm[133]) / 2, (lm[362] + lm[263]) / 2, lm[1], lm[61], lm[291]])
                    out.append((box, pts5, 0.9))
        return out


def _crop_square(frame: np.ndarray, cx: float, cy: float, side: float):
    """Quadratischer Ausschnitt (mit schwarzem Rand, falls er über das Bild hinausragt)."""
    h, w = frame.shape[:2]
    s = max(8, int(round(side)))
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    pad = max(0, -x0, -y0, x0 + s - w, y0 + s - h)
    src = frame
    if pad:
        src = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = src[y0 + pad:y0 + pad + s, x0 + pad:x0 + pad + s]
    return crop, float(x0), float(y0), float(s)


def _overlaps(box, faces: List[Face], w: int, h: int) -> bool:
    x, y, bw, bh = box
    cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
    for f in faces:
        x0, y0, x1, y1 = f.bbox
        mx, my = (x1 - x0) * 0.25, (y1 - y0) * 0.25
        if x0 - mx <= cx <= x1 + mx and y0 - my <= cy <= y1 + my:
            return True
    return False


class FaceAnalyzer:
    """Kapselt den MediaPipe-Landmarker und liefert pro Kamerabild eine Liste von Gesichtern."""

    def __init__(self, model_bytes: bytes, far_model_bytes: Optional[bytes] = None):
        import mediapipe as mp
        self._mp = mp
        self._landmarker = create_landmarker(model_bytes)
        self._last_ts = 0
        self._far = None
        self._crop_landmarker = None
        self._far_cache: list = []
        self._frame_no = 0
        self._last_scan = -10 ** 9
        self.far_mode = "aus"
        if FAR_DETECTION:
            try:
                self._crop_landmarker = create_landmarker(model_bytes, single_image=True)
            except Exception:
                self._crop_landmarker = None
            if far_model_bytes and self._crop_landmarker is not None:
                try:
                    self._far = FarFaceDetector(far_model_bytes)
                    self.far_mode = "yunet"
                except Exception:
                    self._far = None
            if self._far is None and self._crop_landmarker is not None:
                self._far = TileFarDetector(self._crop_landmarker, mp)
                self.far_mode = "kacheln"

    @property
    def far_enabled(self) -> bool:
        return self._far is not None

    def _far_faces(self, frame_bgr: np.ndarray, near: List[Face], w: int, h: int) -> List[Face]:
        """Sucht Gesichter, die MediaPipe übersehen hat (meist weit entfernt), und vermisst sie genau."""
        self._frame_no += 1
        every = YUNET_EVERY_N * (2 if isinstance(self._far, TileFarDetector) else 1)
        if self._frame_no - self._last_scan >= every:  # nur jedes n-te Bild komplett absuchen
            self._last_scan = self._frame_no
            try:
                self._far_cache = self._far.detect(frame_bgr)
            except Exception:
                self._far_cache = []
        cands = [d for d in self._far_cache if not _overlaps(d[0], near, w, h)]
        cands.sort(key=lambda d: -d[0][2] * d[0][3])
        out: List[Face] = []
        for box, pts, score in cands[:MAX_FAR_FACES]:
            x, y, bw, bh = box
            side = max(bw, bh) * FAR_CROP_FACTOR
            crop, ox, oy, s = _crop_square(frame_bgr, x + bw / 2, y + bh / 2, side)
            face = None
            if crop.size and self._crop_landmarker is not None:
                big = cv2.resize(crop, (FAR_CROP_SIZE, FAR_CROP_SIZE), interpolation=cv2.INTER_CUBIC)
                rgb = np.ascontiguousarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
                try:
                    res = self._crop_landmarker.detect(
                        self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
                except Exception:
                    res = None
                if res is not None and res.face_landmarks:
                    lm = np.array([[ox + p.x * s, oy + p.y * s] for p in res.face_landmarks[0]], dtype=np.float64)
                    face = analyze_landmarks(lm, w, h)
                    blends = getattr(res, "face_blendshapes", None) or []
                    if blends and blends[0]:
                        face.blend = {c.category_name: float(c.score) for c in blends[0]}
            if face is None and score >= YUNET_SCORE_POINTS_ONLY:
                face = face_from_points5(box, pts, w, h)
            if face is not None:
                face.far = True
                out.append(face)
        return out

    def process(self, frame_bgr: np.ndarray):
        """Analysiert ein Kamerabild. Liefert (Gesichter, verkleinertes Vorschaubild)."""
        h0, w0 = frame_bgr.shape[:2]
        if w0 > PROCESS_WIDTH:
            s = PROCESS_WIDTH / w0
            frame_bgr = cv2.resize(frame_bgr, (PROCESS_WIDTH, int(round(h0 * s))), interpolation=cv2.INTER_AREA)
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        faces = []
        blends = getattr(result, "face_blendshapes", None) or []
        for i, landmarks in enumerate(result.face_landmarks or []):
            lm = np.array([[p.x * w, p.y * h] for p in landmarks], dtype=np.float64)
            face = analyze_landmarks(lm, w, h)
            if i < len(blends) and blends[i]:
                face.blend = {c.category_name: float(c.score) for c in blends[i]}
            faces.append(face)
        if self._far is not None:
            faces.extend(self._far_faces(frame_bgr, faces, w, h))

        if w > PREVIEW_WIDTH:
            preview = cv2.resize(frame_bgr, (PREVIEW_WIDTH, int(round(h * PREVIEW_WIDTH / w))),
                                 interpolation=cv2.INTER_AREA)
        else:
            preview = frame_bgr
        return faces, preview

    def close(self):
        for lmk in (self._landmarker, self._crop_landmarker):
            try:
                if lmk is not None:
                    lmk.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Verfolgung über mehrere Bilder: feste IDs und Messwerte über die Zeit
# --------------------------------------------------------------------------- #

@dataclass
class _Track:
    tid: int
    center: tuple
    first_seen: float
    last_seen: float
    eye_closed: bool = False
    closed_since: float = 0.0
    blink_times: list = field(default_factory=list)
    blinks: int = 0
    mouth_hist: list = field(default_factory=list)   # (zeit, mundöffnung)
    look_hist: list = field(default_factory=list)    # (zeit, schaut)
    looking_start: Optional[float] = None
    not_looking_since: Optional[float] = None


class FaceTracker:
    MATCH_DIST = 0.18       # max. Verschiebung des Gesichtszentrums zwischen zwei Bildern (normiert)
    EXPIRE_S = 1.5          # so lange wird eine verlorene Spur aufbewahrt
    BLINK_CLOSED = 0.5      # Blendshape-Schwelle "Auge zu"
    EAR_CLOSED, EAR_OPEN = 0.19, 0.24
    ATTENTION_WINDOW_S = 10.0

    def __init__(self):
        self._tracks: List[_Track] = []
        self._next_id = 1

    def update(self, faces: List[Face], now: float) -> None:
        free = list(self._tracks)
        for f in sorted(faces, key=lambda x: -x.area):
            best, best_d = None, self.MATCH_DIST
            for tr in free:
                d = math.hypot(f.center[0] - tr.center[0], f.center[1] - tr.center[1])
                if d < best_d:
                    best, best_d = tr, d
            if best is None:
                best = _Track(self._next_id, f.center, now, now)
                self._next_id += 1
                self._tracks.append(best)
            else:
                free.remove(best)
            best.center, best.last_seen = f.center, now
            self._measure(best, f, now)
        self._tracks = [t for t in self._tracks if now - t.last_seen <= self.EXPIRE_S]

    def _measure(self, tr: _Track, f: Face, now: float) -> None:
        f.track_id = tr.tid

        # Augenöffnung und Lidschlag
        blink = None
        if "eyeBlinkLeft" in f.blend and "eyeBlinkRight" in f.blend:
            blink = (f.blend["eyeBlinkLeft"] + f.blend["eyeBlinkRight"]) / 2
            f.eyes_open = max(0.0, min(1.0, 1.0 - blink))
            closed = blink > self.BLINK_CLOSED
        elif f.ear is not None:
            f.eyes_open = max(0.0, min(1.0, (f.ear - 0.12) / (0.30 - 0.12)))
            closed = f.ear < self.EAR_CLOSED if not tr.eye_closed else f.ear < self.EAR_OPEN
        else:
            closed = tr.eye_closed
        if closed and not tr.eye_closed:
            tr.eye_closed, tr.closed_since = True, now
        elif not closed and tr.eye_closed:
            tr.eye_closed = False
            if now - tr.closed_since < 0.5:      # kurzes Schliessen = Lidschlag
                tr.blinks += 1
                tr.blink_times.append(now)
        tr.blink_times = [t for t in tr.blink_times if now - t <= 60.0]
        age = now - tr.first_seen
        f.blinks = tr.blinks
        f.blink_rate = len(tr.blink_times) * 60.0 / min(60.0, age) if age >= 8.0 else None

        # Sprechen: Mund öffnet und schliesst sich laufend
        if f.mouth is not None:
            tr.mouth_hist.append((now, f.mouth))
        tr.mouth_hist = [(t, m) for t, m in tr.mouth_hist if now - t <= 1.2]
        if len(tr.mouth_hist) >= 6:
            vals = [m for _, m in tr.mouth_hist]
            f.talking = (max(vals) - min(vals)) > 0.12 and max(vals) > 0.10

        # Lächeln
        if "mouthSmileLeft" in f.blend and "mouthSmileRight" in f.blend:
            f.smiling = (f.blend["mouthSmileLeft"] + f.blend["mouthSmileRight"]) / 2 > 0.45

        # Blickdauer und Fokus (Anteil der letzten 10 s mit Blick auf den Bildschirm)
        looking = bool(f.looking)
        if looking:
            tr.not_looking_since = None
            if tr.looking_start is None:
                tr.looking_start = now
        else:
            if tr.not_looking_since is None:
                tr.not_looking_since = now
            if now - tr.not_looking_since > 0.4:
                tr.looking_start = None
        f.looking_since = (now - tr.looking_start) if tr.looking_start is not None else None
        # Weiter weg zählt erst ein längerer, ununterbrochener Blick («Starren»)
        f.staring = looking and (f.dwell_s <= 0 or (f.looking_since or 0.0) >= f.dwell_s)
        tr.look_hist.append((now, looking))
        tr.look_hist = [(t, v) for t, v in tr.look_hist if now - t <= self.ATTENTION_WINDOW_S]
        if age >= 1.0:
            f.attention = sum(1 for _, v in tr.look_hist if v) / len(tr.look_hist)

# --------------------------------------------------------------------------- #
# Zustandslogik
# --------------------------------------------------------------------------- #

@dataclass
class Event:
    kind: str            # "alert" (neue Warnung), "looking" (Blick erkannt), "clear" (Person weg)
    looking: bool = False


@dataclass
class SecurityLogic:
    intruder: bool = False                 # zweite Person aktuell (bestätigt) im Bild
    episode_looked: bool = False           # hat während dieser Episode jemand auf den Bildschirm geschaut?
    look_score: float = 0.0
    looking: bool = False
    _streak_start: Optional[float] = None
    _last_other: float = field(default=-1e9)
    _owner_center: Optional[tuple] = None

    def _assign_roles(self, faces: List[Face]) -> None:
        if not faces:
            return
        max_area = max(f.area for f in faces)
        owner = max(faces, key=lambda f: f.area)
        if self._owner_center is not None:
            candidates = [f for f in faces if f.area >= 0.5 * max_area]
            owner = min(candidates, key=lambda f: (f.center[0] - self._owner_center[0]) ** 2
                        + (f.center[1] - self._owner_center[1]) ** 2)
        for f in faces:
            f.role = "owner" if f is owner else "other"
        self._owner_center = owner.center

    def update(self, now: float, faces: List[Face]) -> List[Event]:
        events: List[Event] = []
        self._assign_roles(faces)
        others = [f for f in faces if f.role == "other"]
        raw = len(others) > 0

        # Blickrichtung glätten
        target = 1.0 if any((f.staring if f.staring is not None else f.looking) for f in others) else 0.0
        self.look_score += LOOK_SMOOTHING * (target - self.look_score)
        if not self.looking and self.look_score >= LOOK_ON:
            self.looking = True
        elif self.looking and self.look_score <= LOOK_OFF:
            self.looking = False

        # Anwesenheit entprellen
        if raw:
            if self._streak_start is None or now - self._last_other > GAP_TOLERANCE_S:
                self._streak_start = now
            self._last_other = now
        elif self._streak_start is not None and now - self._last_other > GAP_TOLERANCE_S:
            self._streak_start = None

        if not self.intruder:
            if self._streak_start is not None and self._last_other - self._streak_start >= CONFIRM_PRESENT_S:
                self.intruder = True
                self.episode_looked = self.looking
                events.append(Event("alert", self.looking))
        else:
            if self.looking and not self.episode_looked:
                self.episode_looked = True
                events.append(Event("looking", True))
            if now - self._last_other >= CONFIRM_ABSENT_S:
                self.intruder = False
                self.episode_looked = False
                self._streak_start = None
                events.append(Event("clear"))
        return events
