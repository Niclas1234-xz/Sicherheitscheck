#!/usr/bin/env python3
"""
Sicherheitscheck – schützt deinen Bildschirm vor fremden Blicken.

Start:  python sicherheitscheck.py            (Standard-Webcam)
        python sicherheitscheck.py --kamera 1 (andere Webcam)
        python sicherheitscheck.py --warnung-testen  (zeigt die Warnung als Vorschau)
"""

from __future__ import annotations

import sys


def _disable_lazy_imports() -> None:
    """
    Python 3.15+ (PEP 810) lädt Teile der Standardbibliothek «lazy». Zusammen mit dem Import-Hook
    von PySide6 bleiben diese Platzhalter unaufgelöst, was MediaPipe abstürzen lässt
    ("'lazy_import' object is not callable"). Darum werden Lazy Imports hier vor allen
    anderen Imports abgeschaltet. Unter älteren Python-Versionen passiert nichts.
    """
    setter = getattr(sys, "set_lazy_imports", None)
    if setter is not None:
        try:
            setter("none")
        except Exception:
            pass
    filt = getattr(sys, "set_lazy_imports_filter", None)
    if filt is not None:
        try:
            filt(lambda *args: False)
        except Exception:
            pass


_disable_lazy_imports()

import argparse  # noqa: E402
import getpass  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import math  # noqa: E402
import concurrent.futures  # noqa: E402
import concurrent.futures.thread  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

# Zusätzliche Absicherung: echte Klasse statt Platzhalter hinterlegen (wird von MediaPipe verwendet)
if not isinstance(concurrent.futures.__dict__.get("ThreadPoolExecutor"), type):
    concurrent.futures.ThreadPoolExecutor = concurrent.futures.thread.ThreadPoolExecutor

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import mediapipe  # noqa: E402,F401  (vor PySide6 laden, damit der Import-Hook nicht stört)

import erkennung as E  # noqa: E402


def _resolve_pending_lazy_imports() -> None:
    """Löst noch offene Lazy-Import-Platzhalter auf, solange PySide6 noch nicht geladen ist."""
    for mod in list(sys.modules.values()):
        try:
            items = list(vars(mod).items())
        except Exception:
            continue
        for name, value in items:
            if type(value).__name__ == "lazy_import":
                try:
                    setattr(mod, name, getattr(mod, name))
                except Exception:
                    pass


_resolve_pending_lazy_imports()

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QAction, QColor, QFont, QGuiApplication, QIcon, QImage,
                           QBrush, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QPolygonF,
                           QRadialGradient)
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy,
                               QSystemTrayIcon, QVBoxLayout, QWidget)


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #

BG = "#0D1017"
CARD = "#151A24"
BORDER = "#242B3A"
TEXT = "#EEF1F7"
MUTED = "#8C95A8"
ACCENT = "#6C8CFF"
ACCENT_2 = "#9B6CFF"
GREEN = "#2FD98E"
AMBER = "#FFB547"
RED = "#FF4D6A"

FONT_FAMILIES = ["Segoe UI Variable Display", "Segoe UI", "SF Pro Display", ".AppleSystemUIFont",
                 "Helvetica Neue", "Inter", "Cantarell", "Ubuntu", "Arial"]

MSG_LOOKING = "Eine zweite Person schaut auf deinen Bildschirm."
AUTO_CLOSE_S = 5.0  # Warnung schliesst sich spätestens nach dieser Zeit


def ui_font(px: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


def shield_path(r: QRectF) -> QPainterPath:
    x, y, w, h = r.x(), r.y(), r.width(), r.height()
    p = QPainterPath()
    p.moveTo(x + w * 0.5, y)
    p.cubicTo(x + w * 0.70, y + h * 0.09, x + w * 0.86, y + h * 0.12, x + w, y + h * 0.13)
    p.lineTo(x + w, y + h * 0.46)
    p.cubicTo(x + w, y + h * 0.74, x + w * 0.76, y + h * 0.91, x + w * 0.5, y + h)
    p.cubicTo(x + w * 0.24, y + h * 0.91, x, y + h * 0.74, x, y + h * 0.46)
    p.lineTo(x, y + h * 0.13)
    p.cubicTo(x + w * 0.14, y + h * 0.12, x + w * 0.30, y + h * 0.09, x + w * 0.5, y)
    p.closeSubpath()
    return p


def eye_path(cx: float, cy: float, w: float) -> QPainterPath:
    h = w * 0.62
    p = QPainterPath()
    p.moveTo(cx - w / 2, cy)
    p.quadTo(cx, cy - h, cx + w / 2, cy)
    p.quadTo(cx, cy + h, cx - w / 2, cy)
    p.closeSubpath()
    return p


def draw_eye(p: QPainter, cx: float, cy: float, w: float, fill: QColor, pupil: QColor) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(fill)
    p.drawPath(eye_path(cx, cy, w))
    p.setBrush(pupil)
    p.drawEllipse(QPointF(cx, cy), w * 0.17, w * 0.17)


_icon_cache: dict = {}


def make_icon(color: str, size: int = 128) -> QIcon:
    if color in _icon_cache:
        return _icon_cache[color]
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    r = QRectF(size * 0.12, size * 0.06, size * 0.76, size * 0.88)
    g = QLinearGradient(r.topLeft(), r.bottomRight())
    g.setColorAt(0, QColor(color).lighter(120))
    g.setColorAt(1, QColor(color).darker(125))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(g)
    p.drawPath(shield_path(r))
    draw_eye(p, size * 0.5, size * 0.48, size * 0.46, QColor("#FFFFFF"), QColor(color).darker(160))
    p.end()
    icon = QIcon(pm)
    _icon_cache[color] = icon
    return icon


# --------------------------------------------------------------------------- #
# macOS: verhindert, dass das System die Erkennung im Hintergrund drosselt (App Nap)
# --------------------------------------------------------------------------- #

def begin_background_activity():
    if sys.platform != "darwin":
        return None
    try:
        from Foundation import NSProcessInfo  # optional (pyobjc)
        opts = 0x00FFFFFF & ~(1 << 20)  # NSActivityUserInitiatedAllowingIdleSystemSleep
        return NSProcessInfo.processInfo().beginActivityWithOptions_reason_(opts, "Sicherheitscheck aktiv")
    except Exception:
        return None


def end_background_activity(token) -> None:
    if token is None:
        return
    try:
        from Foundation import NSProcessInfo
        NSProcessInfo.processInfo().endActivity_(token)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Erkennung im Hintergrund-Thread
# --------------------------------------------------------------------------- #

def hud_payload(f) -> dict:
    """Bereitet die Messwerte eines Gesichts für das HUD auf (gespiegelt wie die Vorschau)."""
    x0, y0, x1, y1 = f.bbox
    d = {
        "box": (1 - x1, y0, 1 - x0, y1), "role": f.role, "id": f.track_id,
        # «looking» = zählt für die Warnung (weiter weg erst nach längerem Hinschauen)
        "looking": bool(f.staring if f.staring is not None else f.looking),
        "looking_raw": bool(f.looking), "dwell": f.dwell_s, "strict": f.strict,
        "distance": f.distance_m, "eyes_used": f.eyes_used, "eyes_open": f.eyes_open,
        "far": f.far, "eye_weight": f.eye_weight, "has_mesh": f.lm is not None,
        "blink_rate": f.blink_rate, "talking": f.talking, "smiling": f.smiling,
        "looking_since": f.looking_since, "attention": f.attention,
        # Winkel aus Sicht der gespiegelten Vorschau (positiv = nach rechts bzw. unten)
        "yaw": -f.dev_h if f.dev_h is not None else None,
        "pitch": f.dev_v,
        "roll": -f.roll_deg if f.roll_deg is not None else None,
    }
    lm = f.lm
    if lm is None and f.pts5 is not None and f.frame_w:
        d["pts5"] = [(1.0 - x / f.frame_w, y / f.frame_h) for x, y in f.pts5]
        if f.gaze_vec is not None:
            d["gaze"] = (-float(f.gaze_vec[0]), float(f.gaze_vec[1]))
    if lm is None or not f.frame_w:
        return d
    w, h = float(f.frame_w), float(f.frame_h)

    def P(i):
        return (1.0 - lm[i][0] / w, lm[i][1] / h)

    n = len(lm)
    if n >= 468:
        d["mesh"] = [P(i) for i in range(0, 468, 3)]
        d["oval"] = [P(i) for i in E.FACE_OVAL]
        d["eyes"] = [[P(i) for i in E.RIGHT_EYE_CONTOUR], [P(i) for i in E.LEFT_EYE_CONTOUR]]
    if n >= 478:
        irises = []
        for c, a, b in ((468, 469, 471), (473, 474, 476)):
            irises.append((P(c), float(np.linalg.norm(lm[a] - lm[b])) / 2.0 / w))
        d["irises"] = irises
        if f.gaze_vec is not None:
            d["gaze"] = (-float(f.gaze_vec[0]), float(f.gaze_vec[1]))  # gespiegelt
    return d


class DetectionWorker(QThread):
    status = Signal(dict)
    preview = Signal(QImage, list)
    alert = Signal(bool)          # neue Warnung (True = schaut auf Bildschirm)
    alert_update = Signal(bool)   # während laufender Warnung wurde ein Blick erkannt
    cleared = Signal()            # zweite Person ist verschwunden
    failed = Signal(str)
    running_ok = Signal()

    def __init__(self, camera_index: int):
        super().__init__()
        self.camera_index = camera_index
        self.preview_enabled = True
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self._stop and time.monotonic() < end:
            time.sleep(min(0.05, max(0.0, end - time.monotonic())))

    def run(self) -> None:
        token = begin_background_activity()
        analyzer = None
        cap = None
        try:
            self.status.emit({"phase": "starting", "text": "Erkennungsmodell wird geladen …"})
            model = E.load_model_bytes(lambda m: self.status.emit({"phase": "starting", "text": m}))
            if self._stop:
                return
            far_model = E.load_far_model_bytes(lambda m: self.status.emit({"phase": "starting", "text": m}))
            if self._stop:
                return
            analyzer = E.FaceAnalyzer(model, far_model)

            self.status.emit({"phase": "starting", "text": "Webcam wird geöffnet …"})
            cap = E.open_camera(self.camera_index)
            if cap is None:
                self.failed.emit("Die Webcam konnte nicht geöffnet werden. Prüfe, ob sie angeschlossen ist, "
                                 "nicht von einem anderen Programm verwendet wird und der Kamerazugriff "
                                 "in den Datenschutzeinstellungen erlaubt ist.")
                return
            self.running_ok.emit()

            logic = E.SecurityLogic()
            tracker = E.FaceTracker()
            interval = 1.0 / E.MAX_FPS
            fails = 0
            fps = 0.0
            last_frame_t = time.monotonic()
            last_status_t = 0.0

            while not self._stop:
                t0 = time.monotonic()
                ok, frame = cap.read()
                if not ok or frame is None:
                    fails += 1
                    if fails >= 30:
                        cap.release()
                        cap = None
                        self.status.emit({"phase": "camera_lost"})
                        while not self._stop and cap is None:
                            self._sleep(2.0)
                            if not self._stop:
                                cap = E.open_camera(self.camera_index)
                        fails = 0
                        last_frame_t = time.monotonic()
                    else:
                        time.sleep(0.03)
                    continue
                fails = 0

                faces, small = analyzer.process(frame)
                now = time.monotonic()
                tracker.update(faces, now)  # zuerst: liefert die Blickdauer für die Warnlogik
                for ev in logic.update(now, faces):
                    if ev.kind == "alert":
                        self.alert.emit(ev.looking)
                    elif ev.kind == "looking":
                        self.alert_update.emit(True)
                    elif ev.kind == "clear":
                        self.cleared.emit()

                dt = now - last_frame_t
                last_frame_t = now
                if dt > 0:
                    fps = (1.0 / dt) if fps == 0 else fps * 0.9 + (1.0 / dt) * 0.1

                if now - last_status_t >= 0.25:
                    last_status_t = now
                    others = [f for f in faces if f.role == "other"]
                    self.status.emit({
                        "phase": "active",
                        "faces": len(faces),
                        "others": len(others),
                        "intruder": logic.intruder,
                        "looking": logic.looking,
                        "fps": fps,
                        "far": analyzer.far_enabled,
                    })

                if self.preview_enabled:
                    mirrored = cv2.cvtColor(cv2.flip(small, 1), cv2.COLOR_BGR2RGB)
                    h, w = mirrored.shape[:2]
                    img = QImage(mirrored.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                    self.preview.emit(img, [hud_payload(f) for f in faces])

                rest = interval - (time.monotonic() - t0)
                if rest > 0:
                    self._sleep(rest)
        except E.ModelError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Unerwarteter Fehler: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if analyzer is not None:
                analyzer.close()
            end_background_activity(token)


# --------------------------------------------------------------------------- #
# Warn-Insel (im Stil der Dynamic Island): klappt oben in der Bildschirmmitte auf
# --------------------------------------------------------------------------- #

ISLAND_TOP_MARGIN = 14      # Abstand zum oberen Bildschirmrand
ISLAND_PAD = 48             # unsichtbarer Rand für Schatten und das federnde Überschwingen
ISLAND_HEIGHT = 150         # Höhe der aufgeklappten Insel
ISLAND_RADIUS = 32
PILL_W, PILL_H = 168, 38    # Grösse der kleinen Pille zu Beginn
OPEN_S = 1.05               # Dauer der Öffnungsanimation
CLOSE_S = 0.6               # Dauer der Schliessanimation


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _ease_out_cubic(x: float) -> float:
    x = _clamp01(x)
    return 1 - (1 - x) ** 3


def _ease_in_cubic(x: float) -> float:
    x = _clamp01(x)
    return x ** 3


def _ease_out_back(x: float, s: float = 1.4) -> float:
    """Federnd: schiesst leicht über das Ziel hinaus und schwingt zurück."""
    x = _clamp01(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _qflags(*flags) -> int:
    """Kombiniert Qt-Ausrichtungs- und Textflags zu einer Ganzzahl (für QPainter.drawText)."""
    out = 0
    for f in flags:
        out |= int(getattr(f, "value", f))
    return out


class AlertIsland(QWidget):
    """Animierte Warnung oben in der Bildschirmmitte. Nimmt keinen Fokus und blockiert nichts."""

    close_requested = Signal()

    def __init__(self, screen):
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        flags |= Qt.WindowType.NoDropShadowWindowHint  # eigener Schatten statt System-Schatten
        if sys.platform != "darwin":
            flags |= Qt.WindowType.Tool
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.setWindowTitle("Sicherheitswarnung")

        g = screen.availableGeometry() if sys.platform == "darwin" else screen.geometry()
        sw = g.width()
        # zwischen einem und zwei Dritteln der Bildschirmbreite, bevorzugt ca. 42 %
        self.card_w = int(max(sw / 3.0, min(sw * 2 / 3.0, max(600.0, sw * 0.42))))
        self.card_h = ISLAND_HEIGHT
        win_w = self.card_w + 2 * ISLAND_PAD
        win_h = ISLAND_TOP_MARGIN + self.card_h + ISLAND_PAD
        self.setGeometry(g.x() + (sw - win_w) // 2, g.y(), win_w, win_h)

        self._when = ""
        self._fraction = 1.0
        self._remaining = AUTO_CLOSE_S
        self._t_open = 0.0
        self._t_close: float | None = None
        self._t0 = time.monotonic()
        self._hover = False
        self._button = QRectF()
        self._content_alpha = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    # ---------- Steuerung ----------
    def open(self, when: str) -> None:
        self._when = when
        self._t_open = time.monotonic()
        self._t_close = None
        self.show()
        self.raise_()
        self._timer.start()

    def close_animated(self) -> None:
        if self._t_close is None:
            self._t_close = time.monotonic()

    def close_now(self) -> None:
        self._timer.stop()
        self.hide()
        self.deleteLater()

    def set_countdown(self, remaining: float) -> None:
        self._remaining = remaining
        self._fraction = _clamp01(remaining / AUTO_CLOSE_S)

    # ---------- Animation ----------
    def _state(self):
        """Aktuelle Form: (Rechteck, Eckenradius, Inhalts-Deckkraft, Ausklapp-Grad, fertig_geschlossen)."""
        now = time.monotonic()
        t = (now - self._t_open) / OPEN_S
        drop = _ease_out_cubic(t / 0.28)                         # Pille gleitet von oben herein
        expand = min(1.06, _ease_out_back((t - 0.22) / 0.5))     # Pille federt zur Insel auf
        content = _ease_out_cubic((t - 0.55) / 0.35)             # Inhalt blendet ein
        done = False
        if self._t_close is not None:
            u = (now - self._t_close) / CLOSE_S
            content = min(content, 1 - _ease_out_cubic(u / 0.3))
            expand = min(expand, 1 - _ease_in_cubic((u - 0.15) / 0.5))
            drop = min(drop, 1 - _ease_in_cubic((u - 0.6) / 0.4))
            done = u >= 1.0
        expand = max(0.0, expand)
        w = _lerp(PILL_W, self.card_w, expand)
        h = _lerp(PILL_H, self.card_h, expand)
        r = min(_lerp(PILL_H / 2, ISLAND_RADIUS, _clamp01(expand)), h / 2)
        top = _lerp(-PILL_H - 6, ISLAND_TOP_MARGIN, drop)
        rect = QRectF(self.width() / 2 - w / 2, top, w, h)
        return rect, r, content, expand, done

    def _tick(self) -> None:
        if self._state()[4]:
            self.close_now()
            return
        self.update()

    # ---------- Maus ----------
    def _over_button(self, pos) -> bool:
        return self._content_alpha > 0.5 and self._button.contains(pos)

    def mouseMoveEvent(self, e):
        hover = self._over_button(e.position())
        if hover != self._hover:
            self._hover = hover
            if hover:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.unsetCursor()
            self.update()
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        if self._hover:
            self._hover = False
            self.unsetCursor()
            self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if self._over_button(e.position()):
            self.close_requested.emit()
        super().mousePressEvent(e)

    # ---------- Zeichnen ----------
    def paintEvent(self, _):
        rect, radius, content, expand, _done = self._state()
        self._content_alpha = content
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        phase = (time.monotonic() - self._t0)
        pulse = 0.5 - 0.5 * math.cos(phase * 2 * math.pi / 1.6)
        red = QColor(RED)

        # weicher Schatten
        p.setPen(Qt.PenStyle.NoPen)
        layers = 12
        for i in range(layers, 0, -1):
            spread = i * 2.4
            a = int((3 + 10 * (1 - i / layers)) * (0.4 + 0.6 * _clamp01(expand)))
            p.setBrush(QColor(0, 0, 0, a))
            p.drawRoundedRect(rect.adjusted(-spread, -spread + 8, spread, spread + 8),
                              radius + spread, radius + spread)

        # Insel
        shape = QPainterPath()
        shape.addRoundedRect(rect, radius, radius)
        p.fillPath(shape, QColor(10, 10, 14, 250))
        p.save()
        p.setClipPath(shape)
        glow = QRadialGradient(QPointF(rect.x() + 70, rect.center().y()), rect.height() * 1.1)
        g0 = QColor(red)
        g0.setAlpha(int(40 + 40 * pulse))
        glow.setColorAt(0, g0)
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(rect, glow)
        p.restore()
        border = QColor(red)
        border.setAlphaF(0.30 + 0.40 * pulse)
        p.setPen(QPen(border, 1.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(shape)

        # kleiner roter Punkt, solange die Insel noch eine Pille ist
        dot_alpha = _clamp01(1 - expand * 2.2)
        if dot_alpha > 0:
            d = QColor(red)
            d.setAlphaF(dot_alpha * (0.6 + 0.4 * pulse))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(d)
            p.drawEllipse(QPointF(rect.x() + rect.height() / 2 + 2, rect.center().y()), 5.5, 5.5)

        if content <= 0.01:
            return

        # ---------- Inhalt ----------
        p.save()
        p.setClipPath(shape)
        p.setOpacity(content)
        p.translate(0, (1 - content) * 10)
        pad = 26

        # Symbol: pulsierendes Auge
        core = 31.0
        c = QPointF(rect.x() + pad + 44, rect.center().y() - 6)
        for k in range(3):
            ph = (phase / 2.2 + k / 3.0) % 1.0
            rr = core + 20 * ph
            ring = QColor(red)
            ring.setAlphaF(0.55 * (1 - ph))
            p.setPen(QPen(ring, 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, rr, rr)
        scale = 0.85 + 0.15 * _ease_out_back(content)
        cg = QRadialGradient(c - QPointF(core * 0.3, core * 0.35), core * 1.5)
        cg.setColorAt(0, red.lighter(125))
        cg.setColorAt(1, red.darker(120))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(cg)
        p.drawEllipse(c, core * scale, core * scale)
        dark = QColor("#170A0E")
        draw_eye(p, c.x(), c.y(), core * 1.3 * scale, dark, red.lighter(115))
        p.setBrush(dark)
        p.drawEllipse(c, core * 0.09 * scale, core * 0.09 * scale)

        # Schaltfläche «Schliessen»
        bw, bh = 132.0, 44.0
        self._button = QRectF(rect.right() - pad - bw, rect.center().y() - bh / 2 - 6, bw, bh)
        p.setBrush(QColor("#E6E9F0") if self._hover else QColor("#FFFFFF"))
        p.drawRoundedRect(self._button, bh / 2, bh / 2)
        p.setPen(QColor("#0D1017"))
        p.setFont(ui_font(15, QFont.Weight.Bold))
        p.drawText(self._button, _qflags(Qt.AlignmentFlag.AlignCenter), "Schliessen")

        # Texte
        x0 = c.x() + core + 26
        tw = self._button.left() - 22 - x0
        top = rect.y() + 22
        p.setPen(QColor("#FF6B82"))
        p.setFont(ui_font(22, QFont.Weight.Bold))
        p.drawText(QRectF(x0, top, tw, 30), _qflags(Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignVCenter),
                   "Achtung!")
        p.setPen(QColor("#FFFFFF"))
        p.setFont(ui_font(17, QFont.Weight.DemiBold))
        p.drawText(QRectF(x0, top + 32, tw, 48),
                   _qflags(Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignTop, Qt.TextFlag.TextWordWrap),
                   MSG_LOOKING)
        p.setPen(QColor(255, 255, 255, 140))
        p.setFont(ui_font(12))
        secs = max(1, math.ceil(self._remaining))
        p.drawText(QRectF(x0, rect.bottom() - 40, tw, 18),
                   _qflags(Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignVCenter),
                   f"Erkannt um {self._when} Uhr  ·  schliesst in {secs} s")

        # Countdown-Balken
        bar = QRectF(rect.x() + pad, rect.bottom() - 13, rect.width() - 2 * pad, 3)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 26))
        p.drawRoundedRect(bar, 1.5, 1.5)
        if self._fraction > 0:
            p.setBrush(red)
            p.drawRoundedRect(QRectF(bar.x(), bar.y(), bar.width() * self._fraction, bar.height()), 1.5, 1.5)
        p.restore()


class OverlayManager(QObject):
    """Zeigt die Warn-Insel auf allen angeschlossenen Bildschirmen und schliesst sie nach AUTO_CLOSE_S."""

    def __init__(self):
        super().__init__()
        self._islands: list[AlertIsland] = []
        self._deadline = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

    def is_visible(self) -> bool:
        return bool(self._islands)

    def _tick(self) -> None:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            self.hide_all()
            return
        for isl in self._islands:
            isl.set_countdown(remaining)

    def show_alert(self, looking: bool = True) -> None:
        when = datetime.now().strftime("%H:%M:%S")
        self._deadline = time.monotonic() + AUTO_CLOSE_S
        if not self._islands:
            for screen in QGuiApplication.screens():
                isl = AlertIsland(screen)
                isl.close_requested.connect(self.hide_all)
                isl.open(when)
                self._islands.append(isl)
        self._tick()
        self._timer.start()

    def hide_all(self, immediate: bool = False) -> None:
        self._timer.stop()
        for isl in self._islands:
            if immediate:
                isl.close_now()
            else:
                isl.close_animated()  # löscht sich nach der Animation selbst
        self._islands = []


# --------------------------------------------------------------------------- #
# Dezenter Hinweis-Streifen am oberen Bildschirmrand
# --------------------------------------------------------------------------- #

STRIPE_BAR_PX = 6     # Höhe des farbigen Streifens
STRIPE_GLOW_PX = 18   # sanfter Schein darunter


class TopStripe(QWidget):
    """Farbiger Streifen oben am Bildschirm. Durchklickbar, nimmt nie den Fokus."""

    def __init__(self):
        flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                 | Qt.WindowType.WindowTransparentForInput | Qt.WindowType.WindowDoesNotAcceptFocus)
        if sys.platform != "darwin":
            flags |= Qt.WindowType.Tool
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("Sicherheitshinweis")
        self._color = QColor(AMBER)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def set_color(self, color: str) -> None:
        if QColor(color) != self._color:
            self._color = QColor(color)
            self.update()

    def showEvent(self, e):
        self._timer.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._timer.stop()
        super().hideEvent(e)

    def _tick(self):
        self._phase = (self._phase + 0.033 / 1.8) % 1.0  # ruhiges «Atmen», 1,8 s pro Zyklus
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        w = self.width()
        breath = 0.5 - 0.5 * math.cos(self._phase * 2 * math.pi)
        p.fillRect(QRectF(0, 0, w, STRIPE_BAR_PX), self._color)
        glow = QLinearGradient(0, STRIPE_BAR_PX, 0, STRIPE_BAR_PX + STRIPE_GLOW_PX)
        top = QColor(self._color)
        top.setAlphaF(0.25 + 0.35 * breath)
        glow.setColorAt(0, top)
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(QRectF(0, STRIPE_BAR_PX, w, STRIPE_GLOW_PX), glow)


class StripeManager(QObject):
    """Zeigt den Streifen auf allen Bildschirmen."""

    def __init__(self):
        super().__init__()
        self._stripes: list[TopStripe] = []
        self._color = AMBER

    def show(self, color: str = AMBER) -> None:
        self._color = color
        if not self._stripes:
            for screen in QGuiApplication.screens():
                # macOS: unterhalb der Menüleiste, sonst ganz oben
                g = screen.availableGeometry() if sys.platform == "darwin" else screen.geometry()
                s = TopStripe()
                s.setGeometry(g.x(), g.y(), g.width(), STRIPE_BAR_PX + STRIPE_GLOW_PX)
                self._stripes.append(s)
        for s in self._stripes:
            s.set_color(color)
            s.show()
            s.raise_()

    def set_color(self, color: str) -> None:
        if self._stripes:
            self._color = color
            for s in self._stripes:
                s.set_color(color)

    def hide(self) -> None:
        for s in self._stripes:
            s.hide()
            s.deleteLater()
        self._stripes = []


# --------------------------------------------------------------------------- #
# Hauptfenster
# --------------------------------------------------------------------------- #

MONO_FAMILIES = ["JetBrains Mono", "Cascadia Mono", "SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono",
                 "Liberation Mono", "Courier New"]
HUD = "#5EE7FF"   # Grundfarbe der HUD-Elemente


def mono_font(px: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont()
    f.setFamilies(MONO_FAMILIES)
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


def _qc(color, alpha: float) -> QColor:
    c = QColor(color)
    c.setAlphaF(max(0.0, min(1.0, alpha)))
    return c


class PreviewWidget(QWidget):
    """Live-Vorschau im HUD-Stil: Gesichtsnetz, Augen-Tracker, Blickstrahl und Datenfelder.
    Standardmässig aus (spart Energie); über den Knopf «Live-Ansicht» einblendbar."""

    live_toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._img: QImage | None = None
        self._faces: list = []
        self._message = "Kamera ist aus"
        self._fps = 0.0
        self._far = False
        self._t0 = time.monotonic()
        self._offer_live = False  # Schutz läuft, Live-Ansicht ist aus -> grossen Knopf zeigen

        self.live_btn = QPushButton("◉  LIVE-ANSICHT", self)
        self.live_btn.setObjectName("live")
        self.live_btn.setFont(mono_font(18, QFont.Weight.Bold))
        self.live_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.live_btn.setFixedSize(280, 68)
        self.live_btn.clicked.connect(lambda: self.live_toggled.emit(True))
        self.live_btn.hide()

        self.off_btn = QPushButton("LIVE-ANSICHT AUS  ✕", self)
        self.off_btn.setObjectName("liveoff")
        self.off_btn.setFont(mono_font(11, QFont.Weight.DemiBold))
        self.off_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.off_btn.adjustSize()
        self.off_btn.clicked.connect(lambda: self.live_toggled.emit(False))
        self.off_btn.hide()

    def set_live_state(self, active: bool, live: bool) -> None:
        """active: Sicherheitscheck läuft; live: Live-Ansicht eingeschaltet."""
        self._offer_live = active and not live
        self.live_btn.setVisible(self._offer_live)
        self.off_btn.setVisible(active and live)
        if not live:
            self._img, self._faces = None, []
        self._place_buttons()
        self.update()

    def _place_buttons(self) -> None:
        self.off_btn.adjustSize()
        c = self.rect().center()
        self.live_btn.move(c.x() - self.live_btn.width() // 2, c.y() - self.live_btn.height() // 2)
        self.off_btn.move(c.x() - self.off_btn.width() // 2, 16)

    def resizeEvent(self, e):
        self._place_buttons()
        super().resizeEvent(e)

    def set_frame(self, img: QImage, faces: list) -> None:
        self._img, self._faces = img, faces
        self.update()

    def set_meta(self, fps: float, far: bool = False) -> None:
        self._fps = fps
        self._far = far

    def clear(self, message: str) -> None:
        self._img, self._faces, self._message = None, [], message
        self.update()

    # ---------- Hilfen ----------
    @staticmethod
    def _brackets(p: QPainter, r: QRectF, length: float, color: QColor, width: float = 2.0) -> None:
        p.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
        x0, y0, x1, y1 = r.left(), r.top(), r.right(), r.bottom()
        for (ax, ay, bx, by) in ((x0, y0, x0 + length, y0), (x0, y0, x0, y0 + length),
                                 (x1, y0, x1 - length, y0), (x1, y0, x1, y0 + length),
                                 (x0, y1, x0 + length, y1), (x0, y1, x0, y1 - length),
                                 (x1, y1, x1 - length, y1), (x1, y1, x1, y1 - length)):
            p.drawLine(QPointF(ax, ay), QPointF(bx, by))

    # ---------- Zeichnen ----------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        clip = QPainterPath()
        clip.addRoundedRect(r, 18, 18)
        p.fillPath(clip, QColor("#05080D"))
        p.setClipPath(clip)
        phase = time.monotonic() - self._t0

        if self._img is None:
            self._paint_idle(p, r, phase)
        else:
            iw, ih = self._img.width(), self._img.height()
            s = max(r.width() / iw, r.height() / ih)
            dw, dh = iw * s, ih * s
            ox, oy = r.x() + (r.width() - dw) / 2, r.y() + (r.height() - dh) / 2
            p.drawImage(QRectF(ox, oy, dw, dh), self._img)
            p.fillRect(r, QColor(0, 12, 22, 70))  # leichte Tönung für den HUD-Look

            def M(pt):
                return QPointF(ox + pt[0] * dw, oy + pt[1] * dh)

            self._paint_grid(p, r, phase)
            boxes = [self._face_box(f, M) for f in self._faces]
            # Flächen, die Datenfelder nicht verdecken sollen (bei Zielerfassung inkl. Ring)
            self._occupied = [self._face_zone(f, b) for f, b in zip(self._faces, boxes)]
            for f, box in zip(self._faces, boxes):
                self._paint_face(p, r, f, M, dw, phase, box)
            self._paint_frame(p, r, phase)

        p.setClipping(False)
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r, 18, 18)

    def _paint_idle(self, p: QPainter, r: QRectF, phase: float) -> None:
        self._paint_grid(p, r, phase, strong=False)
        c = r.center()
        if self._offer_live:
            p.setPen(_qc(GREEN, 0.9))
            p.setFont(mono_font(12, QFont.Weight.Bold))
            p.drawText(QRectF(r.x(), c.y() - 80, r.width(), 20), _qflags(Qt.AlignmentFlag.AlignCenter),
                       "● SCHUTZ AKTIV")
            p.setPen(_qc(HUD, 0.6))
            p.setFont(mono_font(11))
            p.drawText(QRectF(r.x() + 20, c.y() + 46, r.width() - 40, 36),
                       _qflags(Qt.AlignmentFlag.AlignHCenter, Qt.AlignmentFlag.AlignTop, Qt.TextFlag.TextWordWrap),
                       "Live-Ansicht ist aus und spart Energie.\nDie Erkennung läuft trotzdem weiter.")
            return
        p.setPen(QPen(_qc(HUD, 0.25), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, 46, 46)
        draw_eye(p, c.x(), c.y(), 58, _qc(HUD, 0.18), QColor("#05080D"))
        p.setPen(_qc(HUD, 0.7))
        p.setFont(mono_font(13, QFont.Weight.DemiBold))
        p.drawText(QRectF(r.x(), c.y() + 56, r.width(), 24), _qflags(Qt.AlignmentFlag.AlignCenter),
                   self._message.upper())

    def _paint_grid(self, p: QPainter, r: QRectF, phase: float, strong: bool = True) -> None:
        p.setPen(QPen(_qc(HUD, 0.05 if strong else 0.04), 1))
        step = 32
        x = r.x() + step
        while x < r.right():
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            x += step
        y = r.y() + step
        while y < r.bottom():
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            y += step
        # wandernde Scan-Linie
        sy = r.y() + ((phase % 3.2) / 3.2) * (r.height() + 80) - 40
        g = QLinearGradient(0, sy - 40, 0, sy)
        g.setColorAt(0, _qc(HUD, 0.0))
        g.setColorAt(1, _qc(HUD, 0.10 if strong else 0.06))
        p.fillRect(QRectF(r.x(), sy - 40, r.width(), 40), g)
        p.setPen(QPen(_qc(HUD, 0.35 if strong else 0.2), 1))
        p.drawLine(QPointF(r.left(), sy), QPointF(r.right(), sy))

    def _paint_frame(self, p: QPainter, r: QRectF, phase: float) -> None:
        inner = r.adjusted(12, 12, -12, -12)
        self._brackets(p, inner, 22, _qc(HUD, 0.75), 1.6)
        p.setFont(mono_font(11, QFont.Weight.DemiBold))
        blink = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(phase * 5))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_qc(RED, blink))
        p.drawEllipse(QPointF(inner.x() + 14, inner.y() + 16), 3.5, 3.5)
        p.setPen(_qc(HUD, 0.9))
        top = QRectF(inner.x() + 24, inner.y() + 7, inner.width() - 48, 18)
        p.drawText(top, _qflags(Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignVCenter), "LIVE-ANALYSE")
        p.drawText(top, _qflags(Qt.AlignmentFlag.AlignRight, Qt.AlignmentFlag.AlignVCenter),
                   datetime.now().strftime("%H:%M:%S"))
        bottom = QRectF(inner.x() + 10, inner.bottom() - 25, inner.width() - 20, 18)
        n = len(self._faces)
        p.drawText(bottom, _qflags(Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignVCenter),
                   f"PERSONEN {n:02d}  ·  {self._fps:4.1f} FPS")
        p.setPen(_qc(HUD, 0.55))
        p.drawText(bottom, _qflags(Qt.AlignmentFlag.AlignRight, Qt.AlignmentFlag.AlignVCenter),
                   "IRIS-TRACKING · FERN-ERKENNUNG" if self._far else "IRIS-TRACKING · LOKAL")
        if n == 0:
            p.setPen(_qc(HUD, 0.5 + 0.5 * math.sin(phase * 3)))
            p.setFont(mono_font(13, QFont.Weight.DemiBold))
            p.drawText(r, _qflags(Qt.AlignmentFlag.AlignCenter), "SUCHE GESICHTER …")

    @staticmethod
    def _face_zone(f: dict, box: QRectF) -> QRectF:
        if f["looking"] and f["role"] != "owner":
            c = box.center()
            rad = max(box.width(), box.height()) * 0.62 + 4
            return QRectF(c.x() - rad, c.y() - rad, 2 * rad, 2 * rad)
        return QRectF(box)

    @staticmethod
    def _face_box(f: dict, M) -> QRectF:
        x0, y0, x1, y1 = f["box"]
        return QRectF(M((x0, y0)), M((x1, y1))).adjusted(-14, -18, 14, 12)

    def _paint_face(self, p: QPainter, r: QRectF, f: dict, M, dw: float, phase: float, box: QRectF) -> None:
        owner = f["role"] == "owner"
        looking = f["looking"] and not owner
        color = QColor(GREEN if owner else (RED if looking else AMBER))

        # Gesichtsnetz und Umriss
        if "mesh" in f:
            p.setPen(QPen(_qc(color, 0.45), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPoints(QPolygonF([M(pt) for pt in f["mesh"]]))
        if "oval" in f:
            p.setPen(QPen(_qc(color, 0.35), 1.0, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(QPolygonF([M(pt) for pt in f["oval"]]))

        # Eck-Markierungen
        self._brackets(p, box, max(12.0, min(box.width(), box.height()) * 0.22), color, 2.2)

        # Augen-Tracker
        if "eyes" in f:
            p.setPen(QPen(_qc(HUD, 0.9), 1.3))
            p.setBrush(_qc(HUD, 0.07))
            for contour in f["eyes"]:
                p.drawPolygon(QPolygonF([M(pt) for pt in contour]))
        centers = []
        if "irises" in f:
            for (pt, rad) in f["irises"]:
                c = M(pt)
                centers.append(c)
                rr = max(3.0, rad * dw)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(_qc(HUD, 0.95), 1.3))
                p.drawEllipse(c, rr, rr)
                tick = rr * 0.9
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p.drawLine(QPointF(c.x() + dx * (rr + 2), c.y() + dy * (rr + 2)),
                               QPointF(c.x() + dx * (rr + 2 + tick), c.y() + dy * (rr + 2 + tick)))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(_qc("#FFFFFF", 0.95))
                p.drawEllipse(c, 1.6, 1.6)

        if not centers and "pts5" in f:  # sehr weit weg: nur Grobpunkte (Augen, Nase, Mund)
            p.setPen(QPen(_qc(HUD, 0.9), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for i, pt in enumerate(f["pts5"]):
                c = M(pt)
                if i < 2:
                    centers.append(c)
                    p.drawEllipse(c, 3.5, 3.5)
                p.drawLine(QPointF(c.x() - 5, c.y()), QPointF(c.x() + 5, c.y()))
                p.drawLine(QPointF(c.x(), c.y() - 5), QPointF(c.x(), c.y() + 5))

        # Blickstrahl: wohin die Augen bzw. der Kopf zeigen
        if centers and "gaze" in f:
            origin = QPointF(sum(c.x() for c in centers) / len(centers), sum(c.y() for c in centers) / len(centers))
            gx, gy = f["gaze"]
            mag = math.hypot(gx, gy)
            if mag < 0.16:
                # schaut direkt in die Kamera: pulsierender Zielpunkt
                pr = 7 + 4 * (0.5 + 0.5 * math.sin(phase * 6))
                p.setPen(QPen(_qc(color, 0.9), 1.4))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(origin, pr, pr)
            else:
                length = box.width() * 1.1 * min(1.0, mag * 1.4)
                end = QPointF(origin.x() + gx / mag * length, origin.y() + gy / mag * length)
                grad = QLinearGradient(origin, end)
                grad.setColorAt(0, _qc(color, 0.95))
                grad.setColorAt(1, _qc(color, 0.0))
                p.setPen(QPen(QBrush(grad), 2.0))
                p.drawLine(origin, end)
                mid = QPointF((origin.x() + end.x()) / 2, (origin.y() + end.y()) / 2)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(_qc(color, 0.8))
                p.drawEllipse(mid, 2.2, 2.2)

        # Zielerfassung, wenn eine fremde Person auf den Bildschirm schaut
        if looking:
            c = box.center()
            rad = max(box.width(), box.height()) * 0.62
            p.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(_qc(RED, 0.85), 1.6)
            p.setPen(pen)
            rot = (phase * 90) % 360
            ring = QRectF(c.x() - rad, c.y() - rad, 2 * rad, 2 * rad)
            for k in range(4):
                p.drawArc(ring, int((rot + k * 90) * 16), int(50 * 16))
            inner = ring.adjusted(8, 8, -8, -8)
            p.setPen(QPen(_qc(RED, 0.45), 1.0))
            for k in range(4):
                p.drawArc(inner, int((-rot * 1.5 + k * 90 + 30) * 16), int(30 * 16))

        # Kennzeichnung
        tag = f"DU · ID {f['id']:02d}" if owner else f"ZIEL {f['id']:02d}"
        if f.get("far") and not owner:
            tag += " · FERN"
        chip_color = QColor(color)
        if looking:
            tag += "  ◉ BLICK ERFASST"
            if (phase * 2.5) % 1.0 >= 0.65:  # blinkt
                chip_color = _qc(RED, 0.55)
        p.setFont(mono_font(11, QFont.Weight.Bold))
        fm = p.fontMetrics()
        tw, th = fm.horizontalAdvance(tag) + 14, fm.height() + 6
        chip = QRectF(box.x(), box.y() - th - 5, tw, th)
        if chip.y() < r.y() + 34:
            chip.moveTop(box.bottom() + 5)
        chip.moveLeft(max(r.left() + 12, min(chip.left(), r.right() - 12 - tw)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(chip_color)
        p.drawRect(chip)
        p.setPen(QColor("#05080D"))
        p.drawText(chip, _qflags(Qt.AlignmentFlag.AlignCenter), tag)

        self._paint_panel(p, r, box, f, color, looking)

    def _paint_panel(self, p: QPainter, r: QRectF, box: QRectF, f: dict, color: QColor, looking: bool) -> None:
        def deg(v, pos, neg):
            if v is None:
                return "–"
            return f"{pos if v >= 0 else neg} {abs(v):2.0f}°"

        rows = []
        rows.append(("ENTF", f"≈ {f['distance']:.1f} m" if f.get("distance") else "–", None))
        rows.append(("KOPF", f"{deg(f.get('yaw'), '→', '←')}  {deg(f.get('pitch'), '↓', '↑')}", None))
        rows.append(("NEIG", deg(f.get("roll"), "↻", "↺"), None))
        if f["role"] == "owner":
            rows.append(("BLICK", "BILDSCHIRM" if f["looking"] else "ABGEWANDT", None))
        else:
            if looking:
                rows.append(("BLICK", "AUF BILDSCHIRM", RED))
            elif f.get("looking_raw") and (f.get("dwell") or 0) > 0:
                since = f.get("looking_since") or 0.0
                rows.append(("BLICK", f"PRÜFE {min(since, f['dwell']):.1f}/{f['dwell']:.1f} S", AMBER))
            else:
                rows.append(("BLICK", "ABGEWANDT", None))
            if (f.get("strict") or 0) > 0.05:
                rows.append(("MODUS", f"STRENG {f['strict'] * 100:3.0f} %", None))
        ew = f.get("eye_weight") or 0.0
        if not f.get("has_mesh", True):
            src = "KOPF · 5 PUNKTE"
        elif ew >= 0.95:
            src = "IRIS + KOPF"
        elif ew > 0.05:
            src = f"IRIS {ew * 100:2.0f}% · KOPF"
        else:
            src = "KOPFHALTUNG"
        rows.append(("QUELLE", src, None))
        eo = f.get("eyes_open")
        rows.append(("AUGEN", f"{eo * 100:3.0f} % OFFEN" if eo is not None else "–", None))
        br = f.get("blink_rate")
        rows.append(("LIDSCHL", f"{br:3.0f} /MIN" if br is not None else "MESSE …", None))
        rows.append(("MUND", "SPRICHT" if f.get("talking") else "RUHIG", AMBER if f.get("talking") else None))
        if f.get("smiling") is not None:
            rows.append(("MIMIK", "LÄCHELT" if f["smiling"] else "NEUTRAL", None))
        att = f.get("attention")
        rows.append(("FOKUS", f"{att * 100:3.0f} % / 10 S" if att is not None else "–", None))
        if f.get("looking_since") and f["role"] != "owner":
            rows.append(("DAUER", f"{f['looking_since']:4.1f} S", RED))

        p.setFont(mono_font(10, QFont.Weight.DemiBold))
        fm = p.fontMetrics()
        lh = fm.height() + 1
        label_w = fm.horizontalAdvance("LIDSCHL ") + 4
        val_w = max(fm.horizontalAdvance(v) for _, v, _ in rows)
        pw, ph = label_w + val_w + 20, lh * len(rows) + 12
        zone0 = self._face_zone(f, box)
        top_y = max(r.top() + 34, min(box.top(), r.bottom() - ph - 34))
        mid_x = box.center().x() - pw / 2
        candidates = [
            QRectF(zone0.right() + 12, top_y, pw, ph),                 # rechts
            QRectF(zone0.left() - 12 - pw, top_y, pw, ph),             # links
            QRectF(mid_x, zone0.bottom() + 28, pw, ph),                # unten
            QRectF(mid_x, zone0.top() - 32 - ph, pw, ph),              # oben
        ]
        area = r.adjusted(8, 30, -8, -30)
        zone = self._face_zone(f, box)
        others = [o for o in getattr(self, "_occupied", []) if o != zone]
        panel = None
        for cand in candidates:
            if area.contains(cand) and not any(cand.intersects(o) for o in others):
                panel = cand
                break
        if panel is None:  # kein freier Platz: rechts bzw. links innerhalb des Bildes
            panel = candidates[0] if candidates[0].right() <= area.right() else candidates[1]
            panel.moveLeft(max(area.left(), min(panel.left(), area.right() - pw)))
        if hasattr(self, "_occupied"):
            self._occupied.append(panel)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(4, 9, 15, 200))
        p.drawRect(panel)
        p.setBrush(color)
        p.drawRect(QRectF(panel.x(), panel.y(), 2.5, panel.height()))
        # Verbindungslinie zum Gesicht
        p.setPen(QPen(_qc(color, 0.5), 1, Qt.PenStyle.DotLine))
        pc, bc = panel.center(), box.center()
        if panel.left() >= box.right():
            p.drawLine(QPointF(box.right(), box.top() + 10), QPointF(panel.left(), panel.top() + 10))
        elif panel.right() <= box.left():
            p.drawLine(QPointF(box.left(), box.top() + 10), QPointF(panel.right(), panel.top() + 10))
        elif pc.y() > bc.y():
            p.drawLine(QPointF(bc.x(), box.bottom()), QPointF(pc.x(), panel.top()))
        else:
            p.drawLine(QPointF(bc.x(), box.top()), QPointF(pc.x(), panel.bottom()))

        y = panel.y() + 6
        for label, value, vcol in rows:
            p.setPen(_qc(HUD, 0.6))
            p.drawText(QRectF(panel.x() + 10, y, label_w, lh),
                       _qflags(Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignVCenter), label)
            p.setPen(QColor(vcol) if vcol else QColor("#EAF6FF"))
            p.drawText(QRectF(panel.x() + 10 + label_w, y, val_w + 4, lh),
                       _qflags(Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignVCenter), value)
            y += lh


class ShieldBadge(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(52, 52)
        self._color = ACCENT

    def set_color(self, color: str) -> None:
        self._color = color
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.drawPixmap(0, 0, make_icon(self._color).pixmap(52, 52))


class MainWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        self.camera_index = camera_index
        self.worker: DetectionWorker | None = None
        self._retired: list[DetectionWorker] = []
        self._quitting = False
        self.overlays = OverlayManager()
        self.stripes = StripeManager()
        self.live_view = False  # Live-Ansicht standardmässig aus (spart Energie)

        self.setObjectName("main")
        self.setWindowTitle("Sicherheitscheck")
        self.resize(860, 920)
        self.setMinimumSize(620, 640)
        self._positioned = False
        self._build_ui()
        self._build_tray()
        if self.tray is not None and sys.platform != "darwin":
            # Nur im Infobereich: kein Eintrag bei den Apps in der Taskleiste
            # (macOS: stattdessen kein Dock-Symbol, siehe main()).
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self._set_idle()

    # ---------- Aufbau ----------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 26, 28, 24)
        root.setSpacing(16)

        header = QHBoxLayout()
        header.setSpacing(14)
        self.badge = ShieldBadge()
        header.addWidget(self.badge)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("Sicherheitscheck")
        title.setObjectName("title")
        subtitle = QLabel("BLICKSCHUTZ  ·  IRIS-TRACKING  ·  100 % LOKAL")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        root.addLayout(header)

        status_row = QHBoxLayout()
        self.status_pill = QLabel()
        self.status_pill.setObjectName("pill")
        status_row.addWidget(self.status_pill, 0)
        status_row.addStretch(1)
        root.addLayout(status_row)

        self.detail = QLabel()
        self.detail.setObjectName("detail")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.detail)

        self.preview = PreviewWidget()
        self.preview.live_toggled.connect(self._set_live)
        root.addWidget(self.preview, 1)

        stats = QHBoxLayout()
        stats.setSpacing(10)
        self.stat_people = QLabel()
        self.stat_look = QLabel()
        self.stat_fps = QLabel()
        for s in (self.stat_people, self.stat_look, self.stat_fps):
            s.setObjectName("stat")
            s.setTextFormat(Qt.TextFormat.RichText)
            stats.addWidget(s, 1)
        root.addLayout(stats)

        self.button = QPushButton("Sicherheitscheck starten")
        self.button.setObjectName("primary")
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.setMinimumHeight(56)
        self.button.clicked.connect(self.toggle)
        root.addWidget(self.button)

        foot = QLabel("Das Programm lebt unten rechts im Infobereich: grün = aktiv, orange = geöffnet, "
                      "rot = nicht aktiv. ✕ blendet nur das Fenster aus, die Erkennung läuft weiter. "
                      "Ganz beenden: Rechtsklick auf das Symbol → «Programm beenden». Keine Töne. "
                      "Alle Bilder werden ausschliesslich lokal auf diesem Gerät ausgewertet "
                      "und nicht gespeichert.")
        foot.setObjectName("foot")
        foot.setWordWrap(True)
        root.addWidget(foot)

        self.setStyleSheet(f"""
            QWidget#main {{ background: {BG}; }}
            QLabel {{ color: {TEXT}; background: transparent; }}
            QLabel#title {{ font-size: 24px; font-weight: 700; }}
            QLabel#subtitle {{ color: {HUD}; font-size: 11px; letter-spacing: 2px; }}
            QLabel#pill {{
                background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px;
                padding: 6px 14px; font-size: 13px; font-weight: 600;
            }}
            QLabel#detail {{ color: {MUTED}; font-size: 13px; }}
            QLabel#stat {{
                background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
                padding: 10px 14px; font-size: 12px;
            }}
            QLabel#foot {{ color: #5E6679; font-size: 11px; }}
            QPushButton#primary {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT}, stop:1 {ACCENT_2});
                color: white; border: none; border-radius: 14px; font-size: 16px; font-weight: 700;
            }}
            QPushButton#primary:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7F9BFF, stop:1 #AA82FF);
            }}
            QPushButton#primary:pressed {{ background: {ACCENT}; }}
            QPushButton#primary[mode="stop"] {{
                background: transparent; color: {RED}; border: 1.5px solid {RED};
            }}
            QPushButton#primary[mode="stop"]:hover {{ background: rgba(255, 77, 106, 0.12); }}
            QPushButton#live {{
                background: rgba(94, 231, 255, 0.10); color: {HUD};
                border: 1.5px solid {HUD}; border-radius: 16px;
            }}
            QPushButton#live:hover {{ background: rgba(94, 231, 255, 0.22); }}
            QPushButton#live:pressed {{ background: rgba(94, 231, 255, 0.32); }}
            QPushButton#liveoff {{
                background: rgba(4, 9, 15, 0.78); color: {HUD};
                border: 1px solid rgba(94, 231, 255, 0.5); border-radius: 11px; padding: 4px 12px;
            }}
            QPushButton#liveoff:hover {{ background: rgba(94, 231, 255, 0.18); }}
        """)

    def _build_tray(self) -> None:
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(make_icon(RED), self)
        menu = QMenu()
        act_open = QAction("Fenster öffnen", self)
        act_open.triggered.connect(self.show_window)
        self.act_toggle = QAction("Sicherheitscheck starten", self)
        self.act_toggle.triggered.connect(self.toggle)
        act_quit = QAction("Programm beenden", self)
        act_quit.triggered.connect(self.quit_app)
        menu.addAction(act_open)
        menu.addAction(self.act_toggle)
        menu.addSeparator()
        self.act_autostart = None
        if autostart_tools() is not None:
            self.act_autostart = QAction("Beim Anmelden automatisch starten", self)
            self.act_autostart.setCheckable(True)
            self.act_autostart.setChecked(autostart_tools().is_autostart_enabled())
            self.act_autostart.toggled.connect(self._toggle_autostart)
            menu.addAction(self.act_autostart)
            menu.addSeparator()
        menu.addAction(act_quit)
        self._tray_menu = menu
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self._update_tray()
        self.tray.show()

    def _update_tray(self) -> None:
        """Grün = Sicherheitscheck aktiv, orange = Fenster offen aber nicht aktiv, rot = nicht aktiv."""
        if self.tray is None:
            return
        if self.worker is not None:
            color, text = GREEN, "aktiv"
        elif self.isVisible():
            color, text = AMBER, "geöffnet, nicht aktiv"
        else:
            color, text = RED, "nicht aktiv"
        self.tray.setIcon(make_icon(color))
        self.tray.setToolTip(f"Sicherheitscheck · {text}")

    def _toggle_autostart(self, enabled: bool) -> None:
        tools = autostart_tools()
        try:
            tools.set_autostart(enabled)
        except Exception:
            pass
        actual = tools.is_autostart_enabled()
        if actual != enabled and self.act_autostart is not None:
            self.act_autostart.blockSignals(True)
            self.act_autostart.setChecked(actual)
            self.act_autostart.blockSignals(False)

    # ---------- Darstellung ----------
    def _set_pill(self, text: str, color: str) -> None:
        self.status_pill.setText(f'<span style="color:{color}; font-size:15px;">●</span>&nbsp;&nbsp;{text}')
        self.badge.set_color(color if color != MUTED else ACCENT)

    def _set_stats(self, people: str, look: str, fps: str) -> None:
        def cell(label, value):
            mono = ", ".join(f"'{m}'" for m in MONO_FAMILIES)
            return (f'<span style="color:{MUTED}; font-size:11px; letter-spacing:1px">{label.upper()}</span><br>'
                    f'<span style="font-family:{mono}; font-size:19px; font-weight:700; color:{HUD}">{value}</span>')
        self.stat_people.setText(cell("Personen im Bild", people))
        self.stat_look.setText(cell("Blick auf Bildschirm", look))
        self.stat_fps.setText(cell("Bildrate", fps))

    def _set_button(self, running: bool) -> None:
        self.button.setText("Sicherheitscheck beenden" if running else "Sicherheitscheck starten")
        self.button.setProperty("mode", "stop" if running else "start")
        self.button.style().unpolish(self.button)
        self.button.style().polish(self.button)
        if self.tray is not None:
            self.act_toggle.setText(self.button.text())
        self._update_tray()

    def _set_idle(self, error: str | None = None) -> None:
        self.stripes.hide()
        self._set_button(False)
        self._set_stats("–", "–", "–")
        self.preview.clear("Kamera ist aus")
        self.preview.set_live_state(False, self.live_view)
        if error:
            self._set_pill("Fehler", RED)
            self.detail.setText(error)
        else:
            self._set_pill("Bereit", MUTED)
            self.detail.setText("Klicke auf «Sicherheitscheck starten», um die Webcam-Überwachung zu aktivieren.")

    def _preview_visible(self) -> bool:
        return self.live_view and self.isVisible() and not self.isMinimized()

    def _set_live(self, on: bool) -> None:
        self.live_view = on
        self._sync_preview()
        if not on:
            self.preview.clear("Live-Ansicht aus")
        elif self.worker is not None:
            self.preview.clear("Live-Ansicht wird geladen …")
        self.preview.set_live_state(self.worker is not None, on)

    def _sync_preview(self) -> None:
        if self.worker is not None:
            self.worker.preview_enabled = self._preview_visible()

    # ---------- Steuerung ----------
    def toggle(self) -> None:
        if self.worker is None:
            self.start_check()
        else:
            self.stop_check()

    def start_check(self) -> None:
        if self.worker is not None:
            return
        w = DetectionWorker(self.camera_index)
        w.preview_enabled = self._preview_visible()
        w.status.connect(lambda d, w=w: self._on_status(w, d))
        w.preview.connect(lambda img, faces, w=w: self._on_preview(w, img, faces))
        w.alert.connect(lambda looking, w=w: self._on_alert(w, looking))
        w.alert_update.connect(lambda _l, w=w: self._on_alert_update(w))
        w.cleared.connect(lambda w=w: self._on_cleared(w))
        w.failed.connect(lambda msg, w=w: self._on_failed(w, msg))
        w.finished.connect(lambda w=w: self._on_finished(w))
        self.worker = w
        self._set_button(True)
        self._set_pill("Wird gestartet …", AMBER)
        self.detail.setText("Die Webcam wird vorbereitet …")
        self.preview.clear("Kamera wird gestartet …")
        self.preview.set_live_state(True, self.live_view)
        w.start()

    def stop_check(self) -> None:
        w = self.worker
        if w is None:
            return
        self.worker = None
        w.stop()
        self._retired.append(w)
        self.overlays.hide_all()
        self._set_idle()

    def _on_finished(self, w: DetectionWorker) -> None:
        if w in self._retired:
            self._retired.remove(w)
        if w is self.worker:  # unerwartet beendet
            self.worker = None
            self._set_idle()

    # ---------- Signale der Erkennung ----------
    def _on_status(self, w, d: dict) -> None:
        if w is not self.worker:
            return
        phase = d.get("phase")
        if phase == "starting":
            self.detail.setText(d.get("text", ""))
        elif phase == "camera_lost":
            self._set_pill("Webcam getrennt", AMBER)
            self.detail.setText("Die Verbindung zur Webcam wurde unterbrochen. Neuer Versuch läuft …")
            self.preview.clear("Warte auf Webcam …")
        elif phase == "active":
            faces, others = d["faces"], d["others"]
            if d["intruder"] and d["looking"]:
                self._set_pill("Zweite Person schaut auf deinen Bildschirm", RED)
                self.detail.setText("Die Warn-Insel erscheint pro Besuch einmal und schliesst sich nach "
                                    "5 Sekunden oder sobald die Person weg ist.")
                self.stripes.set_color(RED)
            elif d["intruder"]:
                self._set_pill("Zweite Person im Bild", AMBER)
                self.detail.setText("Oben am Bildschirm erscheint ein oranger Streifen. Die Warn-Insel kommt "
                                    "erst, wenn die Person auf deinen Bildschirm schaut.")
                self.stripes.set_color(AMBER)
            else:
                if faces == 0:
                    self._set_pill("Aktiv · niemand im Bild", GREEN)
                elif others == 0:
                    self._set_pill("Aktiv · nur du im Bild", GREEN)
                else:
                    self._set_pill("Aktiv · weitere Person wird geprüft", AMBER)
                self.detail.setText("Du kannst normal weiterarbeiten. Die Webcam wird laufend ausgewertet.")
            self.preview.set_meta(d["fps"], d.get("far", False))
            look = ("Ja" if d["looking"] else "Nein") if others > 0 else "–"
            self._set_stats(str(faces), look, f"{d['fps']:.0f} fps")

    def _on_preview(self, w, img: QImage, faces: list) -> None:
        if w is self.worker and self._preview_visible():
            self.preview.set_frame(img, faces)

    def _on_alert(self, w, looking: bool) -> None:
        """Zweite Person bestätigt: immer der Streifen, die Warn-Insel nur bei Blick auf den Bildschirm."""
        if w is not self.worker:
            return
        self.stripes.show(RED if looking else AMBER)
        if looking:
            self.overlays.show_alert(True)

    def _on_alert_update(self, w) -> None:
        """Die Person schaut erstmals während ihres Besuchs auf den Bildschirm -> Warn-Insel."""
        if w is self.worker:
            self.stripes.set_color(RED)
            self.overlays.show_alert(True)

    def _on_cleared(self, w) -> None:
        if w is self.worker:  # keine weitere Person mehr im Bild
            self.stripes.hide()
            self.overlays.hide_all()

    def _on_failed(self, w, msg: str) -> None:
        if w is not self.worker:
            return
        self.worker = None
        self._retired.append(w)
        self._set_idle(error=msg)
        if not self._preview_visible():
            self.show_window()  # Fehler lautlos sichtbar machen (keine Systembenachrichtigung)

    # ---------- Fenster & Tray ----------
    def _place_near_tray(self) -> None:
        """Beim ersten Öffnen unten rechts über dem Infobereich platzieren."""
        screen = None
        if self.tray is not None:
            g = self.tray.geometry()
            if g.isValid() and not g.isEmpty():
                screen = QGuiApplication.screenAt(g.center())
        screen = screen or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        w = min(self.width(), avail.width() - 24)
        h = min(self.height(), avail.height() - 24)
        self.resize(w, h)
        frame_extra = max(0, self.frameGeometry().height() - self.geometry().height())
        x = avail.right() - w - 12
        y = avail.bottom() - h - 12 - frame_extra
        if sys.platform == "darwin":  # Menüleiste ist oben
            y = avail.top() + 12
        self.move(max(avail.left(), x), max(avail.top(), y))
        self._positioned = True

    def show_window(self) -> None:
        if not self._positioned:
            self._place_near_tray()
        self.showNormal()
        self.raise_()
        self.activateWindow()
        if sys.platform == "darwin":
            try:
                from AppKit import NSApplication
                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            except Exception:
                pass

    def _on_tray_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            if self.isVisible() and self.isActiveWindow() and reason == QSystemTrayIcon.ActivationReason.Trigger:
                self.hide()  # zweiter Klick blendet das Fenster wieder aus
            else:
                self.show_window()

    def changeEvent(self, e):
        if e.type() == QEvent.Type.WindowStateChange:
            self._sync_preview()
        super().changeEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        self._sync_preview()
        self._update_tray()

    def hideEvent(self, e):
        super().hideEvent(e)
        self._sync_preview()
        self._update_tray()

    def closeEvent(self, e):
        # ✕ blendet das Fenster aus; das Programm bleibt unten rechts im Infobereich (Farbsymbol).
        # Ganz beenden: Rechtsklick auf das Symbol -> «Programm beenden».
        if self.tray is not None and not self._quitting:
            e.ignore()
            self.hide()
            return
        e.accept()
        self.quit_app()

    def quit_app(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        # Sicherheitsnetz: Falls irgendetwas hängt (z. B. die Webcam), wird der Prozess trotzdem beendet.
        guard = threading.Timer(6.0, lambda: os._exit(0))
        guard.daemon = True
        guard.start()
        if self.worker is not None:
            self.stop_check()
        self.overlays.hide_all(immediate=True)
        self.stripes.hide()
        if self.tray is not None:
            self.tray.hide()
        self.hide()
        for w in list(self._retired):
            w.wait(3000)  # Erkennung sauber beenden, damit die Webcam freigegeben wird
        QApplication.quit()


def autostart_tools():
    """Das Modul verknuepfung.py (Autostart ein/aus) oder None, falls nicht verfügbar."""
    try:
        import verknuepfung
        return verknuepfung if verknuepfung.autostart_supported() else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Start
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Sicherheitscheck – Schutz vor fremden Blicken")
    parser.add_argument("--kamera", type=int, default=0, help="Nummer der Webcam (Standard: 0)")
    parser.add_argument("--warnung-testen", action="store_true", help="zeigt Streifen und Warn-Insel als Vorschau")
    parser.add_argument("--tray", action="store_true", help="still im Infobereich starten (ohne Fenster)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C im Terminal beendet das Programm
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Sicherheitscheck.App")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("Sicherheitscheck")
    app.setQuitOnLastWindowClosed(False)
    app.setFont(ui_font(13))
    icon_file = Path(__file__).resolve().parent / "icon" / "sicherheitscheck.png"
    app.setWindowIcon(QIcon(str(icon_file)) if icon_file.is_file() else make_icon(ACCENT))
    if sys.platform == "darwin":
        try:  # kein Symbol im Dock, nur in der Menüleiste
            from AppKit import NSApplication
            NSApplication.sharedApplication().setActivationPolicy_(1)  # NSApplicationActivationPolicyAccessory
        except Exception:
            pass

    server = _single_instance_server()
    if server is None:  # läuft bereits: dort das Fenster öffnen und hier aufhören
        return

    win = MainWindow(args.kamera)

    def _on_connection():
        conn = server.nextPendingConnection()
        if conn is not None:
            conn.disconnected.connect(conn.deleteLater)
        win.show_window()

    server.newConnection.connect(_on_connection)
    if not (args.tray and win.tray is not None):
        win.show_window()
    if args.warnung_testen:
        QTimer.singleShot(600, lambda: win.stripes.show(AMBER))
        QTimer.singleShot(3600, lambda: (win.stripes.set_color(RED), win.overlays.show_alert(True)))
        QTimer.singleShot(10000, win.stripes.hide)
    code = app.exec()
    server.close()
    win.quit_app()
    # Hart beenden: garantiert, dass kein Teil des Programms im Hintergrund weiterläuft
    if sys.stdout is not None:  # unter pythonw (Windows) gibt es keine Konsole
        sys.stdout.flush()
    os._exit(code)


def _single_instance_server():
    """Sorgt dafür, dass das Programm nur einmal läuft. Liefert None, wenn es schon läuft."""
    name = "Sicherheitscheck-" + re.sub(r"[^A-Za-z0-9_-]", "_", getpass.getuser() or "user")
    probe = QLocalSocket()
    probe.connectToServer(name)
    if probe.waitForConnected(500):
        probe.write(b"show\n")
        probe.flush()
        probe.waitForBytesWritten(500)
        probe.disconnectFromServer()
        return None
    QLocalServer.removeServer(name)  # Überbleibsel eines abgestürzten Laufs entfernen
    server = QLocalServer()
    server.listen(name)
    return server


if __name__ == "__main__":
    main()
