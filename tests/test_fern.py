import os
import sys, math, io, contextlib, types
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erkennung as E
with contextlib.redirect_stdout(io.StringIO()):
    import test_augen as TA
W, H = 1280, 720
f = E.focal_px(W); CAM = np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]])
U = E.MODEL_UNITS_PER_MM
def rot(yaw=0, pitch=0): return TA.rot(yaw, pitch)
ok_all = True
def check(name, cond, info=""):
    global ok_all; ok_all &= bool(cond); print(("OK  " if cond else "FAIL"), name, info)

# 1) Nur 5 Grobpunkte (YuNet), Person 3,5 m entfernt
for dist_m, yaw, expect in ((3.5, 0, True), (3.5, 10, True), (3.5, 25, False), (3.5, 50, False), (5.0, 0, True), (5.0, -45, False)):  # weit weg: streng (15°)
    t = np.array([0.4, -0.3, dist_m]) * 1000 * U
    # Kopf so drehen, dass yaw relativ zur Kamera-Richtung gilt
    base_yaw = math.degrees(math.atan2(t[0], t[2]))
    R = rot(yaw + base_yaw)
    pts,_ = cv2.projectPoints(E.FIVE_POINT_MODEL, cv2.Rodrigues(R)[0], t, CAM, np.zeros(4))
    pts = pts.reshape(5,2)
    x0,y0 = pts.min(0)-8; x1,y1 = pts.max(0)+8
    fc = E.face_from_points5((x0,y0,x1-x0,y1-y0), pts, W, H)
    check(f"5 Punkte, {dist_m} m, Kopf {yaw:+d}°", fc.looking == expect,
          f"Gesicht {x1-x0:.0f}px  Kopf={fc.dev_h:+.1f}°  Entf={fc.distance_m:.1f} m  schaut={fc.looking}")

# 2) Volle Landmarks, aber weit weg (kleine Augen) -> Kopfhaltung entscheidet, strenge Toleranz
for yaw, expect in ((0, True), (10, True), (28, False), (40, False)):  # 4 m: strenger Modus
    t = (900, -500, 4000*U/1.0)
    lm = TA.synth(yaw, 0, t, 'kopf')            # Augen schauen geradeaus mit dem Kopf
    fc = E.analyze_landmarks(lm, W, H)
    check(f"Mesh weit weg, Kopf {yaw}°", fc.looking == expect,
          f"Augengewicht={fc.eye_weight:.2f}  Blick={fc.gaze_h:+.1f}°  schaut={fc.looking}")

# 3) Nah: Augen zählen voll
lm = TA.synth(40, 0, (800,-600,4500), 'kamera'); fc = E.analyze_landmarks(lm, W, H)
check("Nah, Kopf 40° weg, Augen zur Kamera", fc.looking and fc.eye_weight > 0.95, f"Augengewicht={fc.eye_weight:.2f}")

# 4) Ausschnitt + Rückrechnung (YuNet und Zoom-Landmarker simuliert)
frame = np.zeros((H, W, 3), np.uint8)
true_lm = TA.synth(10, 0, (5200, -1800, 4000*U), 'kamera')   # kleines Gesicht oben rechts
bx0, by0 = true_lm[[1,152,33,263,61,291]].min(0) - 10; bx1, by1 = true_lm[[1,152,33,263,61,291]].max(0) + 10
box = np.array([bx0, by0, bx1-bx0, by1-by0]); pts5 = np.array([true_lm[33], true_lm[263], true_lm[1], true_lm[61], true_lm[291]])
class FakeDet:
    def detect(self, img): return [(box, pts5, 0.95)]
captured = {}
class FakeLmk:
    def detect(self, image):
        # Landmarks normiert im Ausschnitt zurückgeben (wie MediaPipe)
        ox, oy, s = captured["crop"]
        pts = [types.SimpleNamespace(x=(p[0]-ox)/s, y=(p[1]-oy)/s) for p in true_lm]
        return types.SimpleNamespace(face_landmarks=[pts], face_blendshapes=[])
orig_crop = E._crop_square
def spy_crop(frame, cx, cy, side):
    r = orig_crop(frame, cx, cy, side); captured["crop"] = (r[1], r[2], r[3]); return r
E._crop_square = spy_crop
an = object.__new__(E.FaceAnalyzer)
an._mp = types.SimpleNamespace(Image=lambda image_format, data: None, ImageFormat=types.SimpleNamespace(SRGB=1))
an._far, an._crop_landmarker, an._far_cache, an._frame_no, an._last_scan = FakeDet(), FakeLmk(), [], 0, -10**9
far = an._far_faces(frame, [], W, H)
check("Fern-Gesicht gefunden", len(far) == 1)
if far:
    err = float(np.abs(far[0].lm - true_lm).max())
    check("Rückrechnung der Gesichtspunkte", err < 1e-6, f"max. Abweichung {err:.2e}px, far={far[0].far}, schaut={far[0].looking}")
# Ausschnitt am Bildrand
crop, ox, oy, s = orig_crop(frame, 5, 5, 100)
check("Ausschnitt am Rand", crop.shape[:2] == (100, 100) and (ox, oy) == (-45.0, -45.0), f"{crop.shape[:2]}, Ursprung ({ox},{oy})")
# bereits von MediaPipe gefundenes Gesicht wird nicht doppelt gezählt
near = E.analyze_landmarks(true_lm, W, H)
check("Kein Doppelzählen", an._far_faces(frame, [near], W, H) == [])
print("ALLE OK" if ok_all else "FEHLER")

# 5) Suchrhythmus: auch ohne ferne Personen nur jedes n-te Bild komplett absuchen
calls = {"n": 0}
class CountDet:
    def detect(self, img): calls["n"] += 1; return []
an2 = object.__new__(E.FaceAnalyzer)
an2._mp = an._mp; an2._far, an2._crop_landmarker, an2._far_cache, an2._frame_no, an2._last_scan = CountDet(), None, [], 0, -10**9
for _ in range(30): an2._far_faces(frame, [], W, H)
check("Suchrhythmus (30 Bilder)", calls["n"] == 10, f"{calls['n']} Suchläufe (erwartet 10 bei jedem {E.YUNET_EVERY_N}. Bild)")

# 6) Ersatzverfahren mit Kacheln (MediaPipe simuliert): findet ein Gesicht in überlappenden Kacheln nur einmal
class TileLmk:
    def detect(self, image):
        tw, th = int(W*0.6), int(H*0.6)
        x0 = 0 if captured_tiles["i"] in (0,1) else W-tw; y0 = 0 if captured_tiles["i"] in (0,2) else H-th
        captured_tiles["i"] += 1
        cx, cy = 640, 300   # Gesicht in der Mitte: liegt in allen 4 Kacheln
        if not (x0 <= cx < x0+tw and y0 <= cy < y0+th): return types.SimpleNamespace(face_landmarks=[])
        pts = [types.SimpleNamespace(x=(p[0]-x0)/tw, y=(p[1]-y0)/th) for p in TA.synth(0,0,(0,-300*U,3000*U),'kamera') * 0 + np.array([cx,cy]) + np.random.default_rng(1).normal(0,8,(478,2))]
        return types.SimpleNamespace(face_landmarks=[pts])
captured_tiles = {"i": 0}
td = E.TileFarDetector(TileLmk(), types.SimpleNamespace(Image=lambda image_format, data: None, ImageFormat=types.SimpleNamespace(SRGB=1)))
found = td.detect(np.zeros((H, W, 3), np.uint8))
check("Kacheln: Gesicht genau einmal", len(found) == 1, f"{len(found)} Treffer aus 4 Kacheln")
print("ALLE OK" if ok_all else "FEHLER")
