import os
import sys, math
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import erkennung as E

W, H = 640, 360
f = E.focal_px(W)
CAM = np.array([[f,0,W/2],[0,f,H/2],[0,0,1]])
IDX = [1,152,33,263,61,291]

def rot(yaw=0, pitch=0):
    # Grundhaltung: Gesicht schaut zur Kamera (180° um x), dann Drehung in Kamerakoordinaten
    base = cv2.Rodrigues(np.array([math.pi,0,0.]))[0]
    Ry = cv2.Rodrigues(np.array([0, math.radians(yaw), 0.]))[0]
    Rx = cv2.Rodrigues(np.array([math.radians(pitch), 0, 0.]))[0]
    return Ry @ Rx @ base

def synth(R, t, noise=0.0, rng=None):
    pts, _ = cv2.projectPoints(E.MODEL_POINTS, cv2.Rodrigues(R)[0], np.array(t, float), CAM, np.zeros(4))
    pts = pts.reshape(-1, 2)
    if noise: pts = pts + rng.normal(0, noise, pts.shape)
    lm = np.zeros((478, 2))
    for i, p in zip(IDX, pts): lm[i] = p
    return lm

def run(name, yaw, pitch, t, expect_h, expect_v, tol=3.0, noise=0.0):
    rng = np.random.default_rng(1)
    lm = synth(rot(yaw, pitch), t, noise, rng)
    dh, dv = E.head_deviation(lm, W, H)
    ok = abs(dh-expect_h) < tol and abs(dv-expect_v) < tol
    print(f"{'OK ' if ok else 'FAIL'} {name:45s} dev_h={dh:6.1f} (erw. {expect_h:6.1f})  dev_v={dv:6.1f} (erw. {expect_v:6.1f})  schaut={E.is_looking(dh,dv)}")
    return ok

res = []
res.append(run("Mitte, schaut zur Kamera", 0, 0, (0,0,5000), 0, 0))
off = math.degrees(math.atan2(1500, 5000))
res.append(run("rechts im Bild, schaut geradeaus (parallel)", 0, 0, (1500,0,5000), off, 0))
# rechts im Bild und zur Kamera gedreht: Gier um -off (Richtung Kamera)
res.append(run("rechts im Bild, zur Kamera gedreht", off, 0, (1500,0,5000), 0, 0))
res.append(run("Mitte, Kopf 40° nach links gedreht", 40, 0, (0,0,5000), -40, 0))
res.append(run("Mitte, Kopf 40° nach rechts gedreht", -40, 0, (0,0,5000), 40, 0))
res.append(run("Mitte, schaut 30° nach unten (Bildschirm)", 0, 30, (0,0,5000), 0, 30))
res.append(run("Mitte, schaut 30° nach oben", 0, -30, (0,0,5000), 0, -30))
res.append(run("mit 1px Rauschen, zur Kamera", 0, 0, (0,0,5000), 0, 0, tol=6, noise=1.0))
print("Geometrie:", "alle OK" if all(res) else "FEHLER")

