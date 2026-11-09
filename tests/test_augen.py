import os
import sys, math
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import erkennung as E

W, H = 1280, 720
CAM = np.array([[E.focal_px(W),0,W/2],[0,E.focal_px(W),H/2],[0,0,1.]])
U = 450/92.0               # Modelleinheiten pro mm
R_EYE = 11.5*U

def rot(yaw=0, pitch=0):
    base = cv2.Rodrigues(np.array([math.pi,0,0.]))[0]
    Ry = cv2.Rodrigues(np.array([0, math.radians(yaw), 0.]))[0]
    Rx = cv2.Rodrigues(np.array([math.radians(pitch), 0, 0.]))[0]
    return Ry @ Rx @ base

# Augen im Kopfmodell (x im Bild links/rechts, inneres Auge etwas weiter vorne)
EYES_MODEL = {  # idx: Punkt
    33: (-225,170,-135), 133: (-225+30*U,170,-135+5*U),
    263: (225,170,-135), 362: (225-30*U,170,-135+5*U),
}
def project(P):
    p = CAM @ P; return p[:2]/p[2]

def synth(yaw, pitch, t, target, noise=0.0, rng=None):
    """target: 'kamera', 'kopf' (geradeaus), oder 3D-Punkt in Kamerakoordinaten"""
    R = rot(yaw, pitch); t = np.array(t, float)
    lm = np.zeros((478,2))
    pts,_ = cv2.projectPoints(E.MODEL_POINTS, cv2.Rodrigues(R)[0], t, CAM, np.zeros(4))
    for i,p in zip([1,152,33,263,61,291], pts.reshape(-1,2)): lm[i]=p
    for i,P in EYES_MODEL.items(): lm[i] = project(R@np.array(P,float)+t)
    for (a,b), iris_idx in (((33,133),468), ((362,263),473)):
        mid = (np.array(EYES_MODEL[a],float)+np.array(EYES_MODEL[b],float))/2
        C = R@(mid - np.array([0,0,R_EYE])) + t       # Augapfelzentrum
        if isinstance(target,str) and target=='kopf': g = R@np.array([0,0,1.])
        else:
            T = np.zeros(3) if (isinstance(target,str) and target=='kamera') else np.array(target,float)
            g = (T-C)/np.linalg.norm(T-C)
        lm[iris_idx] = project(C + R_EYE*g)
    if noise: lm += rng.normal(0,noise,lm.shape)
    return lm

def deg_away_from_cam(t, angle):  # Punkt seitlich neben der Kamera (Blick ins Leere)
    d = np.linalg.norm(t); return np.array([math.tan(math.radians(angle))*d*1.0, 0, 0])

SCREEN = np.array([0, 250*U/4.89*4.89/U*U, 0])  # ~250 mm unter der Kamera
SCREEN = np.array([0, 250*U, 0])
cases = [
  ("Kopf zur Kamera, Augen zur Kamera",             0,  0, 'kamera', True),
  ("Kopf 40° weggedreht, Augen zur Kamera",         40, 0, 'kamera', True),
  ("Kopf -40° weggedreht, Augen zur Kamera",       -40, 0, 'kamera', True),
  ("Kopf 40° weggedreht, Augen auf den Bildschirm", 40, 0, SCREEN,   True),
  ("Kopf 40° weggedreht, Augen geradeaus",          40, 0, 'kopf',   False),
  ("Kopf 60° weggedreht, Augen geradeaus",          60, 0, 'kopf',   False),
  ("Kopf zur Kamera, Augen 40° zur Seite",           0, 0, None,     False),
  ("Kopf 30° gedreht + 20° gesenkt, Augen Bildschirm", 30, 20, SCREEN, True),
]
t = (800,-600,4500)
allok = True
for name, yaw, pitch, target, expect in cases:
    for noise in (0.0, 0.6):
        rng = np.random.default_rng(3)
        if target is None:
            # Blick 40° zur Seite: Zielpunkt seitlich vom Gesicht
            C = np.array(t,float); tgt = C + np.array([math.sin(math.radians(40)), 0, -math.cos(math.radians(40))])*5000
            tg = tgt
        else: tg = target
        lm = synth(yaw, pitch, t, tg, noise, rng)
        f = E.analyze_landmarks(lm, W, H)
        ok = f.looking == expect
        allok &= ok
        print(f"{'OK ' if ok else 'FAIL'} {name:50s} Rauschen={noise:.1f}px  Kopf={f.dev_h:6.1f}°  Blick=({f.gaze_h:6.1f}°, {f.gaze_v:5.1f}°)  Augen={f.eyes_used}  schaut={f.looking}")
print("ALLE OK" if allok else "FEHLER")
