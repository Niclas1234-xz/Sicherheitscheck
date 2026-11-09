import os, sys, math, io, contextlib
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erkennung as E
with contextlib.redirect_stdout(io.StringIO()):
    import test_augen as TA
W, H = 1280, 720
f = E.focal_px(W); CAM = np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]]); U = E.MODEL_UNITS_PER_MM
ok_all = True
def check(name, cond, info=""):
    global ok_all; ok_all &= bool(cond); print(("OK  " if cond else "FAIL"), name, info)

def far_face(dist_m, yaw):
    t = np.array([0.3, -0.25, dist_m]) * 1000 * U
    R = TA.rot(yaw + math.degrees(math.atan2(t[0], t[2])))
    pts,_ = cv2.projectPoints(E.FIVE_POINT_MODEL, cv2.Rodrigues(R)[0], t, CAM, np.zeros(4)); pts = pts.reshape(5,2)
    x0,y0 = pts.min(0)-8; x1,y1 = pts.max(0)+8
    return E.face_from_points5((x0,y0,x1-x0,y1-y0), pts, W, H)

# 1) Nah: unverändert (keine Strenge, keine Wartezeit)
fc = E.analyze_landmarks(TA.synth(40, 0, (800,-600,4500), 'kamera'), W, H)
check("Nah (~0.9 m): wie bisher", fc.strict == 0 and fc.dwell_s == 0 and fc.looking,
      f"Entf {fc.distance_m:.2f} m, streng {fc.strict:.0%}, Wartezeit {fc.dwell_s:.1f} s, schaut={fc.looking}")
# 2) Weit weg: strengerer Winkel
for yaw, expect in ((5, True), (12, True), (22, False), (28, False)):
    fc = far_face(3.0, yaw)
    check(f"3 m, Kopf {yaw}°", fc.looking == expect, f"streng {fc.strict:.0%}, schaut={fc.looking} (vorher bis 32° = schaut)")
# 3) Mittlere Distanz: gleitender Übergang
fc = far_face(1.75, 20)
check("1.75 m: Übergang", 0.4 < fc.strict < 0.6 and abs(fc.dwell_s - 0.75) < 0.1,
      f"streng {fc.strict:.0%}, Wartezeit {fc.dwell_s:.2f} s, schaut bei 20°={fc.looking}")

# 4) Zeitablauf mit Warnlogik
def run(timeline):
    L, T, t, dt, events = E.SecurityLogic(), E.FaceTracker(), 0.0, 1/15, []
    for dur, mk in timeline:
        end = t + dur
        while t < end:
            me = E.Face(bbox=(.3,.3,.6,.9), center=(.45,.6), area=.18, looking=True)
            faces = [me] + ([mk()] if mk else [])
            T.update(faces, t)
            for ev in L.update(t, faces): events.append((round(t, 2), ev.kind))
            t += dt
    return events
far_look = lambda: far_face(3.0, 4)
far_away = lambda: far_face(3.0, 60)
ev = run([(1, None), (1.0, far_away), (0.8, far_look), (1.0, far_away)])
check("3 m: kurzer Blick (0.8 s) -> nur orange", [k for _, k in ev] == ["alert"], str(ev))
ev = run([(1, None), (1.0, far_away), (3.0, far_look)])
red = [t for t, k in ev if k == "looking"]
check("3 m: Starren -> rot nach ~1.5 s", red and 3.5 <= red[0] <= 3.9, f"{ev}  (Blick ab 2.0 s)")
def near_look():
    fc = E.analyze_landmarks(TA.synth(10, 0, (1500,-600,4200), 'kamera'), W, H); return fc
ev = run([(1, None), (0.8, near_look)])
check("Nah: kurzer Blick -> sofort rot (unverändert)", ["alert"] == [k for _, k in ev][:1] and ev and ev[0][1] == "alert" and ev[0][0] < 1.7,
      str(ev))
print("ALLE OK" if ok_all else "FEHLER")
