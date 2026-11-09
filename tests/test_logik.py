import os
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import erkennung as E
me = lambda: E.Face(bbox=(.3,.2,.6,.8), center=(.45,.5), area=.18, looking=True)
other = lambda look: E.Face(bbox=(.75,.1,.85,.3), center=(.8,.2), area=.02, looking=look)
L = E.SecurityLogic(); t = 0.0; dt = 1/15; log = []
def step(sec, faces_fn, label):
    global t
    end = t + sec
    while t < end:
        for ev in L.update(t, faces_fn()):
            log.append(f"{t:5.2f}s [{label}] -> {ev.kind}{' (schaut)' if ev.looking else ''}")
        t += dt
step(2, lambda: [me()], "nur ich")
step(0.3, lambda: [me(), other(False)], "kurz Fremder (0.3s)")
step(2, lambda: [me()], "nur ich")
step(1.5, lambda: [me(), other(False)], "Fremder, schaut nicht")
step(1.5, lambda: [me(), other(True)], "Fremder schaut")
step(0.2, lambda: [me()], "Aussetzer 0.2s")
step(3, lambda: [me(), other(True)], "Fremder schaut weiter")
step(1.0, lambda: [me()], "1s weg")
step(1, lambda: [me(), other(True)], "wieder da (kein neuer Alarm)")
step(3, lambda: [me()], "verschwunden")
step(1, lambda: [me(), other(True)], "neue Person schaut")
step(3, lambda: [], "niemand")
print("\n".join(log))
