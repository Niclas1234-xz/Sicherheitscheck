import os
import sys, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erkennung as E
import test_augen as TA  # erzeugt synthetische Gesichter (druckt seine eigenen Tests)
lm = TA.synth(30, 0, (800,-600,4500), 'kamera')
f = E.analyze_landmarks(lm, TA.W, TA.H)
print(f"\nEntfernung: {f.distance_m:.2f} m (erwartet {np.linalg.norm([800,-600,4500])/E.MODEL_UNITS_PER_MM/1000:.2f} m), Neigung {f.roll_deg:.1f}°, schaut={f.looking}")

T = E.FaceTracker(); t = 0.0; dt = 1/15
def mk(cx, blink, mouth, look, smile=0.0):
    fc = E.Face(bbox=(cx-.05,.2,cx+.05,.4), center=(cx,.3), area=.01, looking=look, mouth=mouth)
    fc.blend = {"eyeBlinkLeft":blink,"eyeBlinkRight":blink,"mouthSmileLeft":smile,"mouthSmileRight":smile}
    return fc
last = {}
for i in range(int(20/dt)):
    blink = 0.9 if (i % 60) in (0,1,2) else 0.05            # alle 4 s ein Lidschlag
    mouth_b = 0.25*abs(math.sin(i*0.9)) if t > 10 else 0.02    # Person B spricht ab 10 s
    a = mk(0.3, 0.05, 0.02, False)
    b = mk(0.7 + 0.002*math.sin(i/10), blink, mouth_b, look=(t > 5), smile=0.6 if t > 15 else 0.1)
    T.update([a, b], t)
    last = {"A": a, "B": b}
    t += dt
a, b = last["A"], last["B"]
print(f"Person A: ID {a.track_id}, Lidschläge {a.blinks}, spricht={a.talking}, Fokus={a.attention:.0%}")
print(f"Person B: ID {b.track_id}, Lidschläge {b.blinks} (erwartet 5), Rate {b.blink_rate:.0f}/min (erwartet 15), "
      f"Augen offen {b.eyes_open:.0%}, spricht={b.talking}, lächelt={b.smiling}, schaut seit {b.looking_since:.1f} s (erwartet ~15), Fokus {b.attention:.0%}")
