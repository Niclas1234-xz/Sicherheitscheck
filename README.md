# Sicherheitscheck

Schützt deinen Bildschirm vor fremden Blicken: Das Programm wertet laufend deine Webcam aus. Taucht neben dir eine weitere Person auf, erscheint oben ein dezenter oranger Streifen; schaut sie auf deinen Bildschirm, klappt oben in der Mitte eine animierte Warn-Insel auf. Ob jemand hinschaut, schätzt das Programm anhand von Kopfhaltung und Blickrichtung der Augen.

Das Programm arbeitet vollständig ohne Ton: keine Warntöne und keine Systembenachrichtigungen. Alle Bilder werden ausschliesslich lokal ausgewertet und nirgends gespeichert oder hochgeladen.

## Installation und Start

Voraussetzung: **Python 3.10 oder neuer** (von python.org oder über den Paketmanager); am besten erprobt sind 3.11 bis 3.13. Unter Python 3.15 schaltet das Programm die neuen «Lazy Imports» automatisch ab, weil sie zusammen mit PySide6 MediaPipe zum Absturz bringen (Fehler «'lazy_import' object is not callable»).

**Windows:** Doppelklick auf `start_windows.bat`.
**macOS / Linux:** Im Terminal `./start_mac_linux.sh` ausführen (eventuell vorher `chmod +x start_mac_linux.sh`).

Beim ersten Start werden die benötigten Pakete in einen eigenen Ordner `.venv` installiert und die Erkennungsmodelle (rund 4 MB plus 230 KB für die Fern-Erkennung) einmalig heruntergeladen. Danach startet das Programm in wenigen Sekunden.

### Unten rechts im Infobereich

Der Sicherheitscheck ist eine Infobereich-App: Er erscheint **nicht** bei den Apps in der Taskleiste und hat kein Desktop-Symbol, sondern ein Farbsymbol unten rechts (macOS: oben in der Menüleiste).

| Symbol | Bedeutung |
|---|---|
| 🟢 grün | Sicherheitscheck ist aktiv |
| 🟠 orange | Fenster ist geöffnet, der Check aber nicht aktiv |
| 🔴 rot | Programm läuft, Check nicht aktiv, Fenster geschlossen |

- **Klick auf das Symbol** öffnet das Programm unten rechts. **✕** blendet das Fenster nur aus, ein laufender Check geht weiter.
- **Rechtsklick auf das Symbol:** Fenster öffnen, Check starten/beenden, «Beim Anmelden automatisch starten» ein/aus, «Programm beenden».
- Beim ersten Start richtet das Startskript alles ein: Eintrag im Startmenü (macOS: Programme-Ordner, Linux: Anwendungsmenü) und **Autostart beim Anmelden**, damit das Symbol immer bereitsteht. Ein altes Desktop-Symbol wird entfernt.
- **Windows 11** versteckt neue Symbole anfangs hinter dem Pfeil **^** im Infobereich. Damit das Farbsymbol immer sichtbar ist: Symbol aus dem Pfeil-Menü auf die Taskleiste ziehen, oder *Einstellungen → Personalisierung → Taskleiste → Andere Symbole der Taskleiste* → «Sicherheitscheck» einschalten.

Hast du den Programmordner verschoben, richtest du alles neu ein mit:

```bash
.venv/bin/python verknuepfung.py          # macOS / Linux
.venv\Scripts\python.exe verknuepfung.py   # Windows
```

Manuell geht es so:

```bash
python -m pip install -r requirements.txt
python sicherheitscheck.py
```

Optionen:

| Option | Wirkung |
|---|---|
| `--kamera 1` | andere Webcam verwenden (Standard: 0) |
| `--tray` | startet still im Infobereich, ohne Fenster (so startet der Autostart) |
| `--warnung-testen` | zeigt als Vorschau erst den orangen Streifen, nach 3 Sekunden die Warn-Insel |

**macOS:** Beim ersten Start fragt macOS nach dem Kamerazugriff für das Terminal bzw. Python. Erlaube ihn unter *Systemeinstellungen → Datenschutz & Sicherheit → Kamera*.

## Bedienung

1. **«Sicherheitscheck starten»** klicken. Die Live-Ansicht ist zuerst aus, um Energie zu sparen. Mit dem grossen Knopf **«Live-Ansicht»** blendest du das HUD mit Augen-Tracking und Messwerten ein (siehe unten), mit «Live-Ansicht aus» wieder aus. Der Schutz läuft in beiden Fällen.
2. Arbeite normal weiter: Die Erkennung läuft im Hintergrund, auch wenn du das Fenster mit **✕** ausblendest. Das Farbsymbol unten rechts zeigt jederzeit den Zustand. **Ganz beenden:** Rechtsklick auf das Symbol → «Programm beenden». Das Programm läuft immer nur einmal: Öffnest du es ein zweites Mal, kommt einfach das bestehende Fenster nach vorne.
3. Taucht eine zweite Person auf, erscheint oben am Bildschirm ein **oranger Streifen**. Er stört nicht: Du kannst durch ihn hindurchklicken, und er nimmt nie den Fokus. Er bleibt, solange die Person im Bild ist, und wird rot, solange sie auf deinen Bildschirm schaut.
4. Erst wenn die Person **auf deinen Bildschirm schaut**, klappt oben in der Bildschirmmitte eine animierte Warn-Insel auf (ähnlich der Dynamic Island): Eine kleine dunkle Pille gleitet herein, federt auf ein bis zwei Drittel der Bildschirmbreite auf und zeigt ein pulsierendes rotes Auge mit «Achtung! Eine zweite Person schaut auf deinen Bildschirm.». Die Insel nimmt keinen Fokus, du kannst also weiterarbeiten. Sie schliesst sich automatisch, sobald niemand mehr da ist, spätestens aber nach **5 Sekunden** (ein ablaufender Balken zeigt die Zeit). Mit **«Schliessen»** blendest du sie sofort aus. Pro Besuch erscheint sie nur einmal: erneut erst, wenn die Person das Bild verlassen hat (mindestens 2 Sekunden) und danach wieder jemand hinschaut. Die Erkennung läuft immer weiter.
5. **«Sicherheitscheck beenden»** stoppt die Erkennung und gibt die Webcam frei. Über das Symbol im Infobereich lässt sich das Programm ganz beenden.

## Live-Vorschau (HUD)

Nach einem Klick auf **«Live-Ansicht»** zeigt die Vorschau in Echtzeit, was das Programm sieht. Ist die Live-Ansicht aus oder das Fenster minimiert bzw. geschlossen, wird nichts gezeichnet; das spart Rechenleistung und Akku, die Erkennung läuft aber unverändert weiter.

- **Eck-Markierungen** um jedes Gesicht: grün = du, orange = weitere Person, rot = schaut auf deinen Bildschirm
- **Gesichtsnetz** aus den erkannten Gesichtspunkten und der Gesichtsumriss
- **Augen-Tracker**: Augenkontur, Fadenkreuz auf jeder Iris und ein **Blickstrahl**, der zeigt, wohin die Augen schauen. Schaut jemand direkt in die Kamera, erscheint ein pulsierender Zielpunkt
- **Zielerfassung**: Schaut eine fremde Person auf deinen Bildschirm, kreist ein roter Ring um ihr Gesicht und die Marke zeigt «BLICK ERFASST»
- **Datenfeld** neben jedem Gesicht:

| Wert | Bedeutung |
|---|---|
| ENTF | geschätzte Entfernung zur Kamera |
| KOPF / NEIG | Kopfdrehung links/rechts, oben/unten und Neigung |
| BLICK | ob die Person auf den Bildschirm schaut |
| QUELLE | ob der Blick aus Iris + Kopf oder nur aus der Kopfhaltung bestimmt wurde |
| AUGEN | wie weit die Augen geöffnet sind |
| LIDSCHL | Lidschläge pro Minute (nach einigen Sekunden Messzeit) |
| MUND | ob die Person spricht |
| MIMIK | ob die Person lächelt |
| FOKUS | Anteil der letzten 10 Sekunden mit Blick auf den Bildschirm |
| DAUER | wie lange eine fremde Person schon ununterbrochen hinschaut |

Alle Werte sind Schätzungen aus dem Kamerabild (die Entfernung z. B. ± 20–30 %, je nach Webcam). Sie werden nur angezeigt und nirgends gespeichert. Das Programm erkennt keine Identität und schätzt weder Alter, Geschlecht noch Gefühle.

## So funktioniert die Erkennung

- **Gesichter:** MediaPipe Face Landmarker findet bis zu fünf Gesichter mit je 478 Gesichtspunkten (inklusive Iris).
- **Fern-Erkennung:** MediaPipe ist auf nahe Gesichter ausgelegt. Zusätzlich sucht OpenCV YuNet etwa fünfmal pro Sekunde im ganzen Bild nach kleinen, weit entfernten Gesichtern. Jedes gefundene Gesicht wird herangezoomt und dort mit MediaPipe genau vermessen. Gelingt das nicht, schätzt das Programm die Kopfhaltung aus fünf Grobpunkten (Augen, Nase, Mundwinkel). Ist YuNet nicht verfügbar, sucht das Programm ersatzweise in vergrösserten Bildvierteln.
- **Nah oder fern:** Je kleiner die Augen im Bild, desto mehr entscheidet die Kopfhaltung statt der Iris. Weil die Augen den Kopf um einige Grad «korrigieren» können, ist die Toleranz dann etwas grösser (32° statt 25°). Im HUD zeigt «QUELLE», worauf die Schätzung beruht.
- **Strenger Modus für Personen weiter weg:** Bis 1,3 m bleibt alles wie oben. Danach wird die Prüfung gleitend strenger und gilt ab 2,2 m voll: Der Blick muss viel direkter auf den Laptop zeigen (15° seitlich, 8° nach oben, 30° nach unten) und **etwa 1,5 Sekunden ununterbrochen** auf dem Bildschirm bleiben. Erst dann wird es rot. Ein kurzer Blick aus dem Hintergrund ergibt nur den orangen Streifen. Im HUD zeigt «BLICK  PRÜFE 0.8/1.5 S», wie lange die Person schon hinschaut, und «MODUS», wie streng geprüft wird. Die Werte lassen sich oben in `erkennung.py` anpassen (`STRICT_FROM_M`, `STRICT_FULL_M`, `FAR_YAW_TOL_DEG`, `FAR_DWELL_S`).
- **Wer bist du?** Das grösste bzw. bisher verfolgte Gesicht gilt als deines, alle anderen als zusätzliche Personen.
- **Entprellung:** Eine zweite Person muss rund 0,5 Sekunden sichtbar sein, bevor gewarnt wird. Kurze Fehlerkennungen lösen also keinen Alarm aus.
- **Kopfhaltung:** Aus sechs Gesichtspunkten (Nase, Kinn, Augen- und Mundwinkel) wird die Kopfrichtung im Raum berechnet und mit der Richtung zur Kamera verglichen. Wer auf den Bildschirm schaut, schaut ungefähr zur Kamera oder etwas darunter.
- **Blickrichtung (Augen):** Aus der Kopfhaltung wird die Lage der Augäpfel im Raum berechnet. Die Position der Iris auf dem Augapfel ergibt dann die tatsächliche Blickrichtung. Entscheidend ist also, wohin die **Augen** schauen: Wer den Kopf bis etwa 45° wegdreht, aber mit den Augen auf deinen Bildschirm schielt, wird als «schaut» erkannt. Das besser sichtbare, der Kamera zugewandte Auge zählt dabei stärker. Nur wenn die Augen zu klein oder verdeckt sind, wird auf die Kopfhaltung zurückgegriffen.
- **Auflösung:** Die Analyse nutzt das volle 720p-Kamerabild, damit die Augen auch bei Personen weiter hinten scharf genug sind.

Die Schwellenwerte (Winkeltoleranzen, Wartezeiten, Bildrate) stehen oben in `erkennung.py` und lassen sich dort anpassen.

## Grenzen

- Erkannt werden nur Personen, deren **Gesicht** sichtbar ist. Wer mit dem Rücken zur Kamera steht, wird nicht gezählt. Wer auf deinen Bildschirm schaut, ist aber praktisch immer von vorne zu sehen.
- Dank Fern-Erkennung werden Gesichter bis etwa 5–6 Meter erkannt (je nach Webcam und Licht). Weit weg beruht die Blickschätzung auf der Kopfhaltung und ist entsprechend gröber.
- Gesichter auf Fotos oder Plakaten im Hintergrund können als Person erkannt werden.
- Das Programm erkennt nicht, **wer** du bist. Verlässt du den Platz und setzt sich jemand anderes hin, gilt diese Person als einzige im Bild.
- Die Blickrichtung ist eine Schätzung: Bei schlechtem Licht, Brillenreflexen oder kleinen Gesichtern bleibt es im Zweifel beim orangen Streifen.
- **macOS:** Über Apps im Vollbildmodus (eigener Schreibtisch/Space) können Streifen und Warn-Insel je nach Systemversion verdeckt bleiben. Das eingefärbte Symbol in der Menüleiste zeigt den Alarm trotzdem an.

## Dateien

| Datei | Inhalt |
|---|---|
| `sicherheitscheck.py` | Oberfläche: Hauptfenster, Streifen, Warn-Insel, Symbol im Infobereich |
| `erkennung.py` | Gesichtserkennung, Kopfhaltung, Blickrichtung und Warnlogik |
| `verknuepfung.py` | richtet Startmenü- bzw. Programme-Eintrag und den Autostart ein |
| `icon/` | App-Symbol (Windows, macOS, Linux) |
| `tests/` | Tests mit simulierten Gesichtern, z. B. `python tests/test_streng.py` |
| `requirements.txt` | benötigte Python-Pakete |
| `start_windows.bat`, `start_mac_linux.sh` | Startskripte mit automatischer Einrichtung |

Falls der automatische Modell-Download blockiert ist (z. B. durch eine Firmen-Firewall), kannst du die Dateien auch manuell in den Programmordner legen. Für die Fern-Erkennung: `face_detection_yunet_2023mar.onnx` von <https://huggingface.co/opencv/face_detection_yunet>. Das Hauptmodell `face_landmarker.task` gibt es unter
<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
herunter und lege die Datei in den Programmordner.
