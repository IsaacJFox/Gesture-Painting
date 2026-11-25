# Gesture Painting

For CSCI-4250 Human Computer Interaction, this project explores drawing with
webcam hand gestures. It includes a blackboard, a live-camera background option,
colors, a highlighter, area erasing, whole-board clearing, smoothing, automatic
shape correction, and optional voice commands.

## Run on Mac

Open the folder containing `main.py` in VS Code, then Terminal > New Terminal.
Use Python 3.9-3.12; Python 3.11 is recommended for the pinned MediaPipe version.
Run each command separately:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The current project's default camera is **1**. If the webcam does not open, try:

```bash
python main.py --camera 0
```

Allow camera access for VS Code or Terminal in macOS System Settings > Privacy
& Security > Camera. Close other programs that are using the camera.

After the initial installation, only run:

```bash
source .venv/bin/activate
python main.py
```

The triple backticks and `bash` labels in this README format code blocks.
Copy only the commands inside them into Terminal.

## Install voice support on Mac

Voice packages are separate so the app can still run in gesture-only mode
without microphone dependencies.

With Homebrew installed, run:

```bash
brew install portaudio flac
source .venv/bin/activate
python -m pip install -r requirements-voice.txt
python main.py --voice
```

If Homebrew is missing (`brew: command not found`), install it using the official
instructions at https://brew.sh, reopen Terminal, and retry the command above.
If the camera needs index 0, use `python main.py --camera 0 --voice`.

Allow microphone access in System Settings > Privacy & Security > Microphone.
At startup, remain quiet during the one-second calibration, then wait for
**Voice: Listening**. Speak one command, pause, and check the **Heard** line and
on-screen result before giving the next command. Recognition runs on a background
thread so the webcam loop can continue.

**Voice recognition needs internet. When enabled, captured speech is sent to
Google Speech Recognition for transcription. This app does not save microphone
recordings.** No API key is configured in the project; it uses SpeechRecognition's
default Google recognizer for a classroom prototype. Service availability and
recognition accuracy are not guaranteed. Voice is off until you pass `--voice`
or press `V` in the application window. `V` also turns it off.

## Voice commands

| Say | Result |
| --- | --- |
| "Change color to blue" | Select blue pen ink. Also accepts white, green, or pink. |
| "Highlight mode" | Keep the highlighter selected while your finger moves. |
| "Drawing mode" | Return to the regular pen. |
| "Eraser mode" | Point to erase with the selected eraser. |
| "Gesture mode" | Restore the original gesture-based tool selection. |
| "Clear screen" | Pause drawing and display a confirmation prompt. |
| "Confirm clear" | Clear everything if a request is still pending. |
| "Cancel" | Cancel the clear request. |

"Color blue", "change colour to blue", "pen mode", "highlighter mode", and
"clear board" are also accepted. An optional "Canvas" prefix is accepted, such
as "Canvas, highlight mode". Give one command at a time; unsupported colors and
unrelated conversation do not match the command parser.

A clear request expires after **10 seconds** unless you say "confirm clear".
Wait for the prompt before speaking the confirmation. Saying another supported
command cancels the request. Escape also cancels while the prompt is visible.
There is no undo, so save important drawings first.

### How voice and gestures work together

Voice changes the selected tool, not the cursor position. After "highlight mode",
point with your index finger or raise index and middle fingers to highlight.
Changing between those two finger poses does **not** switch back to the pen.
Make a fist or move your hand out of view to pause. "Drawing mode" selects the
pen again. An open palm still temporarily wipes, and two palms can still clear.
The toolbar shows both the selected tool and current action.

Color commands change pen ink only. The highlighter stays yellow, so changing
pen color while highlighting does not turn off highlighting. Press `G` or say
"gesture mode" to restore the original tool gestures. Turning the microphone
off does not reset the selected tool; `G` is available even without voice input.

## Gesture controls

Keep your palm facing the camera and fingers mostly upright. Fold unused
fingers. The four fingers used for classification exclude the thumb.

| Action in gesture mode | Gesture |
| --- | --- |
| Draw | Raise the index finger, with middle finger folded. |
| Highlight | Raise index and middle; fold ring and little fingers. |
| Area erase | Open one palm and move it across the marks. |
| Clear entire board | Hold both open palms in view for two seconds. |
| Pause | Make a fist or move your hands out of view. |
| Select pen color | Point at a color box for approximately 0.7 seconds. |

The area eraser follows your palm and removes ink and highlights in a wide band
along its path. It does not delete the entire horizontal row. In voice-selected
eraser mode, pointing instead uses the fingertip to position the eraser.

Two-palm clear displays progress and fires once per hold. Release the gesture
before clearing again. Gesture filtering tolerates a few uncertain frames, so
releasing may take a few frames to register. Voice clearing uses its separate
spoken confirmation instead of the two-palm hold.

## Canvas and tools

- The canvas starts black. A small live-camera preview helps position your hands.
- `B` switches between the blackboard and camera background without removing art.
- Highlights are yellow, 30 pixels wide, and about 35% opaque. Overlap does not
  continually darken them. Their layer is above pen ink, even ink drawn later.
- The eraser has a 50-pixel radius. Pen width can be adjusted from 2 to 25 pixels.
- Keep the drawing cursor below the toolbar by at least 50 pixels and above the
  footer by at least 50 pixels. Toolbars, voice feedback, and preview are UI overlays.
- Automatic shapes attempt to replace roughly closed circles, rectangles, and
  triangles with cleaner outlines when a stroke ends. `T` toggles this. Turn it
  off for freehand writing. Recognition is heuristic and can mistake drawings.
- `[` makes smoothing more responsive; `]` makes it steadier with more lag.
- `S` saves a PNG in `saved_drawings` next to the code. It includes artwork and
  the selected background, without toolbars, preview, cursor, or voice feedback.

## Keyboard shortcuts

Click the app window first.

| Key | Action |
| --- | --- |
| `V` | Enable/disable microphone commands. |
| `G` | Return to gesture tool selection. |
| `B` | Toggle background. |
| `S` | Save PNG. |
| `1` / `2` / `3` / `4` | White / green / blue / pink pen. |
| `+` / `-` | Adjust pen width. |
| `[` / `]` | Adjust smoothing. |
| `T` | Toggle automatic shapes. |
| `D` | Toggle raw/smoothed trails and performance information. |
| `Q` / Escape | Quit; Escape cancels a pending clear first. |

## Microphone troubleshooting

If voice setup fails, the window shows a status and prints details in Terminal;
the painting app remains usable. Install the voice requirements in the same
activated environment. If PyAudio compilation fails, check that PortAudio is
installed and that Homebrew and Python use the same processor architecture.
On Apple Silicon, FLAC is included in the Homebrew setup above for audio encoding.

List microphones with:

```bash
python voice_commands.py --list-microphones
```

Choose a listed input device, for example index 2:

```bash
python main.py --voice --microphone 2
```


Speak clearly, reduce background noise, and wait for **Listening**. Speech
recognition can mishear commands or fail when the internet/service is unavailable.
Press `V` to turn voice off, and `G` to return to gesture tool selection.
If restarting the microphone reports it is stopping, wait a few seconds and retry.

## Files and tests

- `main.py`: webcam loop, interface, keyboard/voice actions, gesture integration.
- `voice_commands.py`: phrase parser, selected-tool rules, clear confirmation,
  and optional microphone worker.
- `gestures.py`: landmark-based gestures, filtering, hand choice, clear timer.q
- `drawing.py`: persistent layers, smoothing, shapes, erasing, image export.
- `hand_tracking.py`: existing MediaPipe tracker and original standalone demo.
- `handTracking.py`: compatibility import.
- `requirements.txt`: original pinned drawing/tracking packages.
- `requirements-voice.txt`: base packages plus SpeechRecognition and PyAudio.
- `tests/test_voice.py`: parser, mode persistence, confirmation timing, stale
  response rejection, optional dependency handling, and app integration tests.

Run checks from the project folder:

```bash
python -m unittest discover -s tests -v
```

The voice/control tests do not need a physical microphone, internet, or MediaPipe.
They use the installed drawing dependencies. The tests simulate recognized text;
they do not measure speech recognition accuracy. The actual webcam and microphone
must be checked on your Mac. The standalone hand tracker still runs with
`python hand_tracking.py`.

## Changes in this update

Added optional voice commands for pen colors, drawing/highlighter/eraser selection,
and clearing with confirmation. Voice-selected tools persist across finger-pose
changes. Added microphone status, recognized-text feedback, voice fallback controls,
voice tests, and current setup documentation. Corrected `.gitignore` and removed
archive metadata. Retained the supplied smoothing, automatic shapes, and tracker.

## Sources and limitations
https://www.youtube.com/watch?v=NZde8Xt78Iw

MediaPipe reference:
https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md

SpeechRecognition reference and microphone installation instructions:
https://github.com/Uberi/speech_recognition
https://pypi.org/project/SpeechRecognition/3.14.3/

Good lighting and fully visible hands help tracking. Sideways or occluded hands
can cause incorrect gestures. Smoothing trades responsiveness for steadiness.
Voice requires an enabled microphone and online service. Shapes and speech can
be wrong; test with your users before the final demonstration. This version does
not add the full shapes palette from the wireframe, undo, or session recovery.
