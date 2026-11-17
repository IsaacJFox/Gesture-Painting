# Gesture Painting

For CSCI-4250 Human Computer Interaction, our project explores painting with
hand gestures instead of relying on a mouse and keyboard.

This version adds a blackboard canvas, drawing, a transparent yellow highlighter,
an area eraser, and a gesture that clears the entire board. The original webcam
hand-tracking demo is still available in `hand_tracking.py`.

## Install and run on Mac

Use Python 3.9-3.12 (Python 3.11 recommended) and a webcam. Open this project
folder in VS Code, choose Terminal > New Terminal, then run:


python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py


Install packages once per virtual environment. After reopening the terminal:

bash
source .venv/bin/activate
python main.py


Allow camera access when prompted. If the camera will not open, close other
camera apps, check macOS camera permissions for VS Code or Terminal, or try:

```bash
python main.py --camera 1
```

Use the exact pinned package versions: this project uses MediaPipe's legacy
`mp.solutions` interface. If pip cannot find a compatible package, use Python
3.11 and create a new virtual environment. On Windows, create the environment
with `py -3.11 -m venv .venv`, then use `.venv\Scripts\python.exe` instead of
`python` for installation and running.

## Hand gestures

Face your palm toward the camera, with fingers pointing upward. Fold unused
fingers. The four tracked fingers below exclude the thumb.

| Action | Gesture |
| --- | --- |
| Draw | Raise only your index finger and move its tip. |
| Highlight | Raise index and middle fingers; fold ring and little fingers. The index tip controls a broad yellow stroke. |
| Erase an area | Open one palm, extending all four fingers, and move it across the drawing. A wide circular eraser follows the palm. |
| Clear the entire board | Open both palms and hold them in view for two seconds. Watch the progress indicator. Lower your hands to rearm. |
| Pause | Make a fist or move your hands out of view. |
| Select a color | Point with only the index finger at a color box for about 0.7 seconds. |

One hand controls painting. When both palms are visible, painting pauses while
the clear countdown runs. Lower either palm before the countdown completes to
cancel. After a clear, the gesture will not clear repeatedly until released.

The eraser wipes a wide band along your hand's path; it does not automatically
erase the entire horizontal row. Both ink and highlighting are removed together.
Yellow highlighting is blended over existing ink at 35% opacity. Overlapping
highlights do not keep getting darker. Highlights use a separate layer, so they
always appear over ink, including ink drawn afterward.

The main canvas starts black, with a small live camera preview at the top right
to help you position your hands. The toolbar and bottom instructions are reserved
areas. Keep your drawing cursor at least 50 pixels below the toolbar and above
the footer. Changing gestures or losing tracking breaks the current stroke so
separate marks do not connect across the board.

## Keyboard shortcuts

Click the application window before using shortcuts.

- `B`: switch between the blackboard and webcam background; preserve your artwork.
- `S`: save a PNG in `saved_drawings` next to the code. The picture includes the
  current background and artwork, without the toolbar, cursor, or camera preview.
- `1` through `4`: white, green, blue, or pink pen.
- `+` / `-`: adjust pen width between 2 and 25 pixels.
- `Q` or Escape: exit.

The highlighter stays yellow and 30 pixels wide. The area eraser is 100 pixels
wide. The displayed gesture name tells you which action is currently active.

## Files

- `main.py`: webcam loop, interface, controls, saving, and gesture actions.
- `drawing.py`: persistent ink, highlight masks, erasing, and image export.
- `gestures.py`: finger rules and the two-second clear timer.
- `hand_tracking.py`: existing MediaPipe tracker and standalone tutorial demo.
- `handTracking.py`: compatibility import used by the main application.
- `requirements.txt`: pinned dependencies.
- `tests/test_board.py`: checks for layer behavior, gesture rules, and clear timing.

Run the original tutorial demo with `python hand_tracking.py`.
Run automated checks with `python -m unittest discover -s tests`.

## Code sources and learning

The original hand-tracking work relates to Murtaza's Workshop's tutorial:
https://www.youtube.com/watch?v=NZde8Xt78Iw

MediaPipe hand tracking reference:
https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md

The implementation and this update were developed with ChatGPT assistance.


## Testing and limitations

Automated checks cover persistent drawing, highlighting, erasing, gesture mapping,
and clear countdown behavior. Webcam tracking and macOS permissions must be tested
on your computer; they cannot be verified in this environment. Test each gesture
and save an image before recording your project demonstration.

Use good lighting and keep hands fully visible. Sideways hands, bent fingers,
occlusion, or a busy background may cause incorrect classifications. Keep fingers
upright. The first detected hand controls strokes, so use one hand except for
clearing. Tracking order can change when multiple hands are present. There is no
undo or automatic session recovery; save important artwork before clearing.

This is a functional desktop prototype based on the supplied repository. It does
not yet implement the wireframe's shapes palette or full gesture-based control
of every setting. Switching backgrounds, saving, and resizing use the keyboard.
