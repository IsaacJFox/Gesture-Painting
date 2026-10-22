# Gesture-Painting
For our class CSCI-4250 Human Computer Interaction we were tasked to create something that would be interacted with beyond the basic mouse and keyboard controls. We chose to do gesture controls to interface with painting.

# Gesture Painting — Hand Tracking Demo

This demo uses a webcam to detect hands and track finger movements.
A yellow dot follows the index fingertip, which could later control
a paintbrush.

## Code source

This code was developed with assistance from ChatGPT and Youtube Video.

Related tutorial:
https://www.youtube.com/watch?v=NZde8Xt78Iw


## How to run on Mac

Open a terminal in the project folder and run:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python hand_tracking.py

Allow camera access if prompted. Press Q to quit.

### What I learned
I learned how to use OpenCV to capture a live webcam feed and MediaPipe to detect hands and track finger positions.This could later be used to control a paintbrush in our Gesture Painting project.