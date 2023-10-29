
# Gaze-Tracking Game

## Overview

This Python-based application uses gaze tracking to control a simple game built with Pygame. The game captures eye movements through a webcam and uses them to control a player on the screen. The game also includes randomly generated blue boxes that the player must avoid.


## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- Pygame
- dlib

You can install the required packages using pip:

```bash
pip install opencv-python pygame dlib
```

## How to Play

1. **Position Yourself**: Sit at a comfortable distance from the webcam to ensure accurate gaze tracking.
   
2. **Start the Game**: Run the game by executing `python main.py`. The game window will open.

3. **Control the Player**: Your eye movement will control the position of the player on the screen. Look left to move the player left, look right to move the player right, and look at the center to keep the player in the middle.

4. **Avoid Blue Boxes**: Blue boxes will appear randomly at the top and move downwards. Your objective is to avoid colliding with these boxes.

5. **End Game**: The game ends if the player collides with a blue box. Close the game window or press the quit button to exit.

## Features

- Real-time gaze tracking for game control
- Randomly generated obstacles
- Collision detection

## How to Run

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-repo/gaze-tracking-game.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd gaze-tracking-game
    ```

3. **Run the game:**

    ```bash
    python main.py
    ```

    If the webcam is not accessible, an error message will be displayed, and the program will exit.

## File Structure

- `main.py`: The main game loop and initialization code.
- `shape_predictor_68_face_landmarks.dat`: The pre-trained dlib model for facial landmark detection. Download the file [here](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)

## Functions

- `cv2_to_pygame(cv_image)`: Converts an OpenCV image to a Pygame surface.
- `generate_blue_box()`: Generates a blue box at a random position.
- `midpoint(pts1, pts2)`: Calculates the midpoint between two points.
- `faceDetector(image, gray, Draw=True)`: Detects faces in the frame.
- `faceLandmakDetector(image, gray, face, Draw=True)`: Detects facial landmarks.
- `EyeTracking(image, gray, eyePoints)`: Tracks the eye and determines its position.
- `Position(ValuesList)`: Determines the eye's position based on black pixel count in different regions of the eye.

## Troubleshooting

- **Webcam not accessible**: Ensure the webcam is not being used by another application and that you have the necessary permissions.

