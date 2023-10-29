import time
import pygame
import cv2 as cv
import numpy as np
import utils as utils  # Your eye tracking module

# Initialize Pygame
pygame.init()
win_width, win_height = 640, 480
win = pygame.display.set_mode((win_width, win_height))

try:
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        raise IOError("Cannot open webcam")
except IOError as e:
    print(e)
    exit(1)

# Initialize game variables
player_width, player_height = 50, 50
player_y = win_height - 60
possible_positions = [
    win_width // 4,
    win_width // 2,
    3 * win_width // 4,
]  # Create only 3 positions
player_x = possible_positions[1]  # Start at center
blue_boxes = []


# Game loop
run = True
position = 1
last_move_time = 0  # Time of the last move
gaze_start_time = 0  # Time when the gaze direction was first detected
current_gaze = None  # Current gaze direction
gaze_hold_time = 0.15  # Time required to hold gaze
move_delay = 0.5  # Time gap between movements in seconds
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Get frame from camera
    ret, frame = camera.read()
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    image, face = utils.faceDetector(frame, grayFrame)

    if face is not None:
        image, PointList = utils.faceLandmakDetector(frame, grayFrame, face, False)
        RightEyePoint = PointList[36:42]
        mask, pos, color = utils.EyeTracking(frame, grayFrame, RightEyePoint)

        # Convert OpenCV eye image to Pygame surface and display
        eye_pygame = utils.cv2_to_pygame(mask)
        win.blit(eye_pygame, (0, 0))

        current_time = time.time()
        if current_gaze != pos:
            gaze_start_time = current_time
            current_gaze = pos

        if current_time - last_move_time > move_delay:
            if current_time - gaze_start_time > gaze_hold_time:
                # Move player based on eye position
                if pos == "Left":
                    position = position - 1 if position > 0 else 0
                elif pos == "Right":
                    position = position + 1 if position < 2 else 2

                player_x = possible_positions[position]
                last_move_time = current_time

    # Generate blue boxes occasionally
    if np.random.rand() < 0.1:  # Adjust frequency as needed
        blue_boxes.append(utils.generate_blue_box())

    # Update and draw blue boxes
    for box in blue_boxes:
        box[1] += 8  # Move down; adjust speed as needed
        pygame.draw.rect(win, (0, 0, 255), box)

        # Check for collision with player
        if pygame.Rect(box).colliderect(
            player_x, player_y, player_width, player_height
        ):
            run = False  # End game or take other action

    # Draw player
    pygame.draw.rect(
        win, (255, 0, 0), (player_x, player_y, player_width, player_height)
    )
    pygame.display.update()
    win.fill((0, 0, 0))


# Quit game and release camera
pygame.quit()
camera.release()
