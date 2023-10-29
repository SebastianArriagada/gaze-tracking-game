import time
import pygame
import cv2 as cv
import numpy as np
import math
import dlib

# Initialize Pygame
pygame.init()
win_width, win_height = 640, 480
win = pygame.display.set_mode((win_width, win_height))

# Initialize camera
camera = cv.VideoCapture(0)

# Initialize game variables
player_width, player_height = 50, 50
player_y = win_height - 60
possible_positions = [win_width // 4, win_width // 2, 3 * win_width // 4]
player_x = possible_positions[1]
blue_boxes = []

# Dlib setup
detectFace = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Color constants
GREEN = (0, 255, 0)
ORANGE = (0, 69, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 247, 255)
MAGENTA = (255, 0, 242)
LIGHT_CYAN = (255, 204, 0)


# Convert OpenCV image to Pygame surface
def cv2_to_pygame(cv_image):
    cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
    shape = cv_image.shape[1::-1]
    pygame_image = pygame.image.frombuffer(cv_image.tobytes(), shape, "RGB")
    return pygame_image


# Generate blue box at random position
def generate_blue_box():
    pos = possible_positions[np.random.randint(0, 3)]
    return [pos, 0, 20, 20]


# Midpoint between two points
def midpoint(pts1, pts2):
    x, y = pts1
    x1, y1 = pts2
    xOut = int((x + x1) / 2)
    yOut = int((y1 + y) / 2)
    return (xOut, yOut)


# Face detector function
def faceDetector(image, gray, Draw=True):
    cordFace1 = (0, 0)
    cordFace2 = (0, 0)
    # getting faces from face detector
    faces = detectFace(gray)

    face = None
    # looping through All the face detected.
    for face in faces:
        # getting coordinates of face.
        cordFace1 = (face.left(), face.top())
        cordFace2 = (face.right(), face.bottom())

        # draw rectangle if draw is True.
        if Draw == True:
            cv.rectangle(image, cordFace1, cordFace2, GREEN, 2)
    return image, face


# Facial landmarks detector function
def faceLandmakDetector(image, gray, face, Draw=True):
    # calling the landmarks predictor
    landmarks = predictor(gray, face)
    pointList = []
    # looping through each landmark
    for n in range(0, 68):
        point = (landmarks.part(n).x, landmarks.part(n).y)
        # getting x and y coordinates of each mark and adding into list.
        pointList.append(point)
        # draw if draw is True.
        if Draw == True:
            # draw circle on each landmark
            cv.circle(image, point, 3, ORANGE, 1)
    return image, pointList


# Eye tracking function
def EyeTracking(image, gray, eyePoints):
    # getting dimensions of image
    dim = gray.shape
    # creating mask .
    mask = np.zeros(dim, dtype=np.uint8)

    # converting eyePoints into Numpy arrays.
    PollyPoints = np.array(eyePoints, dtype=np.int32)
    # Filling the Eyes portion with WHITE color.
    cv.fillPoly(mask, [PollyPoints], 255)

    # Writing gray image where color is White  in the mask using Bitwise and operator.
    eyeImage = cv.bitwise_and(gray, gray, mask=mask)

    # getting the max and min points of eye inorder to crop the eyes from Eye image .

    maxX = (max(eyePoints, key=lambda item: item[0]))[0]
    minX = (min(eyePoints, key=lambda item: item[0]))[0]
    maxY = (max(eyePoints, key=lambda item: item[1]))[1]
    minY = (min(eyePoints, key=lambda item: item[1]))[1]

    # other then eye area will black, making it white
    eyeImage[mask == 0] = 255

    # cropping the eye form eyeImage.
    cropedEye = eyeImage[minY:maxY, minX:maxX]

    # getting width and height of cropedEye
    height, width = cropedEye.shape

    # New divPart calculations based on 0.4-0.2-0.4 ratio
    right_div = int(width * 0.36)
    center_div = int(width * 0.3)
    left_div = int(width * 0.3)

    #  applying the threshold to the eye .
    ret, thresholdEye = cv.threshold(cropedEye, 100, 255, cv.THRESH_BINARY)

    # dividing the eye into Three parts .
    rightPart = thresholdEye[0:height, 0:right_div]
    centerPart = thresholdEye[0:height, right_div : right_div + center_div]
    leftPart = thresholdEye[0:height, right_div + center_div : width]

    # counting Black pixel in each part using numpy.
    rightBlackPx = np.sum(rightPart == 0)
    centerBlackPx = np.sum(centerPart == 0)
    leftBlackPx = np.sum(leftPart == 0)

    pos, color = Position([rightBlackPx, centerBlackPx, leftBlackPx])

    return mask, pos, color


# Determine eye position
def Position(ValuesList):
    maxIndex = ValuesList.index(max(ValuesList))
    posEye = ""
    color = [WHITE, BLACK]
    if maxIndex == 0:
        posEye = "Right"
        color = [YELLOW, BLACK]
    elif maxIndex == 1:
        posEye = "Center"
        color = [BLACK, MAGENTA]
    elif maxIndex == 2:
        posEye = "Left"
        color = [LIGHT_CYAN, BLACK]
    else:
        posEye = "Eye Closed"
        color = [BLACK, WHITE]
    return posEye, color
